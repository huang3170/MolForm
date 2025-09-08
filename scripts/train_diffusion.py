import argparse
import os
import shutil
import socket
import pickle
from scripts.evaluate_diffusion import run_eval_process
from scripts.sample_diffusion import sample_diffusion_ligand
import wandb
import numpy as np
import math

import torch
import torch.distributed as distrib
# import torch.utils.tensorboard
from sklearn.metrics import roc_auc_score
from torch.nn.utils import clip_grad_norm_
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from tqdm.auto import tqdm
from torch.utils.data import Subset

import utils.misc as misc
import utils.train as utils_train
import utils.transforms as trans
from datasets import get_dataset
from datasets.pl_data import FOLLOW_BATCH
from datasets.merge_dataset import MergedProteinLigandData, FOLLOW_BATCH2
from models.molopt_score_model import ScorePosNet3D
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


def get_auroc(y_true, y_pred, feat_mode):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    avg_auroc = 0.0
    possible_classes = set(y_true)
    for c in possible_classes:
        auroc = roc_auc_score(y_true == c, y_pred[:, c])
        avg_auroc += auroc * np.sum(y_true == c)
        mapping = {
            "basic": trans.MAP_INDEX_TO_ATOM_TYPE_ONLY,
            "add_aromatic": trans.MAP_INDEX_TO_ATOM_TYPE_AROMATIC,
            "full": trans.MAP_INDEX_TO_ATOM_TYPE_FULL,
        }
        print(f"atom: {mapping[feat_mode][c]} \t auc roc: {auroc:.4f}")
    return avg_auroc / len(y_true)


def is_port_available(port, host="localhost"):
    """
    Check if a given port is available.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind((host, port))
        s.close()
        return True
    except OSError:
        s.close()
        return False


def main(rank, num_gpus):
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--logdir", type=str, default="./logs_diffusion")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--train_report_iter", type=int, default=100)
    parser.add_argument(
        "--is_debug", action="store_true", help="Enable debug mode", default=False
    )
    parser.add_argument("--name", type=str, default="flow matching")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Path to a previous checkpoint",
    )
    parser.add_argument(
        "--dpo_data",
        type=str,
        default="",
        help="Path to DPO preference data file (pkl format). If provided, enables DPO training."
    )
    args = parser.parse_args()


    # Version control
    branch, version = misc.get_version()
    version_short = "%s-%s" % (branch, version[:7])

    # Load configs
    config = misc.load_config(args.config)
    config_name = os.path.basename(args.config)[
        : os.path.basename(args.config).rfind(".")
    ]
    misc.seed_all(config.train.seed)

    # Logging 
    if args.is_debug and rank == 0:
        logger = misc.get_logger("train", None)
    elif rank == 0:
        run = wandb.init(
            project=args.name, config=config, name=f"{config_name}[{args.tag}]"
        )

        log_dir = misc.get_new_log_dir(
            args.logdir, prefix=f"{config_name}_{version_short}_{args.tag}"
        )

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        with open(os.path.join(log_dir, "commit.txt"), "w") as f:
            f.write(branch + "\n")
            f.write(version + "\n")
        logger = misc.get_logger("train", log_dir)
        ckpt_dir = os.path.join(log_dir, "checkpoints")
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir, exist_ok=True)

        shutil.copyfile(
            args.config, os.path.join(log_dir, os.path.basename(args.config))
        )
        shutil.copytree("./models", os.path.join(log_dir, "models"))

        logger.info(args)
        logger.info(config)

    torch.cuda.set_device(rank)
    distrib.init_process_group(
        backend="nccl", rank=rank, world_size=num_gpus, init_method="env://"
    )

    # Transforms
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_featurizer = trans.FeaturizeLigandAtom(
        config.data.transform.ligand_atom_mode
    )
    bond_featureizer = trans.FeaturizeLigandBond()
    transform_list = [
        protein_featurizer,
        ligand_featurizer,
        bond_featureizer,
    ]
    if config.data.transform.random_rot:
        transform_list.append(trans.RandomRotation())
    transform = Compose(transform_list)

    # Datasets and loaders
    logger.info("Loading dataset...") if rank == 0 else None
    dataset, subsets = get_dataset(config=config.data, transform=transform)
    train_set, val_set = subsets["train"], subsets["test"]

    # Check if DPO training is enabled
    if config.train.use_dpo:
        logger.info("DPO training enabled. Loading preference data...") if rank == 0 else None

        def get_dpo_data(subset, split='train', fullset=dataset):
            logger.info('Getting DPO {} data...'.format(split)) if rank == 0 else None

            with open(args.dpo_data, 'rb') as file:
                dpo_idx = pickle.load(file)

            train_id = subset.indices
            train_id_set = set(train_id)
            cleaned_dpo_idx = {k: [item for item in v if item in train_id_set] for k, v in dpo_idx.items() if v}
            cleaned_dpo_idx = {k: v for k, v in cleaned_dpo_idx.items() if v}  # Remove any keys with empty lists

            new_pair = {}
            for idx in tqdm(subset.indices, 'Creating Preference Train Set'):
                if idx in cleaned_dpo_idx.keys():
                    losing_id = cleaned_dpo_idx[idx]
                    assert losing_id
                    new_pair[idx] = losing_id[0]


            winning_idx = list(new_pair.keys())
            losing_idx = list(new_pair.values())
            subset_1 = Subset(fullset, winning_idx)
            subset_2 = Subset(fullset, losing_idx)

            dpo_subset = MergedProteinLigandData(subset_1, subset_2)
            return dpo_subset

        dpo_train_set = get_dpo_data(train_set, split='train', fullset=dataset)

        train_sampler = DistributedSampler(
            dpo_train_set, num_replicas=num_gpus, rank=rank, shuffle=False, drop_last=False
        )

        logger.info(f"DPO Training: {len(dpo_train_set)} Validation: {len(val_set)}") if rank == 0 else None

        collate_exclude_keys = ['ligand_nbh_list', 'ligand_nbh_list2']
        follow_batch = FOLLOW_BATCH2
        actual_train_set = dpo_train_set
    else:
        logger.info("Standard training enabled.") if rank == 0 else None
        train_sampler = DistributedSampler(
            train_set, num_replicas=num_gpus, rank=rank, shuffle=False, drop_last=False
        )
        logger.info(f"Training: {len(train_set)} Validation: {len(val_set)}") if rank == 0 else None
        collate_exclude_keys = ["ligand_nbh_list"]
        follow_batch = FOLLOW_BATCH
        actual_train_set = train_set

    train_iterator = utils_train.inf_iterator(
        DataLoader(
            actual_train_set,
            batch_size=config.train.batch_size,
            sampler=train_sampler,
            num_workers=config.train.num_workers,
            follow_batch=follow_batch,
            exclude_keys=collate_exclude_keys,
        )
    )

    if rank == 0:
        val_loader = DataLoader(
            val_set,
            config.train.batch_size,
            shuffle=False,
            follow_batch=FOLLOW_BATCH,
            exclude_keys=collate_exclude_keys,
        )

    # Model
    logger.info("Building model...") if rank == 0 else None

    model = ScorePosNet3D(
        config.model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim,
    ).cuda(rank)

    # Load reference model for DPO training if needed
    ref_model = None
    if config.train.use_dpo:
        if hasattr(config.model, 'ref_model_checkpoint') and config.model.ref_model_checkpoint:
            logger.info("Loading reference model for DPO training...") if rank == 0 else None
            ckpt = torch.load(config.model.ref_model_checkpoint, map_location=f"cuda:{rank}")
            new_state_dict = {k.replace('module.', ''): v for k, v in ckpt['model'].items()}

            # Load weights into main model
            model.load_state_dict(new_state_dict)

            # Create reference model
            ref_model = ScorePosNet3D(
                config.model,
                protein_atom_feature_dim=protein_featurizer.feature_dim,
                ligand_atom_feature_dim=ligand_featurizer.feature_dim
            ).cuda(rank)
            ref_model.load_state_dict(new_state_dict)
            ref_model.eval()

            # Freeze reference model parameters
            for param in ref_model.parameters():
                param.requires_grad = False
            logger.info("Reference model loaded and frozen.") if rank == 0 else None
        else:
            raise ValueError("DPO training requires ref_model_checkpoint in config.model")

    model = DDP(model, device_ids=[rank], output_device=rank)
    logger.info(f"protein feature dim: {protein_featurizer.feature_dim} ligand feature dim: {ligand_featurizer.feature_dim}") if rank == 0 else None
    logger.info(f"Model has {misc.count_parameters(model) / 1e6:.4f} M parameters.") if rank == 0 else None

    # Optimizer and scheduler
    optimizer = utils_train.get_optimizer(config.train.optimizer, model)
    scheduler = utils_train.get_scheduler(config.train.scheduler, optimizer)

    # Loading from checkpoint if provided
    if args.checkpoint is not None:
        if os.path.isfile(args.checkpoint):
            logger.info(f"Loading from checkpoint {args.checkpoint}") if rank == 0 else None
            checkpoint = torch.load(
                args.checkpoint, map_location=f"cuda:{rank}"
            )
            start_iter = checkpoint["iteration"]
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            logger.info(f"=> loaded checkpoint {args.checkpoint} (iteration {checkpoint['iteration']})") if rank == 0 else None
        else:
            logger.info(f"=> no checkpoint found at {args.checkpoint}") if rank == 0 else None
            start_iter = 1
    else:
        start_iter = 1

    def train(it):
        model.train()
        optimizer.zero_grad()
        for _ in range(config.train.n_acc_batch):

            batch = next(train_iterator).cuda(rank)

            if config.train.use_dpo:
                # DPO training
                protein_noise = torch.randn_like(batch.protein_pos) * config.train.pos_noise_std
                protein_noise2 = torch.randn_like(batch.protein_pos2) * config.train.pos_noise_std

                gt_protein_pos = batch.protein_pos + protein_noise
                gt_protein_pos2 = batch.protein_pos2 + protein_noise2

                results = model.module.get_diffusion_loss_dpo(
                    beta_dpo=config.train.beta_dpo,
                    discete_beta_dpo=config.train.discete_beta_dpo,
                    ref_model=ref_model,

                    protein_pos_w=gt_protein_pos,
                    protein_v_w=batch.protein_atom_feature.float(),
                    batch_protein_w=batch.protein_element_batch,
                    ligand_pos_w=batch.ligand_pos,
                    ligand_v_w=batch.ligand_atom_feature_full,
                    batch_ligand_w=batch.ligand_element_batch,

                    protein_pos_l=gt_protein_pos2,
                    protein_v_l=batch.protein_atom_feature2.float(),
                    batch_protein_l=batch.protein_element2_batch,
                    ligand_pos_l=batch.ligand_pos2,
                    ligand_v_l=batch.ligand_atom_feature_full2,
                    batch_ligand_l=batch.ligand_element2_batch,
                )

                loss, loss_pos, loss_v, chamfer_loss = (
                    results["loss"],
                    results["loss_pos"],
                    results["loss_v"],
                    results["loss_chamfer"],
                )
            else:
                # Standard training
                protein_noise = (
                    torch.randn_like(batch.protein_pos) * config.train.pos_noise_std
                )
                gt_protein_pos = batch.protein_pos + protein_noise
                results = model.module.get_diffusion_loss(
                    protein_pos=gt_protein_pos,
                    protein_v=batch.protein_atom_feature.float(),
                    batch_protein=batch.protein_element_batch,
                    ligand_pos=batch.ligand_pos,
                    ligand_v=batch.ligand_atom_feature_full,
                    batch_ligand=batch.ligand_element_batch,
                )

                loss, loss_pos, loss_v, chamfer_loss = (
                    results["loss"],
                    results["loss_pos"],
                    results["loss_v"],
                    results["chamfer_loss"],
                )

            loss = loss / config.train.n_acc_batch
            loss.backward()
        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
        optimizer.step()

        if rank==0 and it % args.train_report_iter == 0:
            logger.info(
                "[Train] Iter %d | Loss %.6f (pos %.6f | v %.6f | chamfer %.6f) | Lr: %.6f | Grad Norm: %.6f"
                % (
                    it,
                    loss,
                    loss_pos,
                    loss_v,
                    chamfer_loss,
                    optimizer.param_groups[0]["lr"],
                    orig_grad_norm,
                )
            )
            # Add wandb logging
            if not args.is_debug and rank == 0:
                wandb.log(
                    {
                        "train/loss": loss,
                        "train/loss_pos": loss_pos,
                        "train/loss_v": loss_v,
                        "train/chamfer_loss": chamfer_loss,
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/grad_norm": orig_grad_norm,
                        "iteration": it,
                    }
                )

    def validate(it):
        # fix time steps
        sum_loss, sum_loss_pos, sum_loss_v, sum_chamfer_loss, sum_n = 0, 0, 0, 0, 0
        all_pred_ligand_typ = []
        all_pred_v, all_true_v = [], []
        with torch.no_grad():
            model.eval()
            for batch in tqdm(val_loader, desc="Validate"):

                batch = batch.cuda(rank)
                batch_size = batch.num_graphs
                results = model.module.get_diffusion_loss(
                    protein_pos=batch.protein_pos,
                    protein_v=batch.protein_atom_feature.float(),
                    batch_protein=batch.protein_element_batch,
                    ligand_pos=batch.ligand_pos,
                    ligand_v=batch.ligand_atom_feature_full,
                    batch_ligand=batch.ligand_element_batch,
                )
                loss, loss_pos, loss_v, chamfer_loss = (
                    results["loss"],
                    results["loss_pos"],
                    results["loss_v"],
                    results["chamfer_loss"],
                )
                sum_loss += float(loss) * batch_size
                sum_loss_pos += float(loss_pos) * batch_size
                sum_loss_v += float(loss_v) * batch_size
                sum_chamfer_loss += float(chamfer_loss) * batch_size
                sum_n += batch_size
                all_pred_v.append(results["ligand_v_recon"].detach().cpu().numpy())
                all_true_v.append(batch.ligand_atom_feature_full.detach().cpu().numpy())
                all_pred_ligand_typ.append(
                    results["pred_ligand_v"].detach().cpu().numpy()
                )

        avg_loss = sum_loss / sum_n
        avg_loss_pos = sum_loss_pos / sum_n
        avg_loss_v = sum_loss_v / sum_n
        avg_chamfer_loss = sum_chamfer_loss / sum_n

        atom_auroc = None  # Initialize atom_auroc
        try:
            atom_auroc = get_auroc(
                np.concatenate(all_true_v),
                np.concatenate(all_pred_v, axis=0),
                feat_mode=config.data.transform.ligand_atom_mode,
            )
        except Exception as e:
            logger.info(f"An error occurred while calculating AUROC: {e}")
            logger.info("pred_v has Nan")
            logger.info(np.concatenate(all_pred_ligand_typ, axis=0))
        # Add wandb logging
        if not args.is_debug:
            log_dict = {
                "val/loss": avg_loss,
                "val/loss_pos": avg_loss_pos,
                "val/loss_v": avg_loss_v,
                "val/chamfer_loss": avg_chamfer_loss,
                "iteration": it,
            }
            if atom_auroc is not None:
                log_dict["val/atom_auroc"] = atom_auroc
            wandb.log(log_dict)
        # Ensure that atom_auroc is not None before logging it
        if atom_auroc is not None:
            logger.info(
                "[Validate] Iter %05d | Loss %.6f | Loss pos %.6f | Loss v %.6f | chamfer %.6f |Avg atom auroc %.6f"
                % (it, avg_loss, avg_loss_pos, avg_loss_v, avg_chamfer_loss, atom_auroc)
            )
        else:
            logger.info(
                "[Validate] Iter %05d | Loss %.6f | Loss pos %.6f | Loss v %.6f | chamfer %.6f | Avg atom auroc calculation failed"
                % (it, avg_loss, avg_loss_pos, avg_chamfer_loss, avg_loss_v)
            )
        return avg_loss

    try:
        best_loss, best_iter = None, None
        best_vina_score = None
        for it in range(start_iter, config.train.max_iters + 1):
            train(it)
            if (rank == 0) and (it % 100 == 0 or it == config.train.max_iters):
                val_loss = validate(it)
                if not args.is_debug:
                    final_ckpt = os.path.join(ckpt_dir, "final.pt")
                    torch.save(
                            {
                                "config": config,
                                "model": model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "scheduler": scheduler.state_dict(),
                                "iteration": it,
                            },
                            final_ckpt,
                        )
                if best_loss is None or val_loss < best_loss:
                    logger.info(f"[Validate] Best val loss achieved: {val_loss:.6f}")
                    best_loss, best_iter = val_loss, it
                    if args.is_debug:
                        continue
                    ckpt_path = os.path.join(ckpt_dir, "%d.pt" % it)
                    torch.save(
                        {
                            "config": config,
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "iteration": it,
                        },
                        ckpt_path,
                    )
                else:
                    logger.info(
                        f"[Validate] Val loss is not improved. "
                        f"Best val loss: {best_loss:.6f} at iter {best_iter}"
                    )
            
            if config.train.use_dpo and (rank == 0) and (it % config.train.eval_freq == 0 or it == config.train.max_iters):
                num_eval_pockets = config.train.num_eval_pockets
                num_samples_per_pocket = config.train.num_eval_samples_per_pocket
                protein_root = config.train.protein_root
                eval_results = []
                for eval_idx in range(num_eval_pockets):
                    pred_pos, pred_v, pred_pos_traj, pred_v_traj = sample_diffusion_ligand(
                        model=model.module, 
                        data=val_set[eval_idx], 
                        num_samples=num_samples_per_pocket,
                        batch_size=num_samples_per_pocket, 
                        device=f"cuda:{rank}",
                        num_step=100,
                        ligand_v_temp=0.1,
                        ligand_v_noise=1.0,
                        sample_time_schedule='log',
                        center_pos_mode='protein',
                        sample_num_atoms='prior'
                    )
                    max_indices = [np.argmax(array, axis=1) for array in pred_v]
                    result = {
                        'data': val_set[eval_idx],
                        'pred_ligand_pos': pred_pos,
                        'pred_ligand_v': max_indices,
                        "pred_ligand_pos_traj": pred_pos_traj,
                        "pred_ligand_v_traj": pred_v_traj,
                    }
                    eval_results.append(result)
                eval_result = run_eval_process(
                    results_list=eval_results,
                    atom_enc_mode='add_aromatic',
                    protein_root=protein_root,
                    exhaustiveness=16,
                    docking_mode="vina_score",
                    logger=logger,
                    multiprocess=False
                )
                vina_score = eval_result[-1]
                print(vina_score)
                if not math.isnan(vina_score) and (best_vina_score is None or math.isnan(best_vina_score) or vina_score < best_vina_score):
                    best_vina_score = vina_score
                    best_vina_iter = it
                    logger.info(f"[Validate] Best vina score achieved: {vina_score:.6f}")
                    if not args.is_debug:
                        ckpt_path = os.path.join(ckpt_dir, f"best_vina_{it}.pt")
                        torch.save(
                            {
                                "config": config,
                                "model": model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "scheduler": scheduler.state_dict(),
                                "iteration": it,
                            },
                            ckpt_path,
                        )
                else:
                    logger.info(
                        f"[Validate] Vina score is not improved. "
                        f"Best vina score: {best_vina_score:.6f} at iter {best_vina_iter}"
                    )
                if not args.is_debug:
                    log_dict = {
                    "vina_score": vina_score,
                    "iteration": it,
                    }
                    wandb.log(log_dict)
            
    except KeyboardInterrupt:
        logger.info("Terminating...")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    os.environ["MASTER_ADDR"] = "localhost"

    # Start from port 12355 and find an available port
    port = 12355
    while not is_port_available(port) and port < 13000:
        port += 1

    if port >= 13000:
        raise ValueError("Could not find an available port between 12355 and 13000")

    os.environ["MASTER_PORT"] = str(port)
    num_gpus = torch.cuda.device_count()
    print("num_gpus: ", num_gpus)
    torch.multiprocessing.spawn(main, args=(num_gpus,), nprocs=num_gpus, join=True)