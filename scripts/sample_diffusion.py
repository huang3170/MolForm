import argparse
import os
import shutil
import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.transforms import Compose
from torch_scatter import scatter_sum, scatter_mean
from tqdm.auto import tqdm
import utils.misc as misc
import utils.transforms as trans
from datasets import get_dataset
from datasets.pl_data import FOLLOW_BATCH
from models.molopt_score_model import ScorePosNet3D
from utils.evaluation import atom_num
import torch.nn.functional as F
from utils import misc


def _regularize_step_probs(step_probs, predict_ligand_v):
        num_atoms, num_classes = step_probs.shape
        device = step_probs.device
        assert predict_ligand_v.shape == (num_atoms, )

        step_probs = torch.clamp(step_probs, min=0.0, max=1.0)
        step_probs[
            torch.arange(num_atoms, device=device),
            predict_ligand_v.long().flatten()
        ] = 0.0
        step_probs[
            torch.arange(num_atoms, device=device),
            predict_ligand_v.long().flatten()
        ] = 1.0 - torch.sum(step_probs, dim=-1).flatten()
        step_probs = torch.clamp(step_probs, min=0.0, max=1.0)
        return step_probs


def sample_diffusion_ligand(model:ScorePosNet3D, 
                            data, 
                            num_samples, 
                            num_step,
                            ligand_v_temp=0.01, 
                            ligand_v_noise=1.0,
                            sample_time_schedule='linear',
                            batch_size=16,
                            device='cuda:0',
                            center_pos_mode='protein',
                            sample_num_atoms='prior'):
    # dt = 1.0 / num_step
    if sample_time_schedule == 'log':
        time_points = (1 - np.geomspace(0.01, 1.0, num_step + 1)).tolist()
    elif sample_time_schedule == 'linear':
        time_points = np.linspace(0.01, 1.0, num_step + 1).tolist()
    else:
        raise ValueError(f"Invalid sample_time_schedule: {sample_time_schedule}")   
    time_points.reverse()

    all_pred_pos, all_pred_v = [], []
    all_pred_pos_traj, all_pred_v_traj = [], []
    num_batch = int(np.ceil(num_samples / batch_size))
    current_i = 0
    for i in tqdm(range(num_batch)):
        n_data = batch_size if i < num_batch - 1 else num_samples - batch_size * (num_batch - 1)
        
        batch = Batch.from_data_list([data.clone() for _ in range(n_data)], follow_batch=FOLLOW_BATCH).to(device)
        pred_pos_traj_batch, pred_v_traj_batch = [], []
        with torch.no_grad():
            batch_protein = batch.protein_element_batch
            if sample_num_atoms == 'prior':
                pocket_size = atom_num.get_space_size(data.protein_pos.detach().cpu().numpy())
                ligand_num_atoms = [atom_num.sample_atom_num(pocket_size).astype(int) for _ in range(n_data)]
                batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)
            elif sample_num_atoms == 'range':
                ligand_num_atoms = list(range(current_i + 1, current_i + n_data + 1))
                batch_ligand = torch.repeat_interleave(torch.arange(n_data), torch.tensor(ligand_num_atoms)).to(device)
            elif sample_num_atoms == 'ref':
                batch_ligand = batch.ligand_element_batch
                ligand_num_atoms = scatter_sum(torch.ones_like(batch_ligand), batch_ligand, dim=0).tolist()
            else:
                raise ValueError

            # init ligand pos
            center_pos = scatter_mean(batch.protein_pos, batch_protein, dim=0)
                        
            # Initialize randn_init values
            init_ligand_pos = torch.randn(center_pos[batch_ligand].shape).to(device)
            init_ligand_v = torch.randint(0, model.num_classes,(batch_ligand.shape[0], )).to(device)
            init_ligand_v_onehot = F.one_hot(init_ligand_v, num_classes=model.num_classes).to(device)

            batch.protein_pos = batch.protein_pos - center_pos[batch_protein]

            ligand_pos = init_ligand_pos.clone()
            ligand_v_onehot = init_ligand_v_onehot.clone().float()
            ligand_v_t = init_ligand_v.clone()
           
            for i, t in enumerate(time_points):
                if i == 0:
                    continue
                else:
                    dt = time_points[i] - time_points[i-1]
                with torch.no_grad():           
                    r = model.sample_diffusion(
                        protein_pos=batch.protein_pos,
                        protein_v=batch.protein_atom_feature.float(),
                        batch_protein=batch_protein,
                        ligand_pos=ligand_pos,
                        ligand_v=ligand_v_onehot,
                        batch_ligand=batch_ligand,
                        t=t,
                        device=device,
                        center_pos_mode=center_pos_mode
                    )
                    pred_pos, pred_v = r['pred_pos'], r['pred_v']
                    # update positions
                    velocity_pos = (pred_pos - ligand_pos)/(1-t)
                    ligand_pos +=  velocity_pos * dt

                    # update atomtype
                    pred_v_probs = F.softmax(pred_v / ligand_v_temp, dim=-1) # (L, S)
                    pt_x1_eq_xt_prob = torch.gather(pred_v_probs, dim=-1, index=ligand_v_t.long().unsqueeze(-1)) # (B, D, 1)

                    N = ligand_v_noise
                    step_probs = dt * (pred_v_probs * ((1 + N + N * (model.num_classes - 1) * t) / (1-t)) + N * pt_x1_eq_xt_prob )
                    step_probs = _regularize_step_probs(step_probs, ligand_v_t)

                    ligand_v_t = torch.multinomial(step_probs.view(-1, model.num_classes), num_samples=1).view(step_probs.shape[0], )
                    ligand_v_onehot = F.one_hot(ligand_v_t, num_classes=model.num_classes).to(device)

                    pred_pos_traj_batch.append(ligand_pos+center_pos[batch_ligand])
                    pred_v_traj_batch.append(ligand_v_onehot)

            r = model.sample_diffusion(
                        protein_pos=batch.protein_pos,
                        protein_v=batch.protein_atom_feature.float(),
                        batch_protein=batch_protein,
                        ligand_pos=ligand_pos,
                        ligand_v=ligand_v_onehot,
                        batch_ligand=batch_ligand,
                        t=1.0,
                        device=device,
                        center_pos_mode=center_pos_mode
                    )
            ligand_pos, ligand_v = r['pred_pos'], r['pred_v']
            ligand_pos = ligand_pos + center_pos[batch_ligand]
            
            # unbatch ligand_pos and ligand_v
            ligand_cum_atoms = np.cumsum([0] + ligand_num_atoms)
            ligand_pos_array = ligand_pos.cpu().numpy().astype(np.float64)
            all_pred_pos += [ligand_pos_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in range(n_data)]
            ligand_v_array = ligand_v.cpu().numpy()
            all_pred_v += [ligand_v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in range(n_data)]

            # unbatch pred_pos_traj_batch and pred_v_traj_batch
            pred_pos_traj_batch.append(ligand_pos)
            pred_v_traj_batch.append(ligand_v)
            pred_pos_traj_batch = torch.stack(pred_pos_traj_batch, dim=0).cpu().numpy().astype(np.float64)
            pred_v_traj_batch = torch.stack(pred_v_traj_batch, dim=0).cpu().numpy()
            all_pred_pos_traj += [pred_pos_traj_batch[:, ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in range(n_data)]
            all_pred_v_traj += [pred_v_traj_batch[:, ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in range(n_data)]


            current_i += n_data
            
                    
         
                            
    return all_pred_pos, all_pred_v, all_pred_pos_traj, all_pred_v_traj

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('-i', '--data_id', type=int)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--result_path', type=str, default='./outputs')
    args = parser.parse_args()

    logger = misc.get_logger('sampling')
    
    ## creat output dir
    log_dir = args.result_path
    os.makedirs(log_dir, exist_ok=True)


    # Load config
    config = misc.load_config(args.config)
    logger.info(config)
    result_path = args.result_path
    os.makedirs(result_path, exist_ok=True)
    misc.seed_all(config.sample.seed)

    # Load checkpoint
    ckpt = torch.load(config.model.checkpoint, map_location=args.device)
    
    logger.info(f"Training Config: {ckpt['config']}")

    # Transforms
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_atom_mode = ckpt['config'].data.transform.ligand_atom_mode
    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
    bond_featureizer = trans.FeaturizeLigandBond()

    transform = Compose([
        protein_featurizer,
        ligand_featurizer,
        bond_featureizer,
    ])
    # Load dataset
    dataset, subsets = get_dataset(
        config=ckpt['config'].data,
        transform=transform
    )
    train_set, test_set = subsets['train'], subsets['test']
    logger.info(f'Successfully load the dataset (size: {len(test_set)})!')

    new_state_dict = {k.replace('module.', ''): v for k, v in ckpt['model'].items()}

    model = ScorePosNet3D(
            ckpt['config'].model,
            protein_atom_feature_dim=protein_featurizer.feature_dim,
            ligand_atom_feature_dim=ligand_featurizer.feature_dim,
        ).to(args.device)

    model.load_state_dict(new_state_dict)
    print("test set length: ", len(test_set))
    print("data id: ", args.data_id)
    print(test_set[args.data_id])
    logger.info(f'Successfully load the model! {config.model.checkpoint}')

    data = test_set[args.data_id]
   
    data_ligand_pos = data.ligand_pos
    num_step = config.sample.num_steps

    pred_pos, pred_v, pred_pos_traj, pred_v_traj = sample_diffusion_ligand(model=model, 
                                               data=data, 
                                               num_samples=config.sample.num_samples,
                                               num_step = num_step,
                                               batch_size=args.batch_size, 
                                               ligand_v_temp=config.sample.ligand_v_temp,
                                               ligand_v_noise=config.sample.ligand_v_noise,
                                               sample_time_schedule=config.sample.sample_time_schedule,
                                               device=args.device,
                                                center_pos_mode=config.sample.center_pos_mode,
                                                sample_num_atoms=config.sample.sample_num_atoms  
    )

    max_indices = [
    np.argmax(array, axis=1) for array in pred_v]
    max_indices_traj = [
    np.argmax(array, axis=-1) for array in pred_v_traj]

    
    result = {
        'data': data,
        'pred_ligand_pos': pred_pos,
        'pred_ligand_v': max_indices,
        'pred_ligand_pos_traj': pred_pos_traj,
        'pred_ligand_v_traj': max_indices_traj
    }
    torch.save(result, os.path.join(result_path, f'result_{args.data_id}.pt'))
 

    logger.info('Sample done!')

    
    shutil.copyfile(args.config, os.path.join(result_path, 'sample.yml'))
    shutil.copyfile("./scripts/sample_diffusion.py", os.path.join(result_path, "sample_diffusion.py"))
    