import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
from models.common import compose_context, ShiftedSoftplus
from models.egnn import EGNN
from models.uni_transformer import UniTransformerO2TwoUpdateGeneral

def log_sample_categorical(logits):
    uniform = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
    sample_index = (gumbel_noise + logits).argmax(dim=-1)
    # sample_onehot = F.one_hot(sample, self.num_classes)
    # log_sample = index_to_log_onehot(sample, self.num_classes)
    return sample_index

class MLP_pos(nn.Module):
    def __init__(self, input_dim=3, hidden_num=128):
        super().__init__()

        self.fc1 = nn.Linear(input_dim + 1, hidden_num, bias=True)
        self.fc2 = nn.Linear(hidden_num, hidden_num, bias=True)
        self.fc3 = nn.Linear(hidden_num, input_dim, bias=True)
        self.act = lambda x: torch.tanh(x)

    def forward(self, x_input, t):
        t = t.to(x_input.device)
        inputs = torch.cat([x_input, t], dim=1)
        x = self.fc1(inputs)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        
        return x


class MLP_elem(nn.Module):
    def __init__(self, input_dim=25, hidden_num=128):
        super().__init__()

        self.fc1 = nn.Linear(input_dim + 1, hidden_num, bias=True)
        self.fc2 = nn.Linear(hidden_num, hidden_num, bias=True)
        self.fc3 = nn.Linear(hidden_num, input_dim, bias=True)
        self.act = lambda x: torch.tanh(x)

    def forward(self, x_input, t):
        t = t.to(x_input.device)
        inputs = torch.cat([x_input, t], dim=1)
        # inputs = torch.cat([x_input, t], dim=1)
        x = self.fc1(inputs)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)

        return x


def get_refine_net(refine_net_type, config):
    if refine_net_type == "uni_o2":
        refine_net = UniTransformerO2TwoUpdateGeneral(
            num_blocks=config.num_blocks,
            num_layers=config.num_layers,
            hidden_dim=config.hidden_dim,
            n_heads=config.n_heads,
            k=config.knn,
            edge_feat_dim=config.edge_feat_dim,
            num_r_gaussian=config.num_r_gaussian,
            num_node_types=config.num_node_types,
            act_fn=config.act_fn,
            norm=config.norm,
            cutoff_mode=config.cutoff_mode,
            ew_net_type=config.ew_net_type,
            num_x2h=config.num_x2h,
            num_h2x=config.num_h2x,
            r_max=config.r_max,
            x2h_out_fc=config.x2h_out_fc,
            sync_twoup=config.sync_twoup,
        )
    elif refine_net_type == "egnn":
        refine_net = EGNN(
            num_layers=config.num_layers,
            hidden_dim=config.hidden_dim,
            edge_feat_dim=config.edge_feat_dim,
            num_r_gaussian=1,
            k=config.knn,
            cutoff_mode=config.cutoff_mode,
        )
    else:
        raise ValueError(refine_net_type)
    return refine_net


def get_distance(pos, edge_index):
    return (pos[edge_index[0]] - pos[edge_index[1]]).norm(dim=-1)


def center_pos(protein_pos, ligand_pos, batch_protein, batch_ligand, mode="protein"):
    if mode == "none":
        offset = 0.0
        pass
    elif mode == "protein":
        offset = scatter_mean(protein_pos, batch_protein, dim=0)
        protein_pos = protein_pos - offset[batch_protein]
        ligand_pos = ligand_pos - offset[batch_ligand]
    else:
        raise NotImplementedError
    return protein_pos, ligand_pos, offset


# Model
class ScorePosNet3D(nn.Module):

    def __init__(self, config, protein_atom_feature_dim, ligand_atom_feature_dim):
        super().__init__()
        self.config = config

        # variance schedule
        self.model_mean_type = config.model_mean_type  # ['noise', 'C0']
        self.loss_v_weight = config.loss_v_weight
        self.chamfer_loss_weight = config.chamfer_loss_weight
        self.mlp_pos = MLP_pos(input_dim=3, hidden_num=config.hidden_dim)
        self.mlp_elem = MLP_elem(input_dim=ligand_atom_feature_dim, hidden_num=config.hidden_dim)

        self.hidden_dim = config.hidden_dim
        self.num_classes = ligand_atom_feature_dim
        if self.config.node_indicator:
            emb_dim = self.hidden_dim - 1
        else:
            emb_dim = self.hidden_dim

        # atom embedding
        self.protein_atom_emb = nn.Linear(protein_atom_feature_dim, emb_dim)

        # center pos
        self.center_pos_mode = config.center_pos_mode  # ['none', 'protein']

        self.ligand_atom_emb = nn.Linear(ligand_atom_feature_dim, emb_dim)
        self.refine_net_type = config.model_type
        self.refine_net = get_refine_net(self.refine_net_type, config)
        self.v_inference = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            ShiftedSoftplus(),
            nn.Linear(self.hidden_dim, ligand_atom_feature_dim),
        )

    def forward(
        self,
        protein_pos,
        protein_v,
        batch_protein,
        init_ligand_pos,
        init_ligand_v,
        batch_ligand,
        return_all=False,
        fix_x=False,
    ):


        input_ligand_feat = init_ligand_v
        h_protein = self.protein_atom_emb(protein_v)
        init_ligand_h = self.ligand_atom_emb(input_ligand_feat)

        if self.config.node_indicator:
            h_protein = torch.cat(
                [h_protein, torch.zeros(len(h_protein), 1).to(h_protein)], -1
            )
            init_ligand_h = torch.cat(
                [init_ligand_h, torch.ones(len(init_ligand_h), 1).to(h_protein)], -1
            )

        h_all, pos_all, batch_all, mask_ligand = compose_context(
            h_protein=h_protein,
            h_ligand=init_ligand_h,
            pos_protein=protein_pos,
            pos_ligand=init_ligand_pos,
            batch_protein=batch_protein,
            batch_ligand=batch_ligand,
        )

        outputs = self.refine_net(
            h_all, pos_all, mask_ligand, batch_all, return_all=return_all, fix_x=fix_x
        )
        final_pos, final_h = outputs["x"], outputs["h"]
        final_ligand_pos, final_ligand_h = final_pos[mask_ligand], final_h[mask_ligand]
        final_ligand_v = self.v_inference(final_ligand_h)

        preds = {
            "pred_ligand_pos": final_ligand_pos,
            "pred_ligand_v": final_ligand_v,
            "final_h": final_h,
            "final_ligand_h": final_ligand_h,
        }
        if return_all:
            final_all_pos, final_all_h = outputs["all_x"], outputs["all_h"]
            final_all_ligand_pos = [pos[mask_ligand] for pos in final_all_pos]
            final_all_ligand_v = [self.v_inference(h[mask_ligand]) for h in final_all_h]
            preds.update(
                {
                    "layer_pred_ligand_pos": final_all_ligand_pos,
                    "layer_pred_ligand_v": final_all_ligand_v,
                }
            )
        return preds
    
    def sample_mix_up02_beta(self,shape, device):
        p1, p2 = 1.9, 1.0
        dist = torch.distributions.beta.Beta(p1, p2)
        samples_beta = dist.sample(shape).to(device)
        samples_uniform = torch.rand(shape, device=device)
        u = torch.rand(shape, device=device)
        return torch.where(u < 0.02, samples_uniform, samples_beta)

    
    def modify_input_With_time(self, ligand_pos, ligand_v, batch_ligand):
        
        device = batch_ligand.device
        t_all = self.sample_mix_up02_beta((max(batch_ligand) + 1,), device)
        t = t_all[batch_ligand].unsqueeze(1)
        
        # corrupt pos
        noise_pos = torch.randn(ligand_pos.shape).to(device)
        t_ligand_pos = t * ligand_pos + (1 - t) * noise_pos

        # corrupt elem
        noise_elem = torch.randint(0, self.num_classes,(ligand_v.shape[0], )).to(device)
        elem_mask = torch.rand(ligand_v.shape[0], device=device) < 1 - t.squeeze()
        t_cls_ligand = ligand_v.clone()
        t_cls_ligand[elem_mask] = noise_elem[elem_mask]
        t_cls_ligand = F.one_hot(t_cls_ligand, self.num_classes).float()

        return  t_ligand_pos, t_cls_ligand, t

    def chamfer_distance(self, x, y, batch_ligand):
        
        chamfer_list = []
        num_mols = int(batch_ligand.max().item()) + 1
        for i in range(num_mols):
            x_i = x[batch_ligand == i]
            y_i = y[batch_ligand == i]
            if x_i.size(0) == 0 or y_i.size(0) == 0:
                continue 
            x_i = x_i.unsqueeze(1)  # [n1, 1, 3]
            y_i = y_i.unsqueeze(0)  # [1, n2, 3]
            dist = torch.norm(x_i - y_i, dim=2)  # [n1, n2]
            cd_xy = dist.min(dim=1)[0].mean()
            cd_yx = dist.min(dim=0)[0].mean()
            chamfer = cd_xy + cd_yx
            chamfer_list.append(chamfer)
        if len(chamfer_list) == 0:
            return torch.tensor(0.0, device=x.device)
        return torch.stack(chamfer_list).mean()

    def get_diffusion_loss(
        self,
        protein_pos,
        protein_v,
        batch_protein,
        ligand_pos,
        ligand_v,
        batch_ligand,
    ):

        protein_pos, ligand_pos, _ = center_pos(
            protein_pos,
            ligand_pos,
            batch_protein,
            batch_ligand,
            mode=self.center_pos_mode,
        )
        
        
        ligand_pos_xt, ligand_atom_feature_full_xt, t = self.modify_input_With_time(ligand_pos, ligand_v, batch_ligand)
        ligand_pos_xt = self.mlp_pos(ligand_pos_xt, t)
        ligand_atom_feature_full_xt = self.mlp_elem(ligand_atom_feature_full_xt, t)

        # 3. forward-pass NN, feed perturbed pos and v, output noise
        preds = self(
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,
            init_ligand_pos=ligand_pos_xt,
            init_ligand_v=ligand_atom_feature_full_xt,
            batch_ligand=batch_ligand,
        )

        pred_ligand_pos, pred_ligand_v = (
            preds["pred_ligand_pos"],
            preds["pred_ligand_v"],
        )

        loss_pos = F.mse_loss(ligand_pos, pred_ligand_pos, reduction="none").mean(dim=(0, 1))
        loss_v = F.cross_entropy(pred_ligand_v, ligand_v, reduction="none").mean()  #pred_ligand_v [#atom, num_classes]

        chamfer_loss = self.chamfer_distance(ligand_pos, pred_ligand_pos, batch_ligand)
        loss = loss_pos + loss_v * self.loss_v_weight + self.chamfer_loss_weight * chamfer_loss

        return {
            "loss_pos": loss_pos,
            "loss_v": loss_v,
            "chamfer_loss": chamfer_loss,
            "loss": loss,
            "x0": ligand_pos,
            "pred_ligand_pos": pred_ligand_pos,
            "pred_ligand_v": pred_ligand_v,
            "ligand_v_recon": F.softmax(pred_ligand_v, dim=-1),
        }

    @torch.no_grad()
    def sample_diffusion(
        self,
        protein_pos,
        protein_v,
        batch_protein,
        ligand_pos,
        ligand_v,
        batch_ligand,
        t,
        device,
        center_pos_mode=None,
    ):

        protein_pos, ligand_pos, offset = center_pos(
            protein_pos, ligand_pos, batch_protein, batch_ligand, mode=center_pos_mode
        )

        t_shape = ligand_pos.shape[0]
        t = torch.ones((t_shape, 1)).to(device) * t

        # pass input and t through the model
        ligand_pos = self.mlp_pos(ligand_pos, t)

        ligand_v = self.mlp_elem(ligand_v, t)

        preds = self(
            protein_pos=protein_pos,
            protein_v=protein_v,
            batch_protein=batch_protein,
            init_ligand_pos=ligand_pos,
            init_ligand_v=ligand_v,
            batch_ligand=batch_ligand,
        )
        ligand_pos, ligand_v = preds["pred_ligand_pos"], preds["pred_ligand_v"]
        ligand_pos = ligand_pos + offset[batch_ligand]
        return {"pred_pos": ligand_pos, "pred_v": ligand_v}

    def get_diffusion_loss_dpo(
        self,
        beta_dpo,
        discete_beta_dpo,
        ref_model,
        protein_pos_w,
        protein_v_w,
        batch_protein_w,
        ligand_pos_w,
        ligand_v_w,
        batch_ligand_w,
        protein_pos_l,
        protein_v_l,
        batch_protein_l,
        ligand_pos_l,
        ligand_v_l,
        batch_ligand_l,
    ):
        """
        DPO (Direct Preference Optimization) training loss.

        Args:
            beta_dpo: DPO beta parameter for continuous losses
            discete_beta_dpo: DPO beta parameter for discrete losses
            ref_model: Reference model for DPO
            *_w: Winning (preferred) sample data
            *_l: Losing (less preferred) sample data
        """

        protein_pos_w, ligand_pos_w, _ = center_pos(
            protein_pos_w,
            ligand_pos_w,
            batch_protein_w,
            batch_ligand_w,
            mode=self.center_pos_mode,
        )

        protein_pos_l, ligand_pos_l, _ = center_pos(
            protein_pos_l,
            ligand_pos_l,
            batch_protein_l,
            batch_ligand_l,
            mode=self.center_pos_mode,
        )

        # Process winning sample
        ligand_pos_xt_w, ligand_atom_feature_full_xt_w, t_w = self.modify_input_With_time(ligand_pos_w, ligand_v_w, batch_ligand_w)
        ligand_pos_xt_w = self.mlp_pos(ligand_pos_xt_w, t_w)
        ligand_atom_feature_full_xt_w = self.mlp_elem(ligand_atom_feature_full_xt_w, t_w)

        # Process losing sample
        ligand_pos_xt_l, ligand_atom_feature_full_xt_l, t_l = self.modify_input_With_time(ligand_pos_l, ligand_v_l, batch_ligand_l)
        ligand_pos_xt_l = self.mlp_pos(ligand_pos_xt_l, t_l)
        ligand_atom_feature_full_xt_l = self.mlp_elem(ligand_atom_feature_full_xt_l, t_l)

        # Forward pass for winning sample
        preds_w = self(
            protein_pos=protein_pos_w,
            protein_v=protein_v_w,
            batch_protein=batch_protein_w,
            init_ligand_pos=ligand_pos_xt_w,
            init_ligand_v=ligand_atom_feature_full_xt_w,
            batch_ligand=batch_ligand_w,
        )
        pred_ligand_pos_w, pred_ligand_v_w = (
            preds_w["pred_ligand_pos"],
            preds_w["pred_ligand_v"],
        )

        # Forward pass for losing sample
        preds_l = self(
            protein_pos=protein_pos_l,
            protein_v=protein_v_l,
            batch_protein=batch_protein_l,
            init_ligand_pos=ligand_pos_xt_l,
            init_ligand_v=ligand_atom_feature_full_xt_l,
            batch_ligand=batch_ligand_l,
        )
        pred_ligand_pos_l, pred_ligand_v_l = (
            preds_l["pred_ligand_pos"],
            preds_l["pred_ligand_v"],
        )

        # Reference model predictions (no gradients)
        with torch.no_grad():
            ref_preds_w = ref_model.forward(
                protein_pos=protein_pos_w,
                protein_v=protein_v_w,
                batch_protein=batch_protein_w,
                init_ligand_pos=ligand_pos_xt_w,
                init_ligand_v=ligand_atom_feature_full_xt_w,
                batch_ligand=batch_ligand_w,
            )
            ref_pred_ligand_pos_w, ref_pred_ligand_v_w = (
                ref_preds_w["pred_ligand_pos"],
                ref_preds_w["pred_ligand_v"],
            )

            ref_preds_l = ref_model.forward(
                protein_pos=protein_pos_l,
                protein_v=protein_v_l,
                batch_protein=batch_protein_l,
                init_ligand_pos=ligand_pos_xt_l,
                init_ligand_v=ligand_atom_feature_full_xt_l,
                batch_ligand=batch_ligand_l,
            )
            ref_pred_ligand_pos_l, ref_pred_ligand_v_l = (
                ref_preds_l["pred_ligand_pos"],
                ref_preds_l["pred_ligand_v"],
            )

        # Position loss (DPO style)
        pos_w_diff = F.mse_loss(ligand_pos_w, pred_ligand_pos_w, reduction="none").mean(dim=(0, 1))
        pos_l_diff = F.mse_loss(ligand_pos_l, pred_ligand_pos_l, reduction="none").mean(dim=(0, 1))

        pos_w_diff_ref = F.mse_loss(ligand_pos_w, ref_pred_ligand_pos_w, reduction="none").mean(dim=(0, 1))
        pos_l_diff_ref = F.mse_loss(ligand_pos_l, ref_pred_ligand_pos_l, reduction="none").mean(dim=(0, 1))

        loss_pos = (pos_w_diff - pos_w_diff_ref) - (pos_l_diff - pos_l_diff_ref)
        loss_pos = torch.mean(-F.logsigmoid(-1 * beta_dpo * loss_pos))

        # Chamfer loss (DPO style)
        chamfer_loss_w = self.chamfer_distance(pred_ligand_pos_w, ligand_pos_w, batch_ligand_w)
        chamfer_loss_l = self.chamfer_distance(pred_ligand_pos_l, ligand_pos_l, batch_ligand_l)
        chamfer_loss_ref_w = self.chamfer_distance(ref_pred_ligand_pos_w, ligand_pos_w, batch_ligand_w)
        chamfer_loss_ref_l = self.chamfer_distance(ref_pred_ligand_pos_l, ligand_pos_l, batch_ligand_l)
        chamfer_loss = (chamfer_loss_w - chamfer_loss_ref_w) - (chamfer_loss_l - chamfer_loss_ref_l)
        chamfer_loss = torch.mean(-F.logsigmoid(-1 * beta_dpo * chamfer_loss))

        # Atom type loss (DPO style)
        # v loss
        x1_probs_w = F.softmax(pred_ligand_v_w, dim=-1) # (B, D, S)
        x1_probs_at_xt_w = torch.gather(x1_probs_w, dim=-1, index=ligand_v_w[:, None])
        x1_probs_l = F.softmax(pred_ligand_v_l, dim=-1) # (B, D, S)
        x1_probs_at_xt_l = torch.gather(x1_probs_l, dim=-1, index=ligand_v_l[:, None])

        x1_probs_ref_w = F.softmax(ref_pred_ligand_v_w, dim=-1) 
        x1_probs_at_xt_ref_w = torch.gather(x1_probs_ref_w, dim=-1, index=ligand_v_w[:, None])
        x1_probs_ref_l = F.softmax(ref_pred_ligand_v_l, dim=-1) 
        x1_probs_at_xt_ref_l = torch.gather(x1_probs_ref_l, dim=-1, index=ligand_v_l[:, None])

        loss_v_w = torch.mean(1/(1-t_w)*torch.log((x1_probs_at_xt_w/x1_probs_at_xt_ref_w)))
        loss_v_l = torch.mean(1/(1-t_l)*torch.log((x1_probs_at_xt_l/x1_probs_at_xt_ref_l)))

        loss_v = loss_v_w - loss_v_l
        loss_v = torch.mean(-F.logsigmoid(discete_beta_dpo * loss_v))

        # Total loss
        loss = loss_pos + loss_v * self.loss_v_weight + self.chamfer_loss_weight * chamfer_loss

        return {
            "loss_pos": loss_pos,
            "loss_v": loss_v,
            "loss_chamfer": chamfer_loss,
            "loss": loss,
        }
