import argparse
from multiprocessing import Pool
import os

import numpy as np
from rdkit import Chem
from rdkit import RDLogger
import torch
from tqdm.auto import tqdm
from glob import glob
from collections import Counter

from utils.evaluation import eval_atom_type, scoring_func, analyze, eval_bond_length
from utils import misc, reconstruct, transforms
from utils.evaluation.docking_qvina import QVinaDockingTask
from utils.evaluation.docking_vina import VinaDockingTask


def eval_sample(example_idx, sample_idx, pred_pos, pred_v, ligand_filename, 
                atom_enc_mode, protein_root, exhaustiveness, docking_mode, logger):
    
    # stability check
    recon_success = 0
    complete = 0
    eval_success = 0
    vina_results = None
    bond_dist = None
    dic = None

    pred_atom_type = transforms.get_atomic_number_from_index(pred_v, mode=atom_enc_mode)
    r_stable = analyze.check_stability(pred_pos, pred_atom_type)

    pair_dist = eval_bond_length.pair_distance_from_pos_v(pred_pos, pred_atom_type)
    # reconstruction
    try:
        pred_aromatic = transforms.is_aromatic_from_index(pred_v, mode=atom_enc_mode)
        mol = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic)
        smiles = Chem.MolToSmiles(mol)
    except reconstruct.MolReconsError:
        logger.warning('Reconstruct failed %s' % f'{example_idx}_{sample_idx}')
        return pred_atom_type, r_stable, pair_dist, recon_success, complete, eval_success,bond_dist, dic
    recon_success = 1

    if '.' in smiles:
        return pred_atom_type, r_stable, pair_dist, recon_success, complete, eval_success,bond_dist, dic
    complete = 1

    # chemical and docking check
    try:
        chem_results = scoring_func.get_chem(mol)
        if docking_mode == 'qvina':
            vina_task = QVinaDockingTask.from_generated_mol(
                mol, ligand_filename, protein_root=protein_root)
            vina_results = vina_task.run_sync()
        elif docking_mode in ['vina_score', 'vina_dock']:
            vina_task = VinaDockingTask.from_generated_mol(
                mol, ligand_filename, protein_root=protein_root)
            score_only_results = vina_task.run(mode='score_only', exhaustiveness=exhaustiveness)
            minimize_results = vina_task.run(mode='minimize', exhaustiveness=exhaustiveness)
            vina_results = {
                'score_only': score_only_results,
                'minimize': minimize_results
            }
            if docking_mode == 'vina_dock':
                docking_results = vina_task.run(mode='dock', exhaustiveness=exhaustiveness)
                vina_results['dock'] = docking_results
        else:
            vina_results = None

        eval_success = 1
    except Exception as error:
        
        logger.warning(error)
        logger.warning('Evaluation failed for %s' % f'{example_idx}_{sample_idx}')
        return pred_atom_type, r_stable, pair_dist, recon_success, complete, eval_success, bond_dist, dic

    # now we only consider complete molecules as success
    bond_dist = eval_bond_length.bond_distance_from_mol(mol)
    
    dic = {
        'mol': mol,
        'smiles': smiles,
        'ligand_filename': ligand_filename,
        'pred_pos': pred_pos,
        'pred_v': pred_v,
        'chem_results': chem_results,
        'vina': vina_results
    }
        
        
    return pred_atom_type, r_stable, pair_dist, recon_success, complete, eval_success, bond_dist, dic

def eval_process_wrapper(inputs):
    data, args = inputs
    example_idx, sample_idx, pred_pos, pred_v, ligand_filename = data
    atom_enc_mode, protein_root, exhaustiveness, docking_mode, logger = args
    return eval_sample(example_idx, sample_idx, pred_pos, pred_v, ligand_filename,
                       atom_enc_mode, protein_root, exhaustiveness, docking_mode, logger)

def run_eval_process(results_list, atom_enc_mode, protein_root, 
                     exhaustiveness, docking_mode, logger, multiprocess=True, num_processes=16):
    results = []
    num_samples, all_mol_stable, all_atom_stable, all_n_atom = 0,0,0,0
    n_recon_success, n_eval_success, n_complete = 0,0,0
    all_pair_dist, all_bond_dist, success_pair_dist = [], [], []
    all_atom_types = Counter()
    success_atom_types = Counter()

    
    for example_idx, sample_fn in tqdm(enumerate(results_list), total=len(results_list),desc='Evaluating'):
        if type(sample_fn) == str:
            sample = torch.load(sample_fn)
        else:
            sample = sample_fn
       
        pred_ligand_pos = sample['pred_ligand_pos']
        pred_ligand_v = sample['pred_ligand_v']
        ligand_filename = sample['data'].ligand_filename
        num_samples += len(pred_ligand_pos)

        
        sample_data = [
        (example_idx, sample_idx, pred_pos, pred_v, ligand_filename)
        for sample_idx, (pred_pos, pred_v) in enumerate(zip(pred_ligand_pos, pred_ligand_v))
        ]
    
        input_args = (atom_enc_mode, protein_root, exhaustiveness, docking_mode, logger)
        if multiprocess:
            partitioned_data = [(data, input_args) for data in sample_data]
            with Pool(min(len(partitioned_data), num_processes)) as pool:
                print(f"Pool is using {pool._processes} processes")
                results_data = list(tqdm(pool.imap(eval_process_wrapper, partitioned_data), total=len(partitioned_data)))
        else:
            results_data = [eval_process_wrapper((data, input_args)) for data in sample_data]

        for result in results_data:
    
            pred_atom_type, r_stable, pair_dist, recon_success, complete, eval_success, bond_dist, dic = result
            all_atom_types += Counter(pred_atom_type)
            all_mol_stable += r_stable[0]
            all_atom_stable += r_stable[1]
            all_n_atom += r_stable[2]
            all_pair_dist += pair_dist
            n_recon_success += recon_success
            n_complete += complete
            n_eval_success += eval_success
            
            if eval_success:
                all_bond_dist += bond_dist
                success_pair_dist += pair_dist
                success_atom_types += Counter(pred_atom_type)
                results.append(dic)

    logger.info(f'Evaluate done! {num_samples} samples in total.')

    fraction_mol_stable = all_mol_stable / num_samples
    fraction_atm_stable = all_atom_stable / all_n_atom
    fraction_recon = n_recon_success / num_samples
    fraction_eval = n_eval_success / num_samples
    fraction_complete = n_complete / num_samples
    validity_dict = {
        'mol_stable': fraction_mol_stable,
        'atm_stable': fraction_atm_stable,
        'recon_success': fraction_recon,
        'eval_success': fraction_eval,
        'complete': fraction_complete
    }

    vina, vina_score_only, vina_min, vina_dock = print_results(n_recon_success=n_recon_success, 
                n_complete=n_complete, 
                results=results, 
                logger=logger, 
                docking_mode=docking_mode)

    return results, all_bond_dist, success_pair_dist, success_atom_types, validity_dict, vina_score_only

def print_dict(d, logger):
    for k, v in d.items():
        if v is not None:
            logger.info(f'{k}:\t{v:.4f}')
        else:
            logger.info(f'{k}:\tNone')

def print_results(n_recon_success:float, n_complete:float, results:list, logger, docking_mode:str):
    
    logger.info('Number of reconstructed mols: %d, complete mols: %d, evaluated mols: %d' % (
    n_recon_success, n_complete, len(results)))

    vina = None
    vina_score_only = None
    vina_min = None
    vina_dock = None

    qed = [r['chem_results']['qed'] for r in results]
    sa = [r['chem_results']['sa'] for r in results]
    logger.info('QED:   Mean: %.3f Median: %.3f' % (np.mean(qed), np.median(qed)))
    logger.info('SA:    Mean: %.3f Median: %.3f' % (np.mean(sa), np.median(sa)))
    if docking_mode == 'qvina':
        vina = [r['vina'][0]['affinity'] for r in results]
        logger.info('Vina:  Mean: %.3f Median: %.3f' % (np.mean(vina), np.median(vina)))
        vina = np.mean(vina)
    elif docking_mode in ['vina_dock', 'vina_score']:
        vina_score_only = [r['vina']['score_only'][0]['affinity'] for r in results]
        vina_min = [r['vina']['minimize'][0]['affinity'] for r in results]
        logger.info('Vina Score:  Mean: %.3f Median: %.3f' % (np.mean(vina_score_only), np.median(vina_score_only)))
        logger.info('Vina Min  :  Mean: %.3f Median: %.3f' % (np.mean(vina_min), np.median(vina_min)))
        vina_score_only = np.mean(vina_score_only)
        vina_min = np.mean(vina_min)
        if docking_mode == 'vina_dock':
            vina_dock = [r['vina']['dock'][0]['affinity'] for r in results]
            logger.info('Vina Dock :  Mean: %.3f Median: %.3f' % (np.mean(vina_dock), np.median(vina_dock)))
            vina_dock = np.mean(vina_dock)
    
    return vina, vina_score_only, vina_min, vina_dock


def print_ring_ratio(all_ring_sizes, logger):
    for ring_size in range(3, 10):
        n_mol = 0
        for counter in all_ring_sizes:
            if ring_size in counter:
                n_mol += 1
        logger.info(f'ring size: {ring_size} ratio: {n_mol / len(all_ring_sizes):.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sample_path', type=str)
    parser.add_argument('--verbose', type=eval, default=False)
    parser.add_argument('--eval_step', type=int, default=-1)
    parser.add_argument('--eval_num_examples', type=int, default=None)
    parser.add_argument('--save', type=eval, default=True)
    parser.add_argument('--protein_root', type=str, default='./data/test_set')
    parser.add_argument('--atom_enc_mode', type=str, default='add_aromatic')
    parser.add_argument('--docking_mode', type=str, choices=['qvina', 'vina_score', 'vina_dock', 'none'])
    parser.add_argument('--exhaustiveness', type=int, default=16)
    parser.add_argument('--multiprocess', type=bool, default=True)
    parser.add_argument('--num_processes', type=int, default=16)
    args = parser.parse_args()

    result_path = os.path.join(args.sample_path, 'eval_results')
    os.makedirs(result_path, exist_ok=True)
    logger = misc.get_logger('evaluate', log_dir=result_path)
    if not args.verbose:
        RDLogger.DisableLog('rdApp.*')

    # Load generated data
    results_fn_list = glob(os.path.join(args.sample_path, '*result_*.pt'))
    results_fn_list = sorted(results_fn_list, key=lambda x: int(os.path.basename(x)[:-3].split('_')[-1]))
    if args.eval_num_examples is not None:
        results_fn_list = results_fn_list[:args.eval_num_examples]
    num_examples = len(results_fn_list)
    logger.info(f'Load generated data done! {num_examples} examples in total.')


    results, all_bond_dist, success_pair_dist, success_atom_types, validity_dict, vina_score_only \
        = run_eval_process(results_fn_list, args.atom_enc_mode, args.protein_root, \
                           args.exhaustiveness, args.docking_mode, logger, args.multiprocess, args.num_processes)

    c_bond_length_profile = eval_bond_length.get_bond_length_profile(all_bond_dist)
    c_bond_length_dict = eval_bond_length.eval_bond_length_profile(c_bond_length_profile)
    success_pair_length_profile = eval_bond_length.get_pair_length_profile(success_pair_dist)
    success_js_metrics = eval_bond_length.eval_pair_length_profile(success_pair_length_profile)
    atom_type_js = eval_atom_type.eval_atom_type_distribution(success_atom_types)

    print_dict(validity_dict, logger)

    logger.info('JS bond distances of complete mols: ')
    print_dict(c_bond_length_dict, logger)
    print_dict(success_js_metrics, logger)

    logger.info('Atom type JS: %.4f' % atom_type_js)

    if args.save:
        eval_bond_length.plot_distance_hist(success_pair_length_profile,
                                            metrics=success_js_metrics,
                                            save_path=os.path.join(result_path, f'pair_dist_hist_{args.eval_step}.png'))

    

    # check ring distribution
    print_ring_ratio([r['chem_results']['ring_size'] for r in results], logger)

    if args.save:
        torch.save({
            'stability': validity_dict,
            'bond_length': all_bond_dist,
            'all_results': results
        }, os.path.join(result_path, f'metrics_{args.eval_step}.pt'))