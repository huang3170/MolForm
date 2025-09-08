import os
import subprocess
import random
import string
from easydict import EasyDict
from rdkit import Chem
from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule
from rdkit.Chem import Draw
from tqdm import tqdm
import csv
import multiprocessing


def load_pdb(path):
    with open(path, 'r') as f:
        return f.read()

def parse_qvina_outputs(docked_sdf_path):
    try:
        suppl = Chem.SDMolSupplier(docked_sdf_path)
    except Exception as e:
        print("Error while parsing the file:", e)
        return []

    results = []
    for i, mol in enumerate(suppl):
        if mol is None:
            continue
        line = mol.GetProp('REMARK').splitlines()[0].split()[2:]
        results.append(EasyDict({
            'rdmol': mol,
            'mode_id': i,
            'affinity': float(line[0]),
            'rmsd_lb': float(line[1]),
            'rmsd_ub': float(line[2]),
        }))
    return results

class BaseDockingTask(object):
    def __init__(self, pdb_block, ligand_rdmol):
        self.pdb_block = pdb_block
        self.ligand_rdmol = ligand_rdmol

    def run(self):
        raise NotImplementedError()

    def get_results(self):
        raise NotImplementedError()

class QVinaDockingTask(BaseDockingTask):
    def __init__(self, pdb_block, ligand_rdmol, process_id, folder_name, receptor_filename, ligand_filename, use_uff=True, center=None, size_factor=None):
        super(QVinaDockingTask, self).__init__(pdb_block, ligand_rdmol)
        
        # 使用 process_id 为每个任务创建一个唯一的临时目录
        self.tmp_dir = os.path.join(folder_name, f"process_{process_id}")
        self.ligand_filename = ligand_filename.split('.')[0]
        self.receptor_filename =  receptor_filename.split('.')[0]
        self.receptor_path = os.path.join(self.tmp_dir, self.receptor_filename + '.pdb')
        self.ligand_path = os.path.join(self.tmp_dir, self.ligand_filename + '.sdf')
        self.conda_env = "MolDiff"
        self.use_uff = use_uff
        self.proc = None
        self.results = None
        self.output = None
        self.error_output = None
        self.docked_sdf_path = None

        os.makedirs(self.tmp_dir, exist_ok=True)
        with open(self.receptor_path, 'w') as f:
            f.write(pdb_block)
        
        sdf_writer = Chem.SDWriter(self.ligand_path)
        sdf_writer.write(ligand_rdmol)
        sdf_writer.close()

        pos = ligand_rdmol.GetConformer(0).GetPositions()
        if center is None:
            self.center = (pos.max(0) + pos.min(0)) / 2
        else:
            self.center = center

        if size_factor is None:
            self.size_x, self.size_y, self.size_z = 30, 30, 30
        else:
            self.size_x, self.size_y, self.size_z = (pos.max(0) - pos.min(0)) * size_factor

    def run(self, exhaustiveness=16, seed=1234, cpu=None, num_modes=9, energy_range=3):
        commands = [
            f"eval \"$(conda shell.bash hook)\"",
            f"conda activate {self.conda_env}",
            f"cd {self.tmp_dir}",
            f"/home/dhzhang/targetdiff/AutoDockTools_py3/AutoDockTools/Utilities24/prepare_receptor4.py -r {self.receptor_filename}.pdb",
            f"obabel {self.ligand_filename}.sdf -O{self.ligand_filename}.pdbqt",
            f"pythonsh ~/targetdiff/AutoDock-Vina/example/autodock_scripts/prepare_gpf.py -l {self.ligand_filename}.pdbqt -r {self.receptor_filename}.pdbqt -p npts='{self.size_x},{self.size_y},{self.size_z}' -p gridcenter='{self.center[0]:.4f},{self.center[1]:.4f},{self.center[2]:.4f}'"
            f"autogrid4 -p {self.receptor_filename}.gpf -l {self.receptor_filename}.glg",       
            # 使用AutoDock-GPU进行对接
            f"ad_gpu --ffile {self.receptor_filename}.maps.fld --lfile {self.ligand_filename}.pdbqt --nrun {nrun} --resnam {self.ligand_filename.split('.')[0]}"
            # f"qvina2 --receptor {self.receptor_filename}.pdbqt --ligand {self.ligand_filename}.pdbqt "
            # f"--center_x {self.center[0]:.4f} --center_y {self.center[1]:.4f} --center_z {self.center[2]:.4f} "
            # f"--size_x {self.size_x} --size_y {self.size_y} --size_z {self.size_z} --exhaustiveness {exhaustiveness} "
            # f"--num_modes {num_modes} --energy_range {energy_range} --cpu 256"
        ]


        self.docked_sdf_path = os.path.join(self.tmp_dir, f'{self.ligand_filename}_out.sdf')
        self.proc = subprocess.Popen(
            '/bin/bash',
            shell=False,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        self.proc.stdin.write('\n'.join(commands).encode('utf-8'))
        self.proc.stdin.close()

    def run_sync(self):
        self.run()
        while self.get_results() is None:
            pass
        results = self.get_results()
        if not results:  # 检查results是否为空
            print(f"[Warning] No results for {self.docked_sdf_path}. Skipping.")
            return None
        return results

    def get_results(self):
        if self.proc is None:  # Not started
            return None
        elif self.proc.poll() is None:  # In progress
            return None
        else:
            if self.output is None:
                self.output = self.proc.stdout.readlines()
                self.error_output = self.proc.stderr.readlines()
                try:
                    self.results = parse_qvina_outputs(self.docked_sdf_path)
                except:
                    print('[Error] Vina output error: %s' % self.docked_sdf_path)
                    return []
            return self.results


def get_all_subfolders(base_dir):
    all_items = os.listdir(base_dir)
    subfolders = [os.path.join(base_dir, item) for item in all_items if os.path.isdir(os.path.join(base_dir, item))]
    return subfolders

def dock_all(base_dir, subfolders, process_id, use_uff=True):
    create_folder = os.path.join("/home/dhzhang/targetdiff/create0/", f"process_{process_id}")  # 使用 process_id 创建一个唯一的目录
    if not os.path.exists(create_folder):
        os.makedirs(create_folder)

    docking_info = []

    for subfolder_path in tqdm(subfolders, desc=f"Subfolders process_{process_id}", unit="folder"):
        subfolder_name = subfolder_path.split('/')[-1]
        print(f"Processing subfolder: {subfolder_path}")

        pdb_files = [f for f in os.listdir(subfolder_path) if f.endswith("_rec.pdb")]

        if len(pdb_files) == 0:
            print(f"No PDB file found in subfolder {subfolder_path}")
            continue

        for pdb_file in pdb_files:
            pdb_code = pdb_file.split("_")[0]
            receptor_path = os.path.join(subfolder_path, pdb_file)
            sdf_files = [f for f in os.listdir(subfolder_path) if f.startswith(pdb_code) and f.endswith(".sdf")]

            for sdf_file in sdf_files:
                
                if  os.path.exists(os.path.join(create_folder, subfolder_name,f"process_{process_id}",sdf_file.split('.')[0]+'_out.sdf')):
                    continue
                    
                ligand_path = os.path.join(subfolder_path, sdf_file)
          
                if os.path.exists(receptor_path) and os.path.exists(ligand_path):
                    print(f"Docking receptor {pdb_file} with ligand {sdf_file}")
                   
                    receptor_block = load_pdb(receptor_path)
                    ligand_rdmol = next(iter(Chem.SDMolSupplier(ligand_path)))
                    if ligand_rdmol is None:  # Check if the molecule was read correctly
                        print(f"Failed to read ligand from {sdf_file}. Skipping this molecule.")
                        continue

                    if use_uff:
                        ligand_rdmol = Chem.AddHs(ligand_rdmol, addCoords=True)
                        if ligand_rdmol.GetNumAtoms() == 0 or not any(atom.GetSymbol() == 'H' for atom in ligand_rdmol.GetAtoms()):
                            print(f"Failed to add hydrogens to ligand {sdf_file}. Skipping this molecule.")
                            continue

                        try:
                            UFFOptimizeMolecule(ligand_rdmol)
                        except RuntimeError:
                            print(f"Failed to optimize ligand {sdf_file}. Skipping this molecule.")
                            continue
                    
                    task = QVinaDockingTask(receptor_block, ligand_rdmol, process_id, os.path.join(create_folder, subfolder_name), pdb_file, sdf_file, use_uff=use_uff)
                    task.run_sync()
                    results = task.run_sync()
                    if results:
                        for result in results:
                            docking_info.append((pdb_file, sdf_file, result.rmsd_lb, result.rmsd_ub))


    csv_filename = os.path.join("/home/dhzhang/targetdiff/results", f"results_{process_id}_trail2.csv")
    with open(csv_filename, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["PDB", "SDF", "RMSD_LB", "RMSD_UB"])
        csvwriter.writerows(docking_info)

if __name__ == '__main__':
    numList = []
    base_dir = "/home/dhzhang/targetdiff/data/new_data"

    subfolders = get_all_subfolders(base_dir)
    print(f"Found {len(subfolders)} subfolders.")

    for i in range(32):
        p = multiprocessing.Process(target=dock_all, args=(base_dir, subfolders[len(subfolders)//32*i:len(subfolders)//32*(i+1)], i))  # 将 i 作为 process_id 传递
        numList.append(p)
        p.start()

    for p in numList:
        p.join()  # 确保所有进程都启动后再加入它们

  

