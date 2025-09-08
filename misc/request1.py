import os
import requests
import csv
from tqdm import tqdm

# 指定你的数据目录路径
data_dir = "/home/dhzhang/targetdiff/data/crossdocked_v1.1_rmsd1.0"

# 函数：从文件名提取PDB code
def extract_pdb_code(file_name):
    # 提取文件名的前4个字符作为PDB code
    return file_name[:4]

# 函数：获取EC number
def get_ec_number(pdb_code):
    try:
        url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_code}/1"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            ec_number = data.get('rcsb_polymer_entity', {}).get('pdbx_ec', None)
            return ec_number
        else:
            return None
    except Exception as e:
        return None

# 字典，用于存储PDB代码及其对应的EC number和subfolder名字
pdb_ec_subfolder_dict = {}

# 获取所有PDB文件及其subfolder的路径
all_files = [f for root, dirs, files in os.walk(data_dir) for f in files if f.endswith(".pdb")]
for file_path in tqdm(all_files, desc="Processing PDB files"):
    root = os.path.dirname(file_path)
    file = os.path.basename(file_path)
    
    subfolder_name = os.path.basename(root)
    pdb_code = extract_pdb_code(file)
    if pdb_code not in pdb_ec_subfolder_dict:
        ec_number = get_ec_number(pdb_code)
        if ec_number:
            pdb_ec_subfolder_dict[pdb_code] = {"subfolder": subfolder_name, "ec_number": ec_number}

# 将结果保存为CSV文件
with open('pdb_ec_data_new922.csv', 'w', newline='') as csvfile:
    fieldnames = ['Subfolder', 'PDB Code', 'EC Number']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    for pdb_code, info in pdb_ec_subfolder_dict.items():
        writer.writerow({'Subfolder': info["subfolder"], 'PDB Code': pdb_code, 'EC Number': info["ec_number"]})

print("Data saved to pdb_ec_data_new.csv.")



