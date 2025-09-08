import os
import requests
import csv
from tqdm import tqdm

# 指定你的数据目录路径
data_dir = "/home/dhzhang/targetdiff/data/new_data"

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
            ec_number = data.get('rcsb_polymer_entity', {}).get('pdbx_ec', "No EC number information available")
            return ec_number.split(",")  # 返回一个EC number列表
        else:
            return [f"Error: {response.status_code}"]
    except Exception as e:
        return [f"An error occurred: {e}"]

# 字典，用于存储PDB代码、其父目录及其对应的EC number
pdb_ec_entries = []

# 获取所有PDB文件列表
pdb_files = [os.path.join(root, f) for root, dirs, files in os.walk(data_dir) for f in files if f.endswith(".pdb")]

# 遍历数据目录并提取PDB codes和subfolder
for file_path in tqdm(pdb_files, desc="Processing files"):
    pdb_code = extract_pdb_code(os.path.basename(file_path))
    subfolder = os.path.basename(os.path.dirname(file_path))
    ec_numbers = get_ec_number(pdb_code)
    
    for ec in ec_numbers:
        pdb_ec_entries.append({
            'PDB Code': pdb_code,
            'Subfolder': subfolder,
            'EC Number': ec
        })

# 将结果保存为CSV文件
with open('pdb_ec_data_new.csv', 'w', newline='') as csvfile:
    fieldnames = ['PDB Code', 'Subfolder', 'EC Number']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    for entry in pdb_ec_entries:
        writer.writerow(entry)

print("CSV file generation completed.")





