import csv
import os
import requests
from tqdm import tqdm

# 从CSV文件读取数据
ec_data_file = 'pdb_ec_data_new922.csv'
data_dir = "/home/dhzhang/targetdiff/data/crossdocked_v1.1_rmsd1.0/"

def fetch_sdf(chebi_id, retries=3):
    url = f"https://www.ebi.ac.uk/chebi/saveStructure.do?defaultImage=true&chebiId={chebi_id}&imageId=0"
    for _ in range(retries):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                return response.text
        except requests.exceptions.RequestException as e:
            print(f"Error: {e}. Retrying...")
    return None

# 从CSV读取数据
with open(ec_data_file, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in tqdm(reader, desc="Fetching SDF files"):
        subfolder = os.path.join(data_dir, row['Subfolder'])
        chebi_names = row['ChEBI Names'].split(";")
        chebi_ids = row['ChEBI IDs'].split(";")
        
        for name, chebi_id in zip(chebi_names, chebi_ids):
            # 清理名字，移除不适合用作文件名的字符
            clean_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")
            
            # 为文件名加上 PDB Code
            pdb_code = row['PDB Code']  # 假设CSV的列名为'PDB Code'，请根据实际情况进行调整
            file_name = f"{pdb_code}_{clean_name}_{chebi_id}.sdf"
            file_path = os.path.join(subfolder, file_name)
              
            if not os.path.exists(file_path):
                sdf_content = fetch_sdf(chebi_id.strip('CHEBI:'))
                
                # 如果找到了SDF内容，保存到文件中
                if sdf_content:
                    with open(file_path, 'w') as sdf_file:
                        sdf_file.write(sdf_content)
                    
print("Process completed.")






