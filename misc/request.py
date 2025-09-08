import csv
import requests
from tqdm import tqdm

# 1. 从CSV文件读取EC numbers和subfolder信息
ec_numbers_dict = {}
with open('pdb_ec_data_new.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        ec_number = row['EC Number']
        if "." in ec_number:  # 检查EC编号是否是有效的
            ec_numbers_dict[row['PDB Code']] = {'EC Number': ec_number, 'Subfolder': row['Subfolder']}

print(f"Read {len(ec_numbers_dict)} valid EC numbers from the CSV file.")

# 函数：使用Rhea API查询反应和配体信息
def get_reaction_and_ligand_info(ec_number):
    url = "https://www.rhea-db.org/rhea?"
    parameters = {
        "query": ec_number,
        "columns": "rhea-id,equation,chebi,chebi-id",
        "format": 'tsv',
        "limit": 10,
    }
    response = requests.get(url, params=parameters)
    
    if response.status_code == 200:
        data = response.text
        parsed_data = data.split('\n')[1:-1]  # 返回TSV数据，排除头行和末行（通常是空行）
        return parsed_data
    else:
        return []

# 2. 查询Rhea API并收集结果
results = []
for pdb_code, data_dict in tqdm(ec_numbers_dict.items(), desc="Fetching data from Rhea API"):
    ec_number = data_dict['EC Number']
    data = get_reaction_and_ligand_info(ec_number)
    if data:
        for line in data:
            line_data = line.split('\t')
            # 检查返回数据的长度
            if len(line_data) == 4:
                names = line_data[2].split(';')
                ids = line_data[3].split(';')

                # 使用列表推导来创建过滤后的名称和ID
                filtered_data = [(name, id_) for name, id_ in zip(names, ids) if len(name) > 5]
                if filtered_data:  # 确保有符合条件的数据
                    filtered_names = [item[0] for item in filtered_data]
                    filtered_ids = [item[1] for item in filtered_data]

                    results.append((pdb_code, data_dict['Subfolder'], ec_number, line_data[0], line_data[1],
                                    ';'.join(filtered_names), ';'.join(filtered_ids)))
            else:
                print(f"Unexpected data format for EC number: {ec_number}")
    else:
        print(f"No data found for EC number: {ec_number}")

# 3. 更新原CSV文件
with open('pdb_ec_data_new.csv', 'w', newline='') as csvfile:
    fieldnames = ['PDB Code', 'Subfolder', 'EC Number', 'Rhea ID', 'Equation', 'ChEBI Names', 'ChEBI IDs']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    
    for result in results:
        writer.writerow({
            'PDB Code': result[0],
            'Subfolder': result[1],
            'EC Number': result[2],
            'Rhea ID': result[3],
            'Equation': result[4],
            'ChEBI Names': result[5],
            'ChEBI IDs': result[6]
        })

print("Data retrieval and saving completed.")
















