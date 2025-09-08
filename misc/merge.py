import os
import shutil

def merge_subfolders(src_base, dest_base):
    # 获取源目录中的所有子文件夹
    src_subfolders = [d for d in os.listdir(src_base) if os.path.isdir(os.path.join(src_base, d))]
    
    # 遍历每个子文件夹
    for subfolder in src_subfolders:
        src_subfolder_path = os.path.join(src_base, subfolder)
        dest_subfolder_path = os.path.join(dest_base, subfolder)
        
        # 检查子文件夹是否也存在于目标目录中
        if os.path.exists(dest_subfolder_path):
            # 合并两个子文件夹的内容
            for item in os.listdir(src_subfolder_path):
                src_item_path = os.path.join(src_subfolder_path, item)
                dest_item_path = os.path.join(dest_subfolder_path, item)
                
                if os.path.isdir(src_item_path):
                    shutil.copytree(src_item_path, dest_item_path)
                else:
                    shutil.copy2(src_item_path, dest_item_path)
            print(f"Merged contents of {src_subfolder_path} into {dest_subfolder_path}")
        else:
            print(f"Subfolder {subfolder} not found in destination base folder. Skipping.")

# 调用函数，合并两个目录中的子文件夹内容
merge_subfolders("new_folder", "data/crossdocked_v1.1_rmsd1.0")
