import os

def count_subfolders(directory):
    all_items = os.listdir(directory)
    subfolders = [item for item in all_items if os.path.isdir(os.path.join(directory, item))]
    return len(subfolders)

directory_path = "data/crossdocked_v1.1_rmsd1.0 copy"
number_of_subfolders = count_subfolders(directory_path)
print(f"There are {number_of_subfolders} subfolders in {directory_path}.")
