import torch

# 替换为您的文件路径
# file_path = './sampling_results/all_docking_results_-1.pt'
file_path = './outputs/result_1.pt'
# file_path = './sampling_results/crossdocked_test_vina_docked.pt'
# 加载文件
loaded_data = torch.load(file_path)

# 处理加载的数据
# 这取决于数据的具体类型
print(loaded_data)
