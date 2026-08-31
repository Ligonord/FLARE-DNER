import torch

# 是否可用 CUDA
print("CUDA available:", torch.cuda.is_available())

# 顯卡數量
print("GPU count:", torch.cuda.device_count())

# 顯卡名稱
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    print("Current device:", torch.cuda.current_device())