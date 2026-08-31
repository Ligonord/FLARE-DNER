import torch
from model import Model
import config

dataset = 'cadec'
# dataset = 'share13'
# dataset = 'share14'
# 載入相同設定
config_path = f"./config/{dataset}.json"   # 改成你的設定檔路徑
args = type('', (), {'config': config_path})()  # 假造 argparse 結果
cfg = config.Config(args)
cfg.label_num = 3
cfg.old_label_num = 3   # 舊版 predictor 類別數
cfg.new_label_num = 3   # 新版 predictor 類別數
cfg.lstm_hid_size = 516
cfg.biaffine_size = 516
bert_config = config.Config(args, is_bert=True)

# 建立模型
model = Model(cfg, bert_config)

# # 載入訓練好的參數
# model.load_state_dict(torch.load(f"./{dataset}_model.pt", map_location="cpu"))

# 統計參數量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"🔹 Total parameters: {total_params:,}")
print(f"🔹 Trainable parameters: {trainable_params:,}")

# # 額外：查看每層參數資訊（可選）
# print("\n🔍 Layer-wise parameter summary:")
# for name, param in model.named_parameters():
#     print(f"{name:50s} {param.numel():>10,d} trainable={param.requires_grad}")
