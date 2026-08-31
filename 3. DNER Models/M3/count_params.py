import torch
from InfExtraction.modules import models
from InfExtraction.modules import taggers

# ======= 模擬 settings =======
class Args:
    seed = 2333
    device_num = 0
    model_name = "RAIN"
    tagger_name = "Tagger4RAIN"
    task_type = "re+ner"
    token_level = "subword"

    # encoder 選項
    subwd_encoder = True
    word_encoder = False
    char_encoder = False
    pos_tag_emb = False
    ner_tag_emb = False
    dep_gcn = False
    use_attns4rel = True

    pretrained_model_name = "YelpBERT"
    pretrained_emb_name = "glove.6B.100d.txt"

    # embedding size
    ent_dim = 768
    rel_dim = 768

    handshaking_kernel_config = {
        "ent_shaking_type": "cln+lstm",
        "rel_shaking_type": "cln",
    }

# ======= 模型設定 =======
args = Args()

model_settings = {
    "subwd_encoder_config": {
        "pretrained_model_path": f"data/pretrained_models/{args.pretrained_model_name}",
        "finetune": True,
        "use_last_k_layers": 1,
    } if args.subwd_encoder else None,
    "word_encoder_config": None,
    "char_encoder_config": None,
    "pos_tag_emb_config": None,
    "ner_tag_emb_config": None,
    "dep_config": None,
    "handshaking_kernel_config": args.handshaking_kernel_config,
    "use_attns4rel": args.use_attns4rel,
    "ent_dim": args.ent_dim,
    "rel_dim": args.rel_dim,
    "tok_pair_neg_sampling_rate": 0.5,
    "clique_comp_loss": False,
    "do_span_len_emb": True,
    "loss_weight": 0.5,
    "loss_weight_recover_steps": 0,
}

# ======= 初始化 tagger =======
# 模型初始化時需要 tagger，這裡直接用空列表初始化
tagger_class = getattr(taggers, args.tagger_name)
dummy_data = []
additional_preprocessing_config = {
    "classify_entities_by_relation": False,
    "add_default_entity_type": False,
    "use_bound": True
}
tagger = tagger_class(dummy_data, **additional_preprocessing_config)

# ======= 初始化模型 =======
model_class = getattr(models, args.model_name)
model = model_class(tagger, **model_settings)

# ======= 移到 GPU 或 CPU =======
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ======= 計算可訓練參數量 =======
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {num_params:,}")
