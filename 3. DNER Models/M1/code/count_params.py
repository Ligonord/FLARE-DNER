import torch
from xdai.ner.transition_discontinuous.models import TransitionModel
from xdai.utils.vocab import Vocabulary
import json, os

class Args:
    # Model type & embeddings
    model_type = 'elmo'
    pretrained_model_dir = './data/elmo_2x4096_512_2048cnn_2xhighway_5.5B'
    pretrained_word_embeddings = './data/glove.6B.100d.txt'
    action_embedding_size = 20
    word_embedding_size = 100
    char_embedding_size = 16
    do_lower_case = False

    # LSTM settings
    lstm_cell_size = 200
    lstm_layers = 2
    dropout = 0.5

    # Misc
    cuda_device = [0]       # 注意是 list
    max_seq_length = 128
    labels = "0,1"
    tag_schema = "B,I"

    # Training / evaluation placeholders (可隨意填，因為只算參數量)
    train_batch_size_per_gpu = 8
    eval_batch_size_per_gpu = 8
    learning_rate = 0.001
    do_train = False
    do_eval = False


# ======= 載入設定 =======
output_dir = "./data/Experiments/cadec/52"  # 改成你的目錄
args = Args()

# ======= 載入vocab =======
vocab = Vocabulary.from_files(os.path.join(output_dir, "vocabulary"))

# ======= 建立模型 =======
model = TransitionModel(args, vocab)

# ======= 載入已訓練權重 =======
state_dict = torch.load(os.path.join(output_dir, "best.th"), map_location="cpu")
model.load_state_dict(state_dict)

# ======= 計算參數數量 =======
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {num_params:,}")
