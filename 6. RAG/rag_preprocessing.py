import json
from collections import Counter

threshold = 2
for dataset in ['cadec', 'share13', 'share14']:
    if dataset == 'cadec':
        with open(f'./data/format_alignment/{dataset}_ensemble.json', 'r', encoding="utf-8") as f:
            ensemble_data = json.load(f)
    else:
        with open(f'./data/format_alignment/{dataset}_ensemble_fixed.json', 'r', encoding="utf-8") as f:
            ensemble_data = json.load(f)
    
    output = []
    for i, d in enumerate(ensemble_data, start=1):
        appear_tuples = []
        union = []

        # 1. 遍歷 5 個 entity_list
        for j in range(1, 6):
            for ent in d.get(f'entity_list_{j}', []):
                # 關鍵步驟：將內部所有的 list 轉換為不可變的 tuple
                # 同時用 tuple(dict.items()) 讓整個字典變得可雜湊 (hashable)
                ent_hashable = (
                    ("text", tuple(ent["text"])),
                    ("index", tuple(ent["index"]))
                )
                appear_tuples.append(ent_hashable)

        # 2. 丟給 Counter 計算次數
        count = Counter(appear_tuples)
        ensemble = []
        for ent_tuple, freq in count.items():
            union.append({
                "text": list(dict(ent_tuple)["text"]),
                "index": list(dict(ent_tuple)["index"]),
                'weight': freq
            })

            # 根據 threshold 篩選出通過投票的實體，並還原回原本的 dict/list 格式
            if freq > threshold:
                # 還原結構
                ent_dict = dict(ent_tuple)
                ensemble.append({
                    "text": list(ent_dict["text"]),
                    "index": list(ent_dict["index"])
                })

        # 3. 封裝輸出
        output.append({
            'number': i,
            'sentence': d['sentence'],
            'text': d['text'],
            'entity_list_rag': ensemble,
            'entity_list': union
        })

    with open(f"./data/input/{dataset}_rag_pre.json", "w", encoding="utf-8") as f:
        json.dump(output, f)