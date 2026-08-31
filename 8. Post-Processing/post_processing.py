import json
from collections import Counter
from scipy.stats import entropy

def has_discontinuous_offset(ent):
    """判斷數字列表是否連續"""
    if not ent:  # 空列表視為 False
        return False
    ent = list(ent)
    return not ent == list(range(ent[0], ent[-1] + 1))

def has_discontinuous_text(ent, sentence):
    entity = list(ent)
    if entity:
        return any(e not in sentence for e in entity)
    else:
        return False

def vote_entropy(i, m_o, m_t, v_o, v_t, r_o, r_t, e_d, sen, b, e_th, v_th, out):
    appear_offset = []
    appear_text = []
    total_entropy_offset = 0
    total_entropy_text = 0

    for k in range(1, 6):
        for ent in e_d[f'entity_list_{k}']:
            appear_offset.append(ent['index'])
            appear_text.append(" ".join(ent['text']))

    count_offset = Counter(tuple(l) for l in appear_offset)
    count_text = Counter(appear_text)
    entity_offset_counts = [v for k, v in count_offset.most_common()]
    entity_offset = [k for k, v in count_offset.most_common()]
    entity_text_counts = list(count_text.values())
    entity_text = list(count_text.keys())

    for o in entity_offset_counts:
        total_entropy_offset += entropy([o/5, 1 - o/5], base = b)
    for t in entity_text_counts:
        total_entropy_text += entropy([t/5, 1 - t/5], base = b)

    if len(entity_offset_counts) != 0 and total_entropy_offset / len(entity_offset_counts) / entropy([3/5, 2/5], base = b) > e_th:
        for ent in r_o:
            appear_offset.append(ent)
        count = Counter(tuple(l) for l in appear_offset)
        entity_counts = [v for k, v in count.most_common()]

        temp_entropy = 0
        for o in entity_counts:
            temp_entropy += entropy([o/6, 1 - o/6], base = b)
        
        if temp_entropy / len(entity_counts) / entropy([3/6, 3/6], base = b) < total_entropy_offset / len(entity_offset_counts) / entropy([3/5, 2/5], base = b):
            entity_offset_temp = [k for k, v in count.items() if v > v_th]
        else:
            entity_offset_temp = v_o
    else:
        entity_offset_temp = v_o

    if len(entity_text_counts) != 0 and total_entropy_text / len(entity_text_counts) / entropy([3/5, 2/5], base = b) > e_th:
        for ent in r_t:
            appear_text.append(ent)
        count = Counter(appear_text)
        entity_counts = list(count.values())

        temp_entropy = 0
        for o in entity_counts:
            temp_entropy += entropy([o/6, 1 - o/6], base = b)

        if temp_entropy / len(entity_counts) / entropy([3/6, 3/6], base = b) < total_entropy_text / len(entity_text_counts) / entropy([3/5, 2/5], base = b):
            entity_text_temp = [k for k, v in count.items() if v > v_th]
        else:
            entity_text_temp = v_t
    else:
        entity_text_temp = v_t

    # entity_offset_temp = v_o
    if any(has_discontinuous_offset(ent) for ent in entity_offset):
        for e in r_o:
            if has_discontinuous_offset(e) and e not in entity_offset_temp:
                entity_offset_temp.append(e)

    # entity_text_temp = v_t
    if has_discontinuous_text(entity_text, sen[0]):
        for e in r_t:
            if e not in sen[0] and e not in entity_text_temp:
                entity_text_temp.append(e)

    m_o.append(entity_offset_temp)
    m_t.append(entity_text_temp)

    out.append({
        'number': i + 1,
        "sentence": sen,
        'entity_offset': entity_offset_temp,
        'entity_text': entity_text_temp,
    })

    return

base = 2
entropy_threshold = .8
voting_threshold = 3
dataset = 'cadec'
dataset = 'share13'
dataset = 'share14'

if dataset == 'cadec':
    ensemble_data = json.load(open(f'./input/format alignment/{dataset}_ensemble.json', "r", encoding="utf-8"))
    vote_data = json.load(open(f'./input/meta_learner/{dataset}_voting_2.json', "r", encoding="utf-8"))
else:
    ensemble_data = json.load(open(f'./input/format alignment/{dataset}_ensemble_fixed.json', "r", encoding="utf-8"))
    vote_data = json.load(open(f'./input/meta_learner/{dataset}_voting_2_fixed.json', "r", encoding="utf-8"))
rag_data = json.load(open(f'./input/gpt/{dataset}_output.json', "r", encoding="utf-8"))

sentence = [[g['sentence']] for g in ensemble_data]
vote_entity_offset = [[ent for ent in g["ner_offset"]] for g in vote_data]
vote_entity_text = [[ent for ent in g["ner_text"]]for g in vote_data]
rag_entity_offset = [[ent["index"] for ent in g["entity_list"]] for g in rag_data]
rag_entity_text = [[" ".join(ent["text"]) for ent in g["entity_list"]]for g in rag_data]
merge_entity_offset = []
merge_entity_text = []

out = []
for i, (v_o, v_t, r_o, r_t, e_d, sen) in enumerate(zip(vote_entity_offset, vote_entity_text, rag_entity_offset, rag_entity_text, ensemble_data, sentence)):
    count = vote_entropy(i, merge_entity_offset, merge_entity_text, v_o, v_t, r_o, r_t, e_d, sen, base, entropy_threshold, voting_threshold, out)

with open(f"./output/{dataset}_qbc.json", "w", encoding="utf-8") as f:
    json.dump(out, f)