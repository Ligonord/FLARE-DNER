import json
import networkx as nx

def has_discontinuous(ent):
    return len(ent) > 2

def has_continuous(ent):
    return len(ent) == 2

def evaluate(gold_data, pred_data, mode):
    TP, FP, FN = 0, 0, 0

    for gold_ents, pred_ents in zip(gold_data, pred_data):
        gold_set = set(tuple(span) for span in gold_ents)
        pred_set = set(tuple(span) for span in pred_ents)

        # 判斷是否該句包含非連續
        has_disc = any(has_discontinuous(ent) for ent in gold_ents)
        has_c = any(has_continuous(ent) for ent in gold_ents)

        # 依 mode 過濾
        if mode == "only_c" and not has_c:
            continue
        if mode == "with_disc" and not has_disc:
            continue
        if mode == "only_disc" and not has_disc:
            continue
        if mode == "only_disc":
            gold_set = {ent for ent in gold_set if has_discontinuous(ent)}
            pred_set = {ent for ent in pred_set if has_discontinuous(ent)}
        if mode == "only_c":
            gold_set = {ent for ent in gold_set if has_continuous(ent)}
            pred_set = {ent for ent in pred_set if has_continuous(ent)}
        # print(gold_set)
        # print(pred_set)

        TP += len(gold_set & pred_set)   # 預測對了
        FP += len(pred_set - gold_set)   # 預測有但標註沒有
        FN += len(gold_set - pred_set)   # 標註有但預測沒抓到
    print("TP:", TP, "FP:", FP, "FN:", FN)

    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1

dataset = 'cadec'
dataset = 'share13'
dataset = 'share14'
if dataset == "cadec":
    gold_path = f'm2/{dataset}/test.json'
    pred_path = f'm2/{dataset}/{dataset}_prediction.txt'
else:
    gold_path = f'm2/{dataset}_fixed/test.json'
    pred_path = f'm2/{dataset}_fixed/{dataset}_prediction.txt'
# gold_path = 'm2/gold.json'
# pred_path = "m2/pred.txt"
gold_data = []
pred_data = []
with open(gold_path, "r", encoding="utf-8") as f:
    for line in f:
        gold_data.append(json.loads(line))
with open(pred_path, "r", encoding="utf-8") as f:
    for line in f:
        pred_data.append(json.loads(line))
pred_data = [item for sublist in pred_data for item in sublist]

gold_entity = []
for g in gold_data:
    G = nx.Graph()
    rel = g.get('relations')[0]
    gold = g.get('ner')[0]
    gold = [g[:-1] for g in gold]

    dner_rel = [r for r in rel if r[-1] == "Combined"]
    for dr in dner_rel:
        node1 = tuple(dr[:2])
        node2 = tuple(dr[2:4])
        G.add_edge(node1, node2)
    
    # 找所有 maximal cliques
    dner_slice = [[list(t) for t in sublist] for sublist in list(nx.find_cliques(G))]
    for slice in dner_slice:
        for s in slice:
            if s in gold:
                gold.pop(gold.index(s))

    # 依照每個節點的第一個值排序，然後展平成單一 list
    dner = [sum(sorted(s, key=lambda x: x[0]), []) for s in dner_slice]
    gold.extend(dner)
    gold_entity.append(gold)
# print(gold_entity)

pred_entity = []
for p in pred_data:
    G = nx.Graph()
    rel = p.get('relation')
    pred = p.get('ner')
    pred = [p[:-1] for p in pred]

    dner_rel = [r for r in rel if r[-1] == "Combined"]
    for dr in dner_rel:
        node1 = tuple(dr[:2])
        node2 = tuple(dr[2:4])
        G.add_edge(node1, node2)
    
    # 找所有 maximal cliques
    dner_slice = [[list(t) for t in sublist] for sublist in list(nx.find_cliques(G))]
    for slice in dner_slice:
        for s in slice:
            if s in pred:
                pred.pop(pred.index(s))

    # 依照每個節點的第一個值排序，然後展平成單一 list
    dner = [sum(sorted(s, key=lambda x: x[0]), []) for s in dner_slice]
    pred.extend(dner)
    pred_entity.append(pred)
# print(pred_entity)

p, r, f1 = evaluate(gold_entity, pred_entity, 'all')
print(f"Precision: {p:.4}, Recall: {r:.4}, F1: {f1:.4}")
p, r, f1 = evaluate(gold_entity, pred_entity, 'with_disc')
print(f"Precision: {p:.4}, Recall: {r:.4}, F1: {f1:.4}")
p, r, f1 = evaluate(gold_entity, pred_entity, 'only_disc')
print(f"Precision: {p:.4}, Recall: {r:.4}, F1: {f1:.4}")
p, r, f1 = evaluate(gold_entity, pred_entity, 'only_c')
print(f"Precision: {p:.4}, Recall: {r:.4}, F1: {f1:.4}")