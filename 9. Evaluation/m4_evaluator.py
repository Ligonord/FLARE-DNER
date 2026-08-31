import json

def has_discontinuous(ent):
    """判斷數字列表是否連續"""
    if not ent:  # 空列表視為 False
        return False
    ent = list(ent)
    return not ent == list(range(ent[0], ent[-1] + 1))

def has_continuous(ent):
    """判斷數字列表是否連續"""
    if not ent:  # 空列表視為 False
        return False
    ent = list(ent)
    return ent == list(range(ent[0], ent[-1] + 1))

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
    gold_data = json.load(open(f"m4/{dataset}/test_data.json", "r", encoding="utf-8"))
    pred_data = json.load(open(f"m4/{dataset}/{dataset}_output.json", "r", encoding="utf-8"))
else:
    gold_data = json.load(open(f"m4/{dataset}_fixed/test_data.json", "r", encoding="utf-8"))
    pred_data = json.load(open(f"m4/{dataset}_fixed/{dataset}_output.json", "r", encoding="utf-8"))
# gold_data = json.load(open("m4/gold.json", "r", encoding="utf-8"))
# pred_data = json.load(open("m4/pred.json", "r", encoding="utf-8"))

gold_entity = [[ent["index"] for ent in g["ner"]] for g in gold_data]
pred_entity = [[ent["index"] for ent in p["entity"]] for p in pred_data]
# print(gold_entity)
# print(pred_entity)

p, r, f1 = evaluate(gold_entity, pred_entity, 'all')
print(f"Precision: {p:.4}, Recall: {r:.4}, F1: {f1:.4}")
p, r, f1 = evaluate(gold_entity, pred_entity, 'with_disc')
print(f"Precision: {p:.4}, Recall: {r:.4}, F1: {f1:.4}")
p, r, f1 = evaluate(gold_entity, pred_entity, 'only_disc')
print(f"Precision: {p:.4}, Recall: {r:.4}, F1: {f1:.4}")
p, r, f1 = evaluate(gold_entity, pred_entity, 'only_c')
print(f"Precision: {p:.4}, Recall: {r:.4}, F1: {f1:.4}")