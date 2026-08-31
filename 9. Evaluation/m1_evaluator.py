import math

def read_data(filename):
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]

    i = 0
    while i < len(lines):
        sentence = lines[i].strip()
        if i + 1 < len(lines):
            annotations = lines[i+1].strip()
        else:
            annotations = ""
        
        entities = annotations.split("|") if annotations else []
        data.append((sentence, entities))
        
        # 跳過 sentence, annotation, 空行
        i += 3  

    return data

def has_discontinuous(ent):
    """判斷實體是否為非連續（有超過2個 index）"""
    parts = ent.split()[0].split(",")  # 抓數字部分
    return len(parts) > 2

def has_continuous(ent):
    """判斷實體是否為非連續（有超過2個 index）"""
    parts = ent.split()[0].split(",")  # 抓數字部分
    return len(parts) == 2

def evaluate(gold_data, pred_data, mode):
    TP, FP, FN = 0, 0, 0

    for (sent_g, gold_ents), (sent_p, pred_ents) in zip(gold_data, pred_data):
        gold_set = set(gold_ents)
        pred_set = set(pred_ents)

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

        TP += len(gold_set & pred_set)
        FP += len(pred_set - gold_set)
        FN += len(gold_set - pred_set)

    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    print("TP:", TP, "FP:", FP, "FN:", FN)
    return precision, recall, f1

def calculate_stats(scores):
    if not scores:
        return "0.00 ± 0.00"
    
    # 1. 計算平均值 (Mean)
    n = len(scores)
    mean = sum(scores) / n
    
    # 2. 計算樣本標準差 (Standard Deviation)
    # 分母使用 n - 1 (ddof=1)，這是論文中最常用的統計方式
    variance = sum((x - mean) ** 2 for x in scores) / (n - 1) if n > 1 else 0
    std = math.sqrt(variance)
    
    return f"{mean:.4f} ± {std:.4f}"

dataset = 'cadec'
dataset = 'share13'
dataset = 'share14'

if dataset == 'cadec':
    gold_data = f"m1/{dataset}/test.txt"
    pred_data = f'm1/{dataset}/test.pred'
    # gold_data = "m1/gold.txt"
    # pred_data = 'm1/pred.txt'

    gold_entity = read_data(gold_data)
    pred_entity = read_data(pred_data)

    p, r, f1 = evaluate(gold_entity, pred_entity, 'all')
    print(f"Precision: {p:.4}, Recall: {r:.4}, F1: {f1:.4}")
    p, r, f1 = evaluate(gold_entity, pred_entity, 'with_disc')
    print(f"Precision: {p:.4}, Recall: {r:.4}, F1: {f1:.4}")
    p, r, f1 = evaluate(gold_entity, pred_entity, 'only_disc')
    print(f"Precision: {p:.4}, Recall: {r:.4}, F1: {f1:.4}")
    p, r, f1 = evaluate(gold_entity, pred_entity, 'only_c')
    print(f"Precision: {p:.4}, Recall: {r:.4}, F1: {f1:.4}")

else:
    gold_data = f"m1/{dataset}_fixed/test.txt"
    gold_entity = read_data(gold_data)

    p_result = [[] for _ in range(4)]
    r_result = [[] for _ in range(4)]
    f1_result = [[] for _ in range(4)]

    for seed in [52, 869, 1001, 50542, 353778]:
        pred_data = f'm1/{dataset}_fixed/{seed}/test.pred'
        pred_entity = read_data(pred_data)

        p, r, f1 = evaluate(gold_entity, pred_entity, 'all')
        p_result[0].append(p)
        r_result[0].append(r)
        f1_result[0].append(f1)
        print(f"Precision: {p:.4}, Recall: {r:.4}, F1: {f1:.4}")
        p, r, f1 = evaluate(gold_entity, pred_entity, 'with_disc')
        p_result[1].append(p)
        r_result[1].append(r)
        f1_result[1].append(f1)
        print(f"Precision: {p:.4}, Recall: {r:.4}, F1: {f1:.4}")
        p, r, f1 = evaluate(gold_entity, pred_entity, 'only_disc')
        p_result[2].append(p)
        r_result[2].append(r)
        f1_result[2].append(f1)
        print(f"Precision: {p:.4}, Recall: {r:.4}, F1: {f1:.4}")
        p, r, f1 = evaluate(gold_entity, pred_entity, 'only_c')
        p_result[3].append(p)
        r_result[3].append(r)
        f1_result[3].append(f1)
        print(f"Precision: {p:.4}, Recall: {r:.4}, F1: {f1:.4}")
        print()

    for i in range(4):
        print(f"Precision: {calculate_stats(p_result[i])}")
        print(f"Recall   : {calculate_stats(r_result[i])}")
        print(f"F1-score : {calculate_stats(f1_result[i])}")
        print()