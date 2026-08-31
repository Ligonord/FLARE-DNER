import os
import glob
import json
from collections import Counter
import math

def read_jsonl(filename):
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():  # 避免空行
                data.append(json.loads(line))
    return data

def respan_pred_entity(data):
    respan_entity = []
    for d in data:
        respan_num = []
        num = 0
        sen = d['features']['word_list']
        ent = [e['tok_span'] for e in d["entity_list"]]

        i = 0
        while i < len(sen):
            if i == 0 and not sen[i] == '||||':
                respan_num.append(num)
                i += 1
                continue
            
            if sen[i] == '||||':
                count = 1
                for j in range(i + 1, len(sen)):
                    if sen[j] == '||||':
                        count += 1
                    else:
                        break
                
                real_count = (int)(count / 4)
                for j in range(real_count):
                    respan_num.append(num)
                for j in range(count - real_count):
                    num += 1
                    respan_num.append(num)
                i += count
                continue
            
            if sen[i] == sen[i - 1]:
                num += 1
            
            respan_num.append(num)
            i += 1
        respan_num.append(num)

        for i, e in enumerate(ent):
            ent[i] = [a - respan_num[a] for a in e]

        entity = [
            [i for start, end in zip(span[::2], span[1::2]) for i in range(start, end)]
            for span in ent
        ]

        respan_entity.append(ent)
    return respan_entity

def has_discontinuous_offset(ent):
    """判斷實體是否為非連續（有超過2個 index）"""
    parts = list(map(str, ent))  # 抓數字部分
    return len(parts) > 2

def has_continuous_offset(ent):
    """判斷實體是否為非連續（有超過2個 index）"""
    parts = list(map(str, ent))  # 抓數字部分
    return len(parts) == 2

def evaluate_offset(gold_data, pred_data, mode):
    TP, FP, FN = 0, 0, 0

    for gold_ents, pred_ents in zip(gold_data, pred_data):
        gold_set = set(tuple(span) for span in gold_ents)
        pred_set = set(tuple(span) for span in pred_ents)

        # 判斷是否該句包含非連續
        has_disc = any(has_discontinuous_offset(ent) for ent in gold_ents)
        has_c = any(has_continuous_offset(ent) for ent in gold_ents)

        # 依 mode 過濾
        if mode == "only_c" and not has_c:
            continue
        if mode == "with_disc" and not has_disc:
            continue
        if mode == "only_disc" and not has_disc:
            continue
        if mode == "only_disc":
            gold_set = {ent for ent in gold_set if has_discontinuous_offset(ent)}
            pred_set = {ent for ent in pred_set if has_discontinuous_offset(ent)}
        if mode == "only_c":
            gold_set = {ent for ent in gold_set if has_continuous_offset(ent)}
            pred_set = {ent for ent in pred_set if has_continuous_offset(ent)}
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

def has_discontinuous_text(ent, sentence):
    entity = list(ent)
    if entity:
        return any(e not in sentence for e in entity)
    else:
        return False

def has_continuous_text(ent, sentence):
    entity = list(ent)
    if entity:
        return any(e in sentence for e in entity)
    else:
        return False

def evaluate_text(gold_data, pred_data, sentence, mode):
    TP, FP, FN = 0, 0, 0

    '''
    同一句子中，重複出現entity合併做計算，不考慮位置
    gold_ents = ['chest pain', 'chest pain', 'euphoria'], pred_ents = ['chest pain', 'euphoria']
    gold_set = ('chest pain', 'euphoria'), pred_set = ('chest pain', 'euphoria')
    tp: 2 不是 tp: 2 fn: 1，也就是entity總數會有影響
    '''
    # for gold_ents, pred_ents, sen in zip(gold_data, pred_data, sentence):
    #     gold_set = set(span for span in gold_ents)
    #     pred_set = set(span for span in pred_ents)

    #     # 判斷是否該句包含非連續
    #     has_disc = has_discontinuous_text(gold_set, sen[0])

    #     # 依 mode 過濾
    #     if mode == "with_disc" and not has_disc:
    #         continue
    #     if mode == "only_disc" and not has_disc:
    #         continue
    #     if mode == "only_disc":
    #         gold_disc_NE = [e for e in list(gold_set) if e not in sen[0]]
    #         pred_disc_NE = [e for e in list(pred_set) if e not in sen[0]]
    #         gold_set = set(span for span in gold_disc_NE)
    #         pred_set = set(span for span in pred_disc_NE)
    #     # print(gold_set)
    #     # print(pred_set)

    #     TP += len(gold_set & pred_set)   # 預測對了
    #     FP += len(pred_set - gold_set)   # 預測有但標註沒有
    #     FN += len(gold_set - pred_set)   # 標註有但預測沒抓到
    
    for gold_ents, pred_ents, sen in zip(gold_data, pred_data, sentence):
        gold_list = list(gold_ents)
        pred_list = list(pred_ents)
        TP_temp = 0
        pred_used = [False] * len(pred_ents)

        # 判斷是否該句包含非連續
        has_disc = has_discontinuous_text(gold_list, sen[0])

        # 依 mode 過濾
        if mode == "only_c" and not has_continuous_text(gold_list, sen[0]):
            continue
        if mode == "with_disc" and not has_disc:
            continue
        if mode == "only_disc" and not has_disc:
            continue
        if mode == "only_disc":
            gold_list = [e for e in gold_list if e not in sen[0]]
            pred_list = [e for e in pred_list if e not in sen[0]]
        if mode == "only_c":
            gold_list = [e for e in gold_list if e in sen[0]]
            pred_list = [e for e in pred_list if e in sen[0]]

        for g in gold_list:
            for i, p in enumerate(pred_list):
                if not pred_used[i] and g == p:
                    TP_temp += 1
                    pred_used[i] = True
                    break

        TP += TP_temp
        FN += len(gold_list) - TP_temp
        FP += len(pred_list) - TP_temp
    print("TP:", TP, "FP:", FP, "FN:", FN)

    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

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

# gold_data = read_jsonl("m3/gold.json")
# pred_data = read_jsonl("m3/pred.json")
# # for g_item, p_item in zip(gold_data, pred_data):
# #     print("ID:", g_item["id"])
# #     print("Entities:", g_item["entity_list"])
# #     print("Entities:", p_item["entity_list"])
# #     print()

# gold_entity_offset = [[ent['tok_span'] for ent in g["entity_list"]] for g in gold_data]
# pred_entity_offset = respan_pred_entity(pred_data)
# # print(gold_entity_offset)
# # print(pred_entity_offset)
# print('Offset:')
# p, r, f1 = evaluate_offset(gold_entity_offset, pred_entity_offset, 'all')
# print(f"Precision: {p}, Recall: {r}, F1: {f1}")
# p, r, f1 = evaluate_offset(gold_entity_offset, pred_entity_offset, 'with_disc')
# print(f"Precision: {p}, Recall: {r}, F1: {f1}")
# p, r, f1 = evaluate_offset(gold_entity_offset, pred_entity_offset, 'only_disc')
# print(f"Precision: {p}, Recall: {r}, F1: {f1}")
# print()

# sentence = [[g['text']]for g in gold_data]
# gold_entity_text = [[ent['text'] for ent in g["entity_list"]] for g in gold_data]
# pred_entity_text = [[ent['text'] for ent in p["entity_list"]] for p in pred_data]
# # print(sentence)
# # print(gold_entity_text)
# # print(pred_entity_text)
# print('Text:')
# p, r, f1 = evaluate_text(gold_entity_text, pred_entity_text, sentence, 'all')
# print(f"Precision: {p}, Recall: {r}, F1: {f1}")
# p, r, f1 = evaluate_text(gold_entity_text, pred_entity_text, sentence, 'with_disc')
# print(f"Precision: {p}, Recall: {r}, F1: {f1}")
# p, r, f1 = evaluate_text(gold_entity_text, pred_entity_text, sentence, 'only_disc')
# print(f"Precision: {p}, Recall: {r}, F1: {f1}")

dataset = 'cadec'
dataset = 'share13'
dataset = 'share14'
if dataset == "cadec":
    gold_data = read_jsonl(f"m3/{dataset}4yelp/test_data.json")
    base_path = f"m3/{dataset}4yelp"
else:
    gold_data = read_jsonl(f"m3/{dataset}4yelp_fixed/test_data.json")
    base_path = f"m3/{dataset}4yelp_fixed"

p_result = [[] for _ in range(8)]
r_result = [[] for _ in range(8)]
f1_result = [[] for _ in range(8)]
folders = glob.glob(os.path.join(base_path, "model_state_dict_*"))
for target_folder in folders:
    pred_data = read_jsonl(os.path.join(target_folder, "res_data.json"))

    gold_entity_offset = [[ent['tok_span'] for ent in g["entity_list"]] for g in gold_data]
    pred_entity_offset = respan_pred_entity(pred_data)

    print(target_folder)
    print('Offset:')
    p, r, f1 = evaluate_offset(gold_entity_offset, pred_entity_offset, 'all')
    p_result[0].append(p)
    r_result[0].append(r)
    f1_result[0].append(f1)
    print(f"Precision: {p:.4}, Recall: {r:.4}, F1: {f1:.4}")
    p, r, f1 = evaluate_offset(gold_entity_offset, pred_entity_offset, 'with_disc')
    p_result[1].append(p)
    r_result[1].append(r)
    f1_result[1].append(f1)
    print(f"Precision: {p:.4}, Recall: {r:.4}, F1: {f1:.4}")
    p, r, f1 = evaluate_offset(gold_entity_offset, pred_entity_offset, 'only_disc')
    p_result[2].append(p)
    r_result[2].append(r)
    f1_result[2].append(f1)
    print(f"Precision: {p:.4}, Recall: {r:.4}, F1: {f1:.4}")
    p, r, f1 = evaluate_offset(gold_entity_offset, pred_entity_offset, 'only_c')
    p_result[3].append(p)
    r_result[3].append(r)
    f1_result[3].append(f1)
    print(f"Precision: {p:.4}, Recall: {r:.4}, F1: {f1:.4}")
    print()

    sentence = [[g['text']]for g in gold_data]
    gold_entity_text = [[ent['text'] for ent in g["entity_list"]] for g in gold_data]
    pred_entity_text = [[ent['text'] for ent in p["entity_list"]] for p in pred_data]

    print('Text:')
    p, r, f1 = evaluate_text(gold_entity_text, pred_entity_text, sentence, 'all')
    p_result[4].append(p)
    r_result[4].append(r)
    f1_result[4].append(f1)
    print(f"Precision: {p:.4}, Recall: {r:.4}, F1: {f1:.4}")
    p, r, f1 = evaluate_text(gold_entity_text, pred_entity_text, sentence, 'with_disc')
    p_result[5].append(p)
    r_result[5].append(r)
    f1_result[5].append(f1)
    print(f"Precision: {p:.4}, Recall: {r:.4}, F1: {f1:.4}")
    p, r, f1 = evaluate_text(gold_entity_text, pred_entity_text, sentence, 'only_disc')
    p_result[6].append(p)
    r_result[6].append(r)
    f1_result[6].append(f1)
    print(f"Precision: {p:.4}, Recall: {r:.4}, F1: {f1:.4}")
    p, r, f1 = evaluate_text(gold_entity_text, pred_entity_text, sentence, 'only_c')
    p_result[7].append(p)
    r_result[7].append(r)
    f1_result[7].append(f1)
    print(f"Precision: {p:.4}, Recall: {r:.4}, F1: {f1:.4}")
    print()

    # sum1=0
    # sum2=0
    # for i, (a, b) in enumerate(zip(gold_entity, gold_entity_text)):
    #     sum1+=len(a)
    #     sum2+=len(b)
    #     if len(a) != len(b):
    #         print(f"index {i}: len1={len(a)}, len2={len(b)}")
    # print(sum1,sum2)
    # for i, ents in enumerate(gold_entity_text):
    #     cnt = Counter(ents)
    #     dup = [k for k, v in cnt.items() if v > 1]
    #     if dup:
    #         print(i, dup)
    # print()

for i in range(8):
    print(f"Precision: {calculate_stats(p_result[i])}")
    print(f"Recall   : {calculate_stats(r_result[i])}")
    print(f"F1-score : {calculate_stats(f1_result[i])}")
    print()