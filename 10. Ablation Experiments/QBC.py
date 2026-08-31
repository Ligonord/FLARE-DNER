import json
from collections import Counter
from scipy.stats import entropy
import numpy as np
from scipy.optimize import linear_sum_assignment

def read_jsonl(filename):
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():  # 避免空行
                data.append(json.loads(line))
    return data

def has_discontinuous_offset(ent):
    """判斷數字列表是否連續"""
    if not ent:  # 空列表視為 False
        return False
    ent = list(ent)
    return not ent == list(range(ent[0], ent[-1] + 1))

def has_continuous_offset(ent):
    """判斷數字列表是否連續"""
    if not ent:  # 空列表視為 False
        return False
    ent = list(ent)
    return ent == list(range(ent[0], ent[-1] + 1))

def evaluate_offset(gold_data, pred_data, mode):
    TP, FP, FN = 0, 0, 0

    for gold_ents, pred_ents in zip(gold_data, pred_data):
        gold_set = set(tuple(span) for span in gold_ents)
        pred_set = set(tuple(span) for span in pred_ents)

        # 判斷是否該句包含非連續
        has_disc = any(has_discontinuous_offset(ent) for ent in gold_ents)

        # 依 mode 過濾
        if mode == "only_c" and not any(has_continuous_offset(ent) for ent in gold_ents):
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

def relaxed_matching(gold_entity_offset, gpt_entity_offset):
    gold_count = 0
    pred_count = 0
    TP = 0
    for g_o, m_o in zip(gold_entity_offset, gpt_entity_offset):
        gold_count += len(g_o)
        pred_count += len(m_o)
        for g in g_o:
            for p in m_o:
                if set(g).issubset(set(p)):
                    TP += 1
                    break
    FN = gold_count - TP
    FP = pred_count - TP
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    print('Relaxed Matching:')
    print("TP:", TP, "FP:", FP, "FN:", FN)
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

def discontinuous_offset(ent):
    """判斷數字列表是否連續"""
    if not ent:  # 空列表視為 False
        return False
    ent = list(ent)
    return not ent == list(range(ent[0], ent[-1] + 1))

def entity_f1_offset(gold_entity_offset, pred_entity_offset):
    TP_C, FP_C, FN_C, TP_D, FP_D, FN_D = 0, 0, 0, 0, 0, 0

    for gold_ents, pred_ents in zip(gold_entity_offset, pred_entity_offset):
        gold_set = set(tuple(span) for span in gold_ents)
        pred_set = set(tuple(span) for span in pred_ents)

        for ent in gold_set:
            if discontinuous_offset(ent):
                if ent in pred_set:
                    TP_D += 1
                else:
                    FN_D += 1
            else:
                if ent in pred_set:
                    TP_C += 1
                else:
                    FN_C += 1

        for ent in pred_set:
            if discontinuous_offset(ent):
                if ent not in gold_set:
                    FP_D += 1
            else:
                if ent not in gold_set:
                    FP_C += 1

    precision_C = TP_C / (TP_C + FP_C) if TP_C + FP_C > 0 else 0
    recall_C = TP_C / (TP_C + FN_C) if TP_C + FN_C > 0 else 0
    f1_C = 2 * precision_C * recall_C / (precision_C + recall_C) if precision_C + recall_C > 0 else 0
    precision_D = TP_D / (TP_D + FP_D) if TP_D + FP_D > 0 else 0
    recall_D = TP_D / (TP_D + FN_D) if TP_D + FN_D > 0 else 0
    f1_D = 2 * precision_D * recall_D / (precision_D + recall_D) if precision_D + recall_D > 0 else 0

    print('Macro-F1:')
    print(f"Precision: {(precision_C + precision_D) / 2:.4f}, Recall: {(recall_C + recall_D) / 2:.4f}, F1: {(f1_C + f1_D) / 2:.4f}")
    print()

    weight_C = TP_C / (TP_C + TP_D)
    weight_D = TP_D / (TP_C + TP_D)
    print('Weighted-F1:')
    print(f"Precision: {precision_C * weight_C + precision_D * weight_D:.4f}, Recall: {recall_C * weight_C + recall_D * weight_D:.4f}, F1: {f1_C * weight_C + f1_D * weight_D:.4f}")

def sen_instance_f1(gold_entity_offset, pred_entity_offset):
    p, r, f1 = 0, 0, 0

    for gold_ents, pred_ents in zip(gold_entity_offset, pred_entity_offset):
        TP, FP, FN, p_temp, r_temp, f1_temp = 0, 0, 0, 0, 0, 0

        gold_set = set(tuple(span) for span in gold_ents)
        pred_set = set(tuple(span) for span in pred_ents)

        TP += len(gold_set & pred_set)   # 預測對了
        FP += len(pred_set - gold_set)   # 預測有但標註沒有
        FN += len(gold_set - pred_set)   # 標註有但預測沒抓到

        if len(gold_set) == 0 and len(pred_set) == 0:
            p_temp = 1.0
            r_temp = 1.0
            f1_temp = 1.0
        else:
            p_temp = TP / (TP + FP) if (TP + FP) > 0 else 0
            r_temp = TP / (TP + FN) if (TP + FN) > 0 else 0
            f1_temp = 2 * p_temp * r_temp / (p_temp + r_temp) if (p_temp + r_temp) > 0 else 0

        p += p_temp
        r += r_temp
        f1 += f1_temp

    p /= len(gold_entity_offset)
    r /= len(gold_entity_offset)
    f1 /= len(gold_entity_offset)
    
    print('Sentence Instance-F1:')
    print(f"Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")

def relaxed_matching_jaccard_hungarian(gold_entity_offset, pred_entity_offset):
    gold_count = 0
    pred_count = 0
    total_TP_score = 0.0  # TP 是基於 Jaccard 相似度的累加小數分數
    
    for g_o, m_o in zip(gold_entity_offset, pred_entity_offset):
        gold_count += len(g_o)
        pred_count += len(m_o)
        
        if not g_o or not m_o:
            continue
            
        # 建立該句子的匹配分數矩陣 (Gold x Pred)
        score_matrix = np.zeros((len(g_o), len(m_o)))
        
        for g_idx, g_list in enumerate(g_o):
            g_set = set(g_list)
            
            for p_idx, p_list in enumerate(m_o):
                p_set = set(p_list)
                
                intersection_size = len(g_set.intersection(p_set))
                
                if intersection_size > 0:
                    # 計算 Jaccard 相似度: 交集 / 聯集
                    union_size = len(g_set.union(p_set))
                    score_matrix[g_idx, p_idx] = intersection_size / union_size
                else:
                    # 完全沒有重疊
                    score_matrix[g_idx, p_idx] = 0.0

        # 使用匈牙利演算法解決一對多問題
        cost_matrix = 1.0 - score_matrix
        gold_ind, pred_ind = linear_sum_assignment(cost_matrix)
        
        # 累加本句最佳匹配後的 Jaccard TP 分數
        for g, p in zip(gold_ind, pred_ind):
            actual_score = score_matrix[g, p]
            total_TP_score += actual_score

    # 計算最終指標
    TP = total_TP_score
    FN = max(0.0, gold_count - TP)
    FP = max(0.0, pred_count - TP)
    
    precision = TP / pred_count if pred_count > 0 else 0.0
    recall = TP / gold_count if gold_count > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    print('Relaxed Matching (Jaccard & Hungarian):')
    print(f"TP: {TP:.4f}, FP: {FP:.4f}, FN: {FN:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "TP": TP,
        "FP": FP,
        "FN": FN
    }

def token_coverage(gold_entity_offset, pred_entity_offset):
    total_TP = 0
    total_FP = 0
    total_FN = 0
    
    for g_o, m_o in zip(gold_entity_offset, pred_entity_offset):
        # 將這句話所有實體的 token index 全部打平並取聯集 (Union)
        # 例如: [[1, 2], [1, 5]] -> {1, 2, 5}
        gold_tokens = set().union(*[set(g) for g in g_o]) if g_o else set()
        pred_tokens = set().union(*[set(p) for p in m_o]) if m_o else set()
        
        # 計算 Token 層級的 TP, FP, FN
        tp = len(gold_tokens.intersection(pred_tokens))
        fp = len(pred_tokens.difference(gold_tokens))
        fn = len(gold_tokens.difference(pred_tokens))
        
        total_TP += tp
        total_FP += fp
        total_FN += fn

    # 計算最終指標
    precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0.0
    recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    print('Token Coverage:')
    print(f"TP: {total_TP}, FP: {total_FP}, FN: {total_FN}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    return {"precision": precision, "recall": recall, "f1": f1}

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
    gold_data_offset = json.load(open(f"./input/m5/{dataset}/test_data.json", "r", encoding="utf-8"))
    gold_data_text = read_jsonl(f"./input/m3/{dataset}4yelp/test_data.json")
    ensemble_data = json.load(open(f'./input/format alignment/{dataset}_ensemble.json', "r", encoding="utf-8"))
    vote_data = json.load(open(f'./input/meta_learner/{dataset}_voting_2.json', "r", encoding="utf-8"))
else:
    gold_data_offset = json.load(open(f"./input/m5/{dataset}_fixed/test_data.json", "r", encoding="utf-8"))
    gold_data_text = read_jsonl(f"./input/m3/{dataset}4yelp_fixed/test_data.json")
    ensemble_data = json.load(open(f'./input/format alignment/{dataset}_ensemble_fixed.json', "r", encoding="utf-8"))
    vote_data = json.load(open(f'./input/meta_learner/{dataset}_voting_2_fixed.json', "r", encoding="utf-8"))
rag_data = json.load(open(f'./input/gpt/{dataset}_output.json', "r", encoding="utf-8"))

sentence = [[g['text']] for g in gold_data_text]
gold_entity_offset = [[ent["index"] for ent in g["ner"]] for g in gold_data_offset]
gold_entity_text = [[ent['text'] for ent in g["entity_list"]] for g in gold_data_text]
vote_entity_offset = [[ent for ent in g["ner_offset"]] for g in vote_data]
vote_entity_text = [[ent for ent in g["ner_text"]]for g in vote_data]
rag_entity_offset = [[ent["index"] for ent in g["entity_list"]] for g in rag_data]
rag_entity_text = [[" ".join(ent["text"]) for ent in g["entity_list"]]for g in rag_data]
merge_entity_offset = []
merge_entity_text = []

out = []
for i, (v_o, v_t, r_o, r_t, e_d, sen) in enumerate(zip(vote_entity_offset, vote_entity_text, rag_entity_offset, rag_entity_text, ensemble_data, sentence)):
    count = vote_entropy(i, merge_entity_offset, merge_entity_text, v_o, v_t, r_o, r_t, e_d, sen, base, entropy_threshold, voting_threshold, out)

print('Offset')
p, r, f1 = evaluate_offset(gold_entity_offset, merge_entity_offset, 'all')
print(f"Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")
p, r, f1 = evaluate_offset(gold_entity_offset, merge_entity_offset, 'with_disc')
print(f"Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")
p, r, f1 = evaluate_offset(gold_entity_offset, merge_entity_offset, 'only_disc')
print(f"Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")
p, r, f1 = evaluate_offset(gold_entity_offset, merge_entity_offset, 'only_c')
print(f"Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")

print('Text:')
p, r, f1 = evaluate_text(gold_entity_text, merge_entity_text, sentence, 'all')
print(f"Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")
p, r, f1 = evaluate_text(gold_entity_text, merge_entity_text, sentence, 'with_disc')
print(f"Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")
p, r, f1 = evaluate_text(gold_entity_text, merge_entity_text, sentence, 'only_disc')
print(f"Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")
p, r, f1 = evaluate_text(gold_entity_text, merge_entity_text, sentence, 'only_c')
print(f"Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")
print()

relaxed_matching(gold_entity_offset, merge_entity_offset)
print()
relaxed_matching_jaccard_hungarian(gold_entity_offset, merge_entity_offset)
print()
token_coverage(gold_entity_offset, merge_entity_offset)
print()
entity_f1_offset(gold_entity_offset, merge_entity_offset)
print()
sen_instance_f1(gold_entity_offset, merge_entity_offset)