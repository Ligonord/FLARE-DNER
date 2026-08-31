import json

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

def evaluate_offset(gold_data, pred_data, mode):
    TP, FP, FN = 0, 0, 0

    for gold_ents, pred_ents in zip(gold_data, pred_data):
        gold_set = set(tuple(span) for span in gold_ents)
        pred_set = set(tuple(span) for span in pred_ents)

        # 判斷是否該句包含非連續
        has_disc = any(has_discontinuous_offset(ent) for ent in gold_ents)

        # 依 mode 過濾
        if mode == "with_disc" and not has_disc:
            continue
        if mode == "only_disc" and not has_disc:
            continue
        if mode == "only_disc":
            gold_set = {ent for ent in gold_set if has_discontinuous_offset(ent)}
            pred_set = {ent for ent in pred_set if has_discontinuous_offset(ent)}
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
        if mode == "with_disc" and not has_disc:
            continue
        if mode == "only_disc" and not has_disc:
            continue
        if mode == "only_disc":
            gold_list = [e for e in gold_list if e not in sen[0]]
            pred_list = [e for e in pred_list if e not in sen[0]]

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

dataset = 'cadec'
dataset = 'share13'
dataset = 'share14'

gold_data_offset = json.load(open(f"m5/{dataset}/test_data.json", "r", encoding="utf-8"))
gold_data_text = read_jsonl(f"m3/{dataset}4yelp/test_data.json")
gpt_data = json.load(open(f'./GPT-5-mini/{dataset}/{dataset}_output.json', "r", encoding="utf-8"))
print(len(gpt_data))

sentence = [[g['text']]for g in gold_data_text]
gold_entity_offset = [[ent["index"] for ent in g["ner"]] for g in gold_data_offset]
gold_entity_text = [[ent['text'] for ent in g["entity_list"]] for g in gold_data_text]
gpt_entity_offset = [[ent["index"] for ent in g["entity_list"]] for g in gpt_data]
gpt_entity_text = [[" ".join(ent["text"]) for ent in g["entity_list"]]for g in gpt_data]

print('Offset')
p, r, f1 = evaluate_offset(gold_entity_offset, gpt_entity_offset, 'all')
print(f"Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")
p, r, f1 = evaluate_offset(gold_entity_offset, gpt_entity_offset, 'with_disc')
print(f"Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")
p, r, f1 = evaluate_offset(gold_entity_offset, gpt_entity_offset, 'only_disc')
print(f"Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")

print('Text:')
p, r, f1 = evaluate_text(gold_entity_text, gpt_entity_text, sentence, 'all')
print(f"Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")
p, r, f1 = evaluate_text(gold_entity_text, gpt_entity_text, sentence, 'with_disc')
print(f"Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")
p, r, f1 = evaluate_text(gold_entity_text, gpt_entity_text, sentence, 'only_disc')
print(f"Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")