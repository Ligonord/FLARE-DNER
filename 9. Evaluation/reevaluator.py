import json

def read_jsonl(filename):
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():  # 避免空行
                data.append(json.loads(line))
    return data

def evaluate_offset(gold_data, pred_data):
    TP, FP, FN = 0, 0, 0

    for gold_ents, pred_ents in zip(gold_data, pred_data):
        gold_set = set(tuple(span) for span in gold_ents)
        pred_set = set(tuple(span) for span in pred_ents)
        # print(gold_set)
        # print(pred_set)

        TP += len(gold_set & pred_set)   # 預測對了
        FP += len(pred_set - gold_set)   # 預測有但標註沒有
        FN += len(gold_set - pred_set)   # 標註有但預測沒抓到
    print("TP:", TP, "FP:", FP, "FN:", FN)

    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    return precision, recall, f1

dataset = 'cadec'
dataset = 'share13'
dataset = 'share14'

if dataset == 'cadec':
    gold_data_offset = json.load(open(f"m5/{dataset}/test_data.json", "r", encoding="utf-8"))
    based_data = json.load(open(f'./format_alignment/{dataset}_ensemble.json', "r", encoding="utf-8"))
    vote_data = json.load(open(f'./ensemble_voting/{dataset}_voting_2.json', "r", encoding="utf-8"))
else:
    gold_data_offset = json.load(open(f"m5/{dataset}_fixed/test_data.json", "r", encoding="utf-8"))
    based_data = json.load(open(f'./format_alignment/{dataset}_ensemble_fixed.json', "r", encoding="utf-8"))
    vote_data = json.load(open(f'./ensemble_voting/{dataset}_voting_2_fixed.json', "r", encoding="utf-8"))
gpt_data = json.load(open(f'./ensemble_gpt/3/{dataset}_output.json', "r", encoding="utf-8"))
rag_data = json.load(open(f'./rag/12/{dataset}_output.json', "r", encoding="utf-8"))
qbc_data = json.load(open(f"./qbc_voting_rag/{dataset}_qbc.json", "r", encoding="utf-8"))
llm_auth_data = json.load(open(f"./llm_data_evaluation/{dataset}_llm_authed.json", "r", encoding="utf-8"))

gold_entity_offset = [[ent["index"] for ent in g["ner"]] for g in gold_data_offset]
vote_entity_offset = [v['ner_offset'] for v in vote_data]
gpt_entity_offset = [[ent["index"] for ent in g["entity_list"]] for g in gpt_data]
rag_entity_offset = [[ent["index"] for ent in g["entity_list"]] for g in rag_data]
qbc_entity_offset = [q['entity_offset'] for q in qbc_data]

for a in llm_auth_data:
    gold_entity_offset[a['number'] - 1].append(a['entity_index'])

print('Micro-F1 Offset:')
for i in range(1, 6):
    based_entity_offset = [[ent['index'] for ent in g[f"entity_list_{i}"]] for g in based_data]

    print(f'Based Model {i}:')
    evaluate_offset(gold_entity_offset, based_entity_offset)
    print()
print('Vote:')
evaluate_offset(gold_entity_offset, vote_entity_offset)
print()
print('Ensemble GPT:')
evaluate_offset(gold_entity_offset, gpt_entity_offset)
print()
print('RAG GPT:')
evaluate_offset(gold_entity_offset, rag_entity_offset)
print()
print('QBC:')
evaluate_offset(gold_entity_offset, qbc_entity_offset)