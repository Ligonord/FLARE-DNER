import json

def read_jsonl(filename):
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():  # 避免空行
                data.append(json.loads(line))
    return data

dataset = 'cadec'
dataset = 'share13'
dataset = 'share14'

expt_num = 12

if dataset == 'cadec':
    gold_data_offset = json.load(open(f"./m5/{dataset}/test_data.json", "r", encoding="utf-8"))
    gold_data_text = read_jsonl(f"./m3/{dataset}4yelp/test_data.json")
else:
    gold_data_offset = json.load(open(f"./m5/{dataset}_fixed/test_data.json", "r", encoding="utf-8"))
    gold_data_text = read_jsonl(f"./m3/{dataset}4yelp_fixed/test_data.json")
gpt_data = json.load(open(f'./rag/{expt_num}/{dataset}_output.json', "r", encoding="utf-8"))
rag_pre_data = json.load(open(f'./rag/rag_pre/{dataset}_rag_pre.json', 'r', encoding="utf-8"))
print(len(gpt_data))
print(len(rag_pre_data))

sentence = [[g['text']]for g in gold_data_text]
gold_entity_offset = [[ent["index"] for ent in g["ner"]] for g in gold_data_offset]
gold_entity_text = [[ent['text'] for ent in g["entity_list"]] for g in gold_data_text]
gpt_entity_offset = [[ent["index"] for ent in g["entity_list"]] for g in gpt_data]
gpt_entity_text = [[" ".join(ent["text"]) for ent in g["entity_list"]]for g in gpt_data]
ensemble_entity_offset = []
ensemble_entity_text = []
for e in rag_pre_data:
    temp_offset = []
    temp_text = []
    e_l = e[f'entity_list']
    temp_offset.extend([ent['index']for ent in e_l])
    temp_text.extend([" ".join(ent["text"]) for ent in e_l])
    ensemble_entity_offset.append(temp_offset)
    ensemble_entity_text.append(temp_text)

for i, (gpt_ents_offset, ens_ents_offset, gpt_ents_text, ens_ents_text, gold_ents_offset, gold_ents_text) in enumerate(zip(gpt_entity_offset, ensemble_entity_offset, gpt_entity_text, ensemble_entity_text, gold_entity_offset, gold_entity_text)):
    for g_o, g_t in zip(gpt_ents_offset, gpt_ents_text):
        if g_o not in ens_ents_offset:
            print(i, g_o, g_t)
            if g_o in gold_ents_offset:
                print("Goal!")
        if g_t not in ens_ents_text:
            print(i, g_o, g_t)
            if g_t in gold_ents_text:
                print("Goal!")