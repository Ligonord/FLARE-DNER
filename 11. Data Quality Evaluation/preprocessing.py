import json

for dataset in ['cadec', 'share13', 'share14']:
    if dataset == 'cadec':
        gold = json.load(open(f"./input/m5/{dataset}/test_data.json", "r", encoding="utf-8"))
    else:
        gold = json.load(open(f"./input/m5/{dataset}_fixed/test_data.json", "r", encoding="utf-8"))
    data = json.load(open(f'./input/rag/{dataset}_rag.json', "r", encoding="utf-8"))
    out = []

    # print(len(data))
    for g, d in zip(gold, data):
        for e in d['entity_list']:
            if e['weight'] > 2 and g['ner'] == []:
                out.append({
                    'number': d['number'],
                    'sentence': d['sentence'],
                    'entity': {'text': e['text'], 'index': e['index']},
                    'is_entity': None
                })

    print(len(out))

    with open(f"./output/pre/{dataset}_llm_auth.json", "w", encoding="utf-8") as f:
        json.dump(out, f)