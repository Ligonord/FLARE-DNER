import json
from collections import Counter

threshold = 2
dataset = 'cadec'
dataset = 'share13'
dataset = 'share14'
if dataset == 'cadec':
    vote_data = json.load(open(f'./input/{dataset}_ensemble.json', "r", encoding="utf-8"))
else:
    # vote_data = json.load(open(f'./input/{dataset}_ensemble.json', "r", encoding="utf-8"))
    vote_data = json.load(open(f'./input/{dataset}_ensemble_fixed.json', "r", encoding="utf-8"))

vote_entity_offset = []
vote_entity_text = []
result = []
for d in vote_data:
    appear_offset = []
    appear_text = []

    for k in range(1, 6):
        for ent in d[f'entity_list_{k}']:
            appear_offset.append(ent['index'])
            appear_text.append(" ".join(ent['text']))

    count_offset = Counter(tuple(l) for l in appear_offset)
    count_text = Counter(appear_text)
    entity_offset = [list(k) for k, v in count_offset.items() if v > threshold]
    entity_text = [k for k, v in count_text.items() if v > threshold]

    vote_entity_offset.append(entity_offset)
    vote_entity_text.append(entity_text)
    result.append({
        'sentence': d['sentence'],
        'ner_offset': entity_offset,
        'ner_text': entity_text,
    })

# with open(f"./output/{dataset}_voting_{threshold}.json", "w", encoding="utf-8") as f:
#     json.dump(result, f)
with open(f"./output/{dataset}_voting_{threshold}_fixed.json", "w", encoding="utf-8") as f:
    json.dump(result, f)