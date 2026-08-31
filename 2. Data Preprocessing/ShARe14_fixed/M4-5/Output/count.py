import json

type = 'train'
type = 'valid'
type = 'test'

with open(f'./Output/{type}_data.json', 'r', encoding="utf-8") as f:
    data = json.load(f)

count = 0
for d in data:
    count += len(d['ner'])
print(count)