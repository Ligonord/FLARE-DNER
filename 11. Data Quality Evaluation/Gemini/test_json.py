import json

dataset = 'cadec'
dataset = 'share13'
dataset = 'share14'

with open(f'./output/{dataset}_output.json', 'r', encoding="utf-8") as f:
    data = json.load(f)
print(len(data))