import json

dataset = 'cadec'
dataset = 'share13'
dataset = 'share14'

with open(f'./data/input/{dataset}_llm_auth.json', 'r', encoding="utf-8") as f:
    pre_data = json.load(f)
print(len(pre_data))

with open(f'./data/output/{dataset}_output.json', 'r', encoding="utf-8") as f:
    data = json.load(f)
print(len(data))

prenum = []
for p in pre_data:
    prenum.append(p['number'])
num=[]
for d in data:
    num.append(d['number'])

for p in prenum:
    if p not in num:
        print(p)