import json

def read_m1_data(filename):
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]

    i = 0
    while i < len(lines):
        sentence = lines[i].strip()
        if i + 1 < len(lines):
            annotations = lines[i+1].strip()
        else:
            annotations = ""

        entities = []
        if annotations:
            for n in annotations.split("|"):
                coords = [int(x) for x in n.split()[0].split(",")]
                # 兩兩一組 → 展開成完整 index
                expanded = [k for start, end in zip(coords[::2], coords[1::2]) for k in range(start, end+1)]
                entities.append(expanded)
        data.append({
            'sentence': sentence,
            'entity': entities
        })
        
        # 跳過 sentence, annotation, 空行
        i += 3  

    return data

dataset = 'cadec'
dataset = 'share13'
dataset = 'share14'
m1_data = f'm1/{dataset}/test.txt'
m1_data = f'm1/{dataset}_fixed/test.txt'
m1_entity = read_m1_data(m1_data)
# print(m1_entity)

output = []
for i, m1_e in enumerate(m1_entity, start = 1):
    output.append({
        'number': i,
        'sentence': m1_e['sentence'],
        'text': m1_e['sentence'].split(' ')
    })
# with open(f"./GPT-5-mini/{dataset}/{dataset}.json", "w", encoding="utf-8") as f:
#     json.dump(output, f)
with open(f"./GPT-5-mini/{dataset}/{dataset}_fixed.json", "w", encoding="utf-8") as f:
    json.dump(output, f)