import os
import glob
import json
import networkx as nx

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

def get_m2_entity(data):
    entity = []
    for d in data:
        G = nx.Graph()
        rel = d.get('relation')
        ner = d.get('ner')
        ner = [n[:-1] for n in ner]

        dner_rel = [r for r in rel if r[-1] == "Combined"]
        for dr in dner_rel:
            node1 = tuple(dr[:2])
            node2 = tuple(dr[2:4])
            G.add_edge(node1, node2)
        
        # 找所有 maximal cliques
        dner_slice = [[list(t) for t in sublist] for sublist in list(nx.find_cliques(G))]
        for slice in dner_slice:
            for s in slice:
                if s in ner:
                    ner.pop(ner.index(s))

        # 依照每個節點的第一個值排序，然後展平成單一 list
        dner = [sum(sorted(s, key=lambda x: x[0]), []) for s in dner_slice]
        ner.extend(dner)
        e = []
        if ner:
            for n in ner:
                expanded = [k for start, end in zip(n[::2], n[1::2]) for k in range(start, end+1)]
                e.append(expanded)
        entity.append(e)
    return entity

def read_m3_data(filename):
    data = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():  # 避免空行
                data.append(json.loads(line))
    return data

def respan_m3_pred_entity(data):
    respan_entity = []
    for d in data:
        respan_num = []
        num = 0
        sen = d['features']['word_list']
        ent = [e['tok_span'] for e in d["entity_list"]]

        i = 0
        while i < len(sen):
            if i == 0 and not sen[i] == '||||':
                respan_num.append(num)
                i += 1
                continue
            
            if sen[i] == '||||':
                count = 1
                for j in range(i + 1, len(sen)):
                    if sen[j] == '||||':
                        count += 1
                    else:
                        break
                
                real_count = (int)(count / 4)
                for j in range(real_count):
                    respan_num.append(num)
                for j in range(count - real_count):
                    num += 1
                    respan_num.append(num)
                i += count
                continue
            
            if sen[i] == sen[i - 1]:
                num += 1
            
            respan_num.append(num)
            i += 1
        respan_num.append(num)

        for i, e in enumerate(ent):
            ent[i] = [a - respan_num[a] for a in e]

        entity = [
            [i for start, end in zip(span[::2], span[1::2]) for i in range(start, end)]
            for span in ent
        ]

        respan_entity.append(entity)
    return respan_entity

dataset = 'cadec'
dataset = 'share13'
dataset = 'share14'

m1_data = f'input/m1/{dataset}_fixed/1001/test.pred'
m1_entity = read_m1_data(m1_data)
# print(m1_entity)
m2_path = f'input/m2/{dataset}_fixed/{dataset}_prediction.txt'
m2_data = []
with open(m2_path, "r", encoding="utf-8") as f:
    for line in f:
        m2_data.append(json.loads(line))
m2_data = [item for sublist in m2_data for item in sublist]
m2_entity = get_m2_entity(m2_data)
# print(m2_entity)
m3_base = f"input/m3/{dataset}4yelp_fixed"
m3_folders = glob.glob(os.path.join(m3_base, "model_state_dict_*"))
m3_data = read_m3_data(os.path.join(m3_folders[0], "res_data.json"))
m3_entity = respan_m3_pred_entity(m3_data)
# print(m3_entity)
m4_data = json.load(open(f"input/m4/{dataset}_fixed/{dataset}_output.json", "r", encoding="utf-8"))
m4_entity = [[ent["index"] for ent in p["entity"]] for p in m4_data]
# print(m4_entity)
m5_data = json.load(open(f"input/m5/{dataset}_fixed/{dataset}_predictions.json", "r", encoding="utf-8"))
m5_entity = [[ent for ent in p["entity"]] for p in m5_data]
# print(m5_entity)

# m1_data = f'input/m1/{dataset}/test.pred'
# m1_entity = read_m1_data(m1_data)
# # print(m1_entity)
# m2_path = f'm2/{dataset}/{dataset}_prediction.txt'
# m2_data = []
# with open(input/m2_path, "r", encoding="utf-8") as f:
#     for line in f:
#         m2_data.append(json.loads(line))
# m2_data = [item for sublist in m2_data for item in sublist]
# m2_entity = get_m2_entity(m2_data)
# # print(m2_entity)
# m3_base = f"input/m3/{dataset}4yelp"
# m3_folders = glob.glob(os.path.join(m3_base, "model_state_dict_*"))
# m3_data = read_m3_data(os.path.join(m3_folders[0], "res_data.json"))
# m3_entity = respan_m3_pred_entity(m3_data)
# # print(m3_entity)
# m4_data = json.load(open(f"input/m4/{dataset}/{dataset}_output.json", "r", encoding="utf-8"))
# m4_entity = [[ent["index"] for ent in p["entity"]] for p in m4_data]
# # print(m4_entity)
# m5_data = json.load(open(f"input/m5/{dataset}/{dataset}_predictions.json", "r", encoding="utf-8"))
# m5_entity = [[ent for ent in p["entity"]] for p in m5_data]
# # print(m5_entity)

output = []
for i, (m1_e, m2_e, m3_e, m4_e, m5_e) in enumerate(zip(m1_entity, m2_entity, m3_entity, m4_entity, m5_entity), start = 1):
    m1e = []
    m2e = []
    m3e = []
    m4e = []
    m5e = []
    for e in m1_e['entity']:
        m1e.append({
            'text': [m1_e['sentence'].split(' ')[i] for i in e],
            'index': e
        })
    for e in m2_e:
        m2e.append({
            'text': [m1_e['sentence'].split(' ')[i] for i in e],
            'index': e
        })
    for e in m3_e:
        m3e.append({
            'text': [m1_e['sentence'].split(' ')[i] for i in e],
            'index': e
        })
    for e in m4_e:
        m4e.append({
            'text': [m1_e['sentence'].split(' ')[i] for i in e],
            'index': e
        })
    for e in m5_e:
        m5e.append({
            'text': [m1_e['sentence'].split(' ')[i] for i in e],
            'index': e
        })

    output.append({
        'number': i,
        'sentence': m1_e['sentence'],
        'text': m1_e['sentence'].split(' '),
        'entity_list_1': m1e,
        'entity_list_2': m2e,
        'entity_list_3': m3e,
        'entity_list_4': m4e,
        'entity_list_5': m5e
    })
# with open(f"./output/{dataset}_ensemble.json", "w", encoding="utf-8") as f:
#     json.dump(output, f)
with open(f"./output/{dataset}_ensemble_fixed.json", "w", encoding="utf-8") as f:
    json.dump(output, f)