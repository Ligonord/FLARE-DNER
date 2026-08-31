import json
import networkx as nx

type = 'train'
type = 'dev'
type = 'test'

data = []
with open(f'./Output/{type}.json', "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

entity = []
for g in data:
    G = nx.Graph()
    rel = g.get('relations')[0]
    gold = g.get('ner')[0]
    gold = [g[:-1] for g in gold]

    dner_rel = [r for r in rel if r[-1] == "Combined"]
    for dr in dner_rel:
        node1 = tuple(dr[:2])
        node2 = tuple(dr[2:4])
        G.add_edge(node1, node2)
    
    # 找所有 maximal cliques
    dner_slice = [[list(t) for t in sublist] for sublist in list(nx.find_cliques(G))]
    for slice in dner_slice:
        for s in slice:
            if s in gold:
                gold.pop(gold.index(s))

    # 依照每個節點的第一個值排序，然後展平成單一 list
    dner = [sum(sorted(s, key=lambda x: x[0]), []) for s in dner_slice]
    gold.extend(dner)
    entity.append(gold)
# print(entity)

count = 0
for e in entity:
    count += len(e)
print(count)