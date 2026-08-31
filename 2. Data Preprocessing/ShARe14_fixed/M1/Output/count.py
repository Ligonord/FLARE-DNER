def read_data(filename):
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
        
        entities = annotations.split("|") if annotations else []
        data.append((sentence, entities))
        
        # 跳過 sentence, annotation, 空行
        i += 3  

    return data


type = 'train'
type = 'dev'
type = 'test'
data = f"Output/{type}.txt"
entity = read_data(data)

count = 0
for e in entity:
    count += len(e[1])
print(count)