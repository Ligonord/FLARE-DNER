import json

for dataset in ['cadec', 'share13', 'share14']:
    gemini_data = json.load(open(f'./input/gemini/{dataset}_output.json', "r", encoding="utf-8"))
    llama_data = json.load(open(f'./input/llama/{dataset}_output.json', "r", encoding="utf-8"))
    print(len(gemini_data), len(llama_data))

    new_data = []
    for g, l in zip(gemini_data, llama_data):
        if g['is_entity'] == 'True' and l['is_entity']:
            new_data.append({
                'number': g['number'],
                'entity_index': g['entity']['index'],
                'entity_text': g['entity']['text']
            })
    print(len(new_data))

    with open(f"./output/post/{dataset}_llm_authed.json", "w", encoding="utf-8") as f:
        json.dump(new_data, f)