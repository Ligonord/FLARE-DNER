import json

dataset = 'cadec'
# dataset = 'share13'
# dataset = 'share14'
# gpt_model = 'gpt-5'
gpt_model = 'gpt-5-mini'
# method = 'chinese_select by GPT'
method = 'chinese_select from models'
gpt_data = json.load(open(f'./data/output/{dataset}_output.json', "r", encoding="utf-8"))
print(len(gpt_data))