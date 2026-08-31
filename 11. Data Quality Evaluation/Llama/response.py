import transformers
import torch
import json
import os
import ast

model = "./Llama-3.1-8B-Instruct"

generator = transformers.pipeline(
    "text-generation",
    model = model,
    device = 'cuda',
    torch_dtype = torch.bfloat16
)

dataset = 'cadec'
dataset = 'share13'
dataset = 'share14'

output_file = f'./data/output/{dataset}_output.json'
# output_file = f'./data/output/test_output.json'
with open(f'./data/input/{dataset}_llm_auth.json', 'r', encoding="utf-8") as f:
    data = json.load(f)
# with open(f'./data/input/test.json', 'r', encoding="utf-8") as f:
#     ner_data = json.load(f)

if dataset == 'cadec':
    with open(f'./data/prompt/cadec_prompt_system.txt', 'r', encoding="utf-8") as f:
        prompt_system = f.read()
    with open(f'./data/prompt/cadec_prompt_user.txt', 'r', encoding="utf-8") as f:
        prompt_user = f.read()
else:
    with open(f'./data/prompt/share_prompt_system.txt', 'r', encoding="utf-8") as f:
        prompt_system = f.read()
    with open(f'./data/prompt/share_prompt_user.txt', 'r', encoding="utf-8") as f:
        prompt_user = f.read()

if not os.path.exists(output_file):
    with open(output_file, 'w', encoding="utf-8") as f:
        json.dump([], f)

start_index = 0
total_size = len(data)
print('total number:', total_size)

for i in range(start_index, total_size):
    d = data[i]
    print(f"目前處理 number {i + 1}")

    try:
        # 送出一次 API Request
        response = generator(
            [
                {"role": "system", "content": prompt_system},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"{prompt_user}{d}"
                        }
                    ]
                }
            ],
            do_sample = True,
            temperature = 1.0,
            top_p = 1,
            max_new_tokens = 256
        )
    except Exception as e:
        print('Requset失敗')
        print(e)
        retry_count += 1
        continue

    with open(output_file, 'a', encoding="utf-8") as f:
        f.write(response[0]["generated_text"][-1]['content'])

# for i in range(start_index, total_size):
#     data = data[i]
#     print(f"目前處理 number {i + 1}")

#     correct = False
#     retry_count = 0
#     while not correct and retry_count <= 10:
#         try:
#             # 送出一次 API Request
#             response = generator(
#                 [
#                     {"role": "system", "content": prompt_system},
#                     {
#                         "role": "user",
#                         "content": [
#                             {
#                                 "type": "input_text",
#                                 "text": f"{prompt_user}{data}"
#                             }
#                         ]
#                     }
#                 ],
#                 do_sample = True,
#                 temperature = 1.0,
#                 top_p = 1,
#                 max_new_tokens = 256
#             )
#         except Exception as e:
#             print('Requset失敗')
#             print(e)
#             retry_count += 1
#             continue

#         try:
#             res = ast.literal_eval(response[0]["generated_text"][-1]['content'])
#             for ent in res[0]['entity_list']:
#                 for i in range(len(ent['text'])):
#                     if ent['text'][i] == "\'":
#                         ent['text'][i] = "'"
#             with open('./data/temp.json', 'w', encoding="utf-8") as f:
#                 json.dump(res, f)
#             with open('./data/temp.json', 'r', encoding="utf-8") as f:
#                 r = json.load(f)
#         except Exception as e:
#             print('回傳格式不對，無法使用json.loads讀取')
#             print(response[0]["generated_text"][-1]['content'])
#             # print(r)
#             print(e)
#             with open(f'./data/error.txt', 'a', encoding="utf-8") as f:
#                 f.write(f"Number: {i+1} 回傳格式不對，無法使用json.loads讀取\n")
#                 f.write(str(e) + '\n')
#             retry_count += 1
#             continue
#         correct = True

#     if not correct:
#         print('多次重試不成功')
#         with open(f'./data/error.txt', 'a', encoding="utf-8") as f:
#             f.write(f"Number: {i+1} 多次重試不成功\n")
#         break
    
#     with open(output_file, 'r', encoding="utf-8") as f:
#         exist_output = json.load(f)
#     # 加上新的 batch
#     exist_output.extend(r)
#     # 寫回檔案
#     with open(output_file, 'w', encoding="utf-8") as f:
#         json.dump(exist_output, f)