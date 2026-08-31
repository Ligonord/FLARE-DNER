from openai import OpenAI
from pydantic import BaseModel
import json
import os

# 記得key不要洩漏出去
with open('./data/api_key', 'r', encoding="utf-8") as f:
    api_key = f.read()
client = OpenAI(api_key = api_key)

dataset = 'cadec'
dataset = 'share13'
dataset = 'share14'

# gpt_model = 'gpt-5'
gpt_model = 'gpt-5-mini'

output_file = f'./data/output/{dataset}_output.json'
start_index = 0
batch_size = 50

# if dataset == "cadec":
#     with open(f'./data/input/{dataset}.json', 'r', encoding="utf-8") as f:
#         ner_data = json.load(f)
# else:
#     with open(f'./data/input/{dataset}_fixed.json', 'r', encoding="utf-8") as f:
#         ner_data = json.load(f)

# if dataset == "cadec":
#     with open(f'./data/input/{dataset}_ensemble.json', 'r', encoding="utf-8") as f:
#         ner_data = json.load(f)
# else:
#     with open(f'./data/input/{dataset}_ensemble_fixed.json', 'r', encoding="utf-8") as f:
#         ner_data = json.load(f)

with open(f'./data/input/{dataset}_rag.json', 'r', encoding="utf-8") as f:
    ner_data = json.load(f)

with open(f'./data/prompt_system.txt', 'r', encoding="utf-8") as f:
    prompt_system = f.read()
with open(f'./data/prompt_user.txt', 'r', encoding="utf-8") as f:
    prompt_user = f.read()
if not os.path.exists(output_file):
    with open(output_file, 'w', encoding="utf-8") as f:
        json.dump([], f)

total_size = len(ner_data)
print('total number:', total_size)
for i in range(start_index, total_size, batch_size):
    batch_data = ner_data[i : i + batch_size]
    print(f"目前處理 number {i + 1} 到 {i + len(batch_data)}")

    correct = False
    retry_count = 0
    while not correct and retry_count <= 5:
        try:
            # 送出一次 API Request
            response = client.responses.create(
                model=gpt_model,
                input=[
                    {"role": "system", "content": prompt_system},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": f"{prompt_user}{batch_data}"
                            }
                        ]
                    }
                ],
            )
            print(response.usage)
        except Exception as e:
            print('API Requset失敗')
            print(e)
            retry_count += 1
            continue

        try:
            r = json.loads(response.output_text)
            # print(r)
        except Exception as e:
            print('回傳格式不對，無法使用json.loads讀取')
            print(e)
            with open(f'./data/error.txt', 'a', encoding="utf-8") as f:
                f.write(f"Number: {i+1} - {i+len(batch_data)}\n")
                f.write('回傳格式不對，無法使用json.loads讀取\n')
                f.write(str(e))
            retry_count += 1
            continue
        if(len(r) != len(batch_data)):
            print(f'回傳資料筆數不對，應為{len(batch_data)}，實為{len(r)}')
            with open(f'./data/error.txt', 'a', encoding="utf-8") as f:
                f.write(f"Number: {i+1} - {i+len(batch_data)}\n")
                f.write(f'回傳資料筆數不對，應為{len(batch_data)}，實為{len(r)}\n')
                f.write(str(r)+'\n')
            retry_count += 1
            continue
        correct = True

    if not correct:
        print('多次重試不成功')
        with open(f'./data/error.txt', 'a', encoding="utf-8") as f:
            f.write(f"Number: {i+1} - {i+len(batch_data)}\n")
            f.write('多次重試不成功')
        break
    
    with open(output_file, 'r', encoding="utf-8") as f:
        exist_output = json.load(f)
    # 加上新的 batch
    exist_output.extend(json.loads(response.output_text))
    # 寫回檔案
    with open(output_file, 'w', encoding="utf-8") as f:
        json.dump(exist_output, f)

# file_exist = False
# files = client.files.list()
# # for f in files:
# #     client.files.delete(f.id)
# # for f in files:
# #     print(f.filename)
# for f in files.data:
#     if f.filename == f"{dataset}_ensemble_gpt.pdf":
#         file_id = f.id
#         file_exist = True
#         break
# if not file_exist:
#     new_file = client.files.create(
#         file = open(f"data/{dataset}_ensemble_gpt.pdf", "rb"),
#         purpose = "user_data"
#     )
#     file_id = new_file.id

# start = 1
# output = []
# while start <= total_size:
#     end = min(start + batch_size - 1, total_size)
#     print(f"目前處理 number {start} 到 {end}")

#     # 送出一次 API Request
#     response = client.responses.create(
#         model="gpt-5",
#         input=[
#             {"role": "system", "content": prompt_system},
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "input_file", "file_id": file_id},
#                     {
#                         "type": "input_text",
#                         "text": f"{prompt_user}請給我'number'{start}到{end}的結果"
#                     }
#                 ]
#             }
#         ],
#     )

#     output.extend(json.loads(response.output_text))
#     start = end + 1

# class Entity(BaseModel):
#     text: list[str]
#     index: list[int]

# class NER(BaseModel):
#     number: int
#     # sentence: str
#     # text: list[str]
#     entity_list: list[Entity]

# response = client.responses.parse(
#     model="gpt-5",
#     input=[
#         {
#             "role": "system",
#             "content": prompt_system,
#         },
#         {
#             "role": "user",
#             "content": [
#                 {
#                     "type": "input_file",
#                     "file_id": file_id,
#                 },
#                 {
#                     "type": "input_text",
#                     "text": prompt_user + '請先給我"number"1到10的結果',
#                 },
#             ]
#         }
#     ],
#     text_format=NER,
# )
# # print(response.output_parsed)
# with open('./data/gpt_output.json', 'w', encoding="utf-8") as f:
#     f.write(response.output_parsed.model_dump_json())