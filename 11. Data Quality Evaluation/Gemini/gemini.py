import os
from dotenv import load_dotenv
from google import genai
import json

# 讀取 .env 檔案
load_dotenv()

# 取得環境變數
key = os.getenv("GEMINI_API_KEY")

# 初始化 Client
client = genai.Client(api_key=key)

dataset = 'cadec'
dataset = 'share13'
dataset = 'share14'

output_file = f'./output/{dataset}_output.json'
start_index = 0
batch_size = 50

with open(f'./input/{dataset}_llm_auth.json', 'r', encoding="utf-8") as f:
    data = json.load(f)
if dataset == 'cadec':
    with open(f'./prompt/cadec_prompt.txt', 'r', encoding="utf-8") as f:
        prompt = f.read()
else:
    with open(f'./prompt/share_prompt.txt', 'r', encoding="utf-8") as f:
        prompt = f.read()
if not os.path.exists(output_file):
    with open(output_file, 'w', encoding="utf-8") as f:
        json.dump([], f)

total_size = len(data)
print('total number:', total_size)

for i in range(start_index, total_size, batch_size):
    batch_data = data[i : i + batch_size]
    try:
        interaction = client.interactions.create(
            model = "gemini-3.5-flash",
            # model = "gemini-3.1-flash-lite",
            input = f"{prompt}{batch_data}"
        )
        print(interaction.usage)
    except Exception as e:
        print('API Requset失敗')
        print(e)
        continue

    with open(output_file, 'a', encoding="utf-8") as f:
        f.write(interaction.output_text)

# for i in range(start_index, total_size, batch_size):
#     batch_data = data[i : i + batch_size]

#     correct = False
#     retry_count = 0
#     while not correct and retry_count < 5:
#         try:
#             interaction = client.interactions.create(
#                 model = "gemini-3.5-flash",
#                 # model = "gemini-3.1-flash-lite",
#                 input = f"{prompt}{batch_data}"
#             )
#             print(interaction.usage)
#         except Exception as e:
#             print('API Requset失敗')
#             print(e)
#             retry_count += 1
#             continue

#         try:
#             r = json.loads(interaction.output_text)
#             # print(r)
#         except Exception as e:
#             print('回傳格式不對，無法使用json.loads讀取')
#             print(e)
#             with open(f'./error.txt', 'a', encoding="utf-8") as f:
#                 f.write(f"Number: {i+1} - {i+len(batch_data)}\n")
#                 f.write('回傳格式不對，無法使用json.loads讀取\n')
#                 f.write(str(e))
#             retry_count += 1
#             continue
#         if(len(r) != len(batch_data)):
#             print(f'回傳資料筆數不對，應為{len(batch_data)}，實為{len(r)}')
#             with open(f'./error.txt', 'a', encoding="utf-8") as f:
#                 f.write(f"Number: {i+1} - {i+len(batch_data)}\n")
#                 f.write(f'回傳資料筆數不對，應為{len(batch_data)}，實為{len(r)}\n')
#                 f.write(str(r)+'\n')
#             retry_count += 1
#             continue
#         correct = True

#     if not correct:
#         print('多次重試不成功')
#         with open(f'./error.txt', 'a', encoding="utf-8") as f:
#             f.write(f"Number: {i+1} - {i+len(batch_data)}\n")
#             f.write('多次重試不成功')
#         break

#     with open(output_file, 'r', encoding="utf-8") as f:
#         exist_output = json.load(f)
#     # 加上新的 batch
#     exist_output.extend(json.loads(interaction.output_text))
#     # 寫回檔案
#     with open(output_file, 'w', encoding="utf-8") as f:
#         json.dump(exist_output, f)