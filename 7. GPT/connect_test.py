from openai import OpenAI
import json

# 記得key不要洩漏出去
with open('./data/api_key', 'r', encoding="utf-8") as f:
    api_key = f.read()
client = OpenAI(api_key = api_key)

response = client.responses.create(
    model="gpt-5-mini",
    input=[
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": f"連線測試'收到訊息請回答'有'"
                }
            ]
        }
    ],
)
print(response.output_text)