import transformers
import torch

model = "./Llama-3.1-8B-Instruct"

generator = transformers.pipeline(
    "text-generation",
    model = model,
    device = 'cuda',
    torch_dtype = torch.bfloat16
)

prompt = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

response = generator(
    prompt,
    do_sample = False,
    temperature = 1.0,
    top_p = 1,
    max_new_tokens = 256
)

print(response[0]["generated_text"][-1]['content'])