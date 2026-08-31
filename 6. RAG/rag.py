import os
import json
import torch
from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext, load_index_from_storage, Settings

Settings.embed_model = HuggingFaceEmbedding(
    model_name = 'dmis-lab/biobert-base-cased-v1.2',
    device = 'cuda'
)

print("=== CUDA 初始化檢查 ===")
print(f"CUDA 可用性: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU 型號: {torch.cuda.get_device_name(0)}")
    # 此時模型可能還沒真正搬進去，顯示通常很小
    print(f"初始顯存佔用: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

meddra_hierarchy = 'pt'
meddra_hierarchy = 'pt_llt'
storage_path = f'./data/{meddra_hierarchy}_storage'
Settings.llm = None
storage_context = StorageContext.from_defaults(persist_dir=storage_path)
index = load_index_from_storage(storage_context)
retriever = index.as_retriever(similarity_top_k=5)

for dataset in ['cadec', 'share13', 'share14']:
    output_file = f'./data/output/{dataset}_rag.json'
    with open(f'./data/input/{dataset}_rag_pre.json', 'r', encoding="utf-8") as f:
        ner_data = json.load(f)
    if not os.path.exists(output_file):
        with open(output_file, 'w', encoding="utf-8") as f:
            json.dump([], f)

    threshold = 0.9
    for a, d in enumerate(ner_data):
        # if a < 5621 and dataset == 'share13':
        #     continue
        print(f'{dataset} 第 {a+1} 筆資料')
        rag = []

        '''sentence rag'''
        # nodes = retriever.retrieve(d['sentence'])
        # for node in nodes:
        #     if node.score >= threshold and node.node.get_content() not in rag:
        #         rag.append(node.node.get_content())
        
        '''entity rag'''
        entity = []
        for ent in d[f'entity_list']:
        # for ent in d[f'entity_list_rag']:
            e = " ".join(ent['text'])
            if e not in entity:
                entity.append(e)
        entity = [e for e in entity if e.strip()]
        for ent in entity:
            nodes = retriever.retrieve(ent)
            for node in nodes:
                if node.score >= threshold and node.node.get_content() not in rag:
                    rag.append(node.node.get_content())

        '''token rag'''
        # for t in d['text']:
        #     nodes = retriever.retrieve(t)
        #     for node in nodes:
        #         if node.score >= threshold and node.node.get_content() not in rag:
        #             rag.append(node.node.get_content())

        d.update({"rag": rag})
        del d['entity_list_rag']
        with open(output_file, 'r', encoding="utf-8") as f:
            exist_output = json.load(f)
        # 加上新的 batch
        exist_output.append(d)
        # 寫回檔案
        with open(output_file, 'w', encoding="utf-8") as f:
            json.dump(exist_output, f)

# f = open(f'./data/test_out.txt', 'w+', encoding="utf-8")
# t='urine full of blood'
# f.write(t+'\n')
# nodes = retriever.retrieve(t)
# f.write("\n--- 來源驗證 ---\n")
# for i, node_with_score in enumerate(nodes):
#     f.write(f"來源 {i+1} (相似度分數: {node_with_score.score:.4f}):\n")
#     f.write(f"內容節錄: {node_with_score.node.get_content()}\n") 
#     f.write("-" * 30+'\n')
# f.close()

# with open('./data/api_key', 'r', encoding="utf-8") as f:
#     api_key = f.read()
# llm = OpenAI(
#     model="gpt-5-mini",
#     api_key=api_key
# )
# messages = [
#     ChatMessage(
#         role="system", content="You are a pirate with a colorful personality"
#     ),
#     ChatMessage(role="user", content="What is your name"),
# ]
# resp = llm.chat(messages)
# print(resp)