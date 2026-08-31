from llama_index.llms.openai import OpenAI
from llama_index.core.llms import ChatMessage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext, load_index_from_storage, Settings

with open('./data/api_key', 'r', encoding="utf-8") as f:
    api_key = f.read()

Settings.embed_model = HuggingFaceEmbedding(
    model_name = 'dmis-lab/biobert-base-cased-v1.2',
    device = 'cuda'
)
Settings.llm = OpenAI(
    model="gpt-5-mini",
    api_key=api_key
)
storage_path = './data/storage'

storage_context = StorageContext.from_defaults(persist_dir=storage_path)
index = load_index_from_storage(storage_context)
query_engine = index.as_query_engine(similarity_top_k=5)

# --- 實作程式碼 ---

# 1. 定義你的內容與指令
user_content = "to add to this i couldnt pass urine , when i eventualy did it was full of blood ."
instruction = f"This is a NER task, please list all ADRs (Adverse Drug Reactions) related to the following patient description: '{user_content}'. Don't change any words from the description, because NER task need the entity to be the same in sentence."

# 2. 手動執行檢索 (只用 user_content 進行 Embedding 匹配)
# 這樣 BioBERT 只會處理病人描述的部分
retriever = index.as_retriever(similarity_top_k=3)
nodes = retriever.retrieve(user_content)

# 3. 手動構建上下文供 LLM 參考
context_str = "\n".join([n.node.get_content() for n in nodes])

# 4. 讓 LLM 根據檢索到的術語回答問題
# 我們把指令跟檢索結果餵給 LLM
prompt = f"""
Instruction: {instruction}

Based on the official medical terms below, identify which ones best describe the symptoms:
{context_str}

Final Answer:
"""
response = Settings.llm.complete(prompt)

print(f"檢索到的相關術語：\n{context_str}")
print(f"\nAI 分析結果：\n{response}")