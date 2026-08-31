from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 1. 設定本地 Embedding 模型
Settings.embed_model = HuggingFaceEmbedding(
    model_name='dmis-lab/biobert-base-cased-v1.2',
    device='cuda'
)

# 2. 告訴 LlamaIndex 我們現在不需要 LLM (避免它去抓 OpenAI)
Settings.llm = None 

# 3. 載入索引
storage_path = './data/storage'
storage_context = StorageContext.from_defaults(persist_dir=storage_path)
index = load_index_from_storage(storage_context)

# 4. 改用 Retriever 而不是 Query Engine
# query_engine 會試圖總結答案(需要LLM)，retriever 只負責找資料(不需要LLM)
retriever = index.as_retriever(similarity_top_k=2)

# 5. 測試檢索
results = retriever.retrieve("Which terms are related to stomach discomfort?")

print("-" * 30)
for res in results:
    print(f"找到術語: {res.text} (相似度分數: {res.score:.4f})")