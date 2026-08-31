import pandas as pd
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Document, VectorStoreIndex, Settings

Settings.embed_model = HuggingFaceEmbedding(
    model_name = 'dmis-lab/biobert-base-cased-v1.2',
    device = 'cuda'
)

meddra_hierarchy = 'pt'
# meddra_hierarchy = 'pt_llt'
# doc_path = './data/doc/meddra_unique_adrs.csv'
# doc_path = './data/doc/meddra_unique_pt_adrs.csv'
# doc_path = './data/doc/sider_unique_adrs.csv'
# doc_path = './data/doc/sider_unique_pt_adrs.csv'
if meddra_hierarchy == 'pt':
    doc_path = './data/doc/combined_unique_pt_adrs.csv'
elif meddra_hierarchy == 'pt_llt':
    doc_path = './data/doc/combined_unique_adrs.csv'
storage_path = f'./data/{meddra_hierarchy}_storage'

df = pd.read_csv(doc_path, encoding='utf-8-sig')

documents = [
    Document(text=str(name).strip()) 
    for name in df['side_effect_name'].tolist() 
    if pd.notna(name)
]
index = VectorStoreIndex.from_documents(documents, show_progress = True)
index.storage_context.persist(persist_dir=storage_path)

print(f"成功嵌入了 {len(documents)} 筆資料")

# embed_model = HuggingFaceEmbedding(
#     model_name = 'dmis-lab/biobert-base-cased-v1.2',
#     device = 'cuda'
# )
# embeddings = embed_model.get_text_embedding(
#     "Hello World"
# )
# print(embeddings[:5])
# print(len(embeddings))