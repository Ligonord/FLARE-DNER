import pandas as pd

# 根據官方說明定義欄位名稱 (Column Names)
column_names_meddra = [
    'UMLS_id', 
    'MedDRA_hierarchy', 
    'MedDRA_id', 
    'side_effect_name'
]
column_names_sider = [
    "stitch_id_flat",
    "stitch_id_stereo",
    "umls_id_label",
    "meddra_type",
    "umls_id_meddra",
    "side_effect_name"
]

# 讀取檔案
df_meddra = pd.read_csv('./data/meddra_sider/meddra.tsv', sep='\t', names=column_names_meddra)
df_sider = pd.read_csv('./data/meddra_sider/meddra_all_se.tsv', sep='\t', names=column_names_sider)

# --- 過濾出 PT 層級的副作用 ---
pt_df_meddra = df_meddra[df_meddra['MedDRA_hierarchy'] == 'PT'].copy()
pt_df_sider = df_sider[df_sider['meddra_type'] == 'PT'].copy()

# 移除重複的副作用名稱
unique_list_meddra = df_meddra[['side_effect_name']].drop_duplicates()
unique_list_sider = df_sider[['side_effect_name']].drop_duplicates()
unique_pt_list_meddra = pt_df_meddra[['side_effect_name']].drop_duplicates()
unique_pt_list_sider = pt_df_sider[['side_effect_name']].drop_duplicates()
unique_list = pd.concat([unique_list_meddra, unique_list_sider]).drop_duplicates()
unique_pt_list = pd.concat([unique_pt_list_meddra, unique_pt_list_sider]).drop_duplicates()

print(f"Meddra 總共有 {len(unique_list_meddra)} 個不重複的 PT + LLT 副作用。")
print(f"Meddra 總共有 {len(unique_pt_list_meddra)} 個不重複的 PT 副作用。")
print(f"SIDER 總共有 {len(unique_list_sider)} 個不重複的 PT + LLT 副作用。")
print(f"SIDER 總共有 {len(unique_pt_list_sider)} 個不重複的 PT 副作用。")
print(f"整合後，總共有 {len(unique_list)} 個不重複的 PT + LLT 副作用。")
print(f"整合後，總共有 {len(unique_pt_list)} 個不重複的 PT 副作用。")

# unique_list_meddra.to_csv('./data/doc/meddra_unique_adrs.csv', index=False, encoding='utf-8-sig')
# unique_pt_list_meddra.to_csv('./data/doc/meddra_unique_pt_adrs.csv', index=False, encoding='utf-8-sig')
# unique_list_sider.to_csv('./data/doc/sider_unique_adrs.csv', index=False, encoding='utf-8-sig')
# unique_pt_list_sider.to_csv('./data/doc/sider_unique_pt_adrs.csv', index=False, encoding='utf-8-sig')
unique_list.to_csv('./data/doc/combined_unique_adrs.csv', index=False, encoding='utf-8-sig')
unique_pt_list.to_csv('./data/doc/combined_unique_pt_adrs.csv', index=False, encoding='utf-8-sig')