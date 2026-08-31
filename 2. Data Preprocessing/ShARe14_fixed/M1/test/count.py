import os

folder_path = r"./ann"  # ← 改成你的資料夾路徑
total_lines = 0

for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            line_count = sum(1 for _ in f)
            total_lines += line_count
            print(f"{filename}: {line_count} 行")

print("-" * 30)
print(f"所有 .txt 檔案總行數: {total_lines}")