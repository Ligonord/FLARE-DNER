from fpdf import FPDF
import json

dataset = 'cadec'
# dataset = 'share13'
# dataset = 'share14'
with open(f"data/{dataset}_ensemble_gpt.json", "r", encoding="utf-8") as f:
    data = f.read()

pdf = FPDF()
pdf.add_page()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.set_font("Arial", size=12)

# 將 JSON 每行文字寫入 PDF
for line in data.splitlines():
    pdf.multi_cell(0, 5, line)

pdf.output(f"data/{dataset}_ensemble_gpt.pdf")