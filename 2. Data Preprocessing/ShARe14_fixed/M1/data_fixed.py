import os

def fix_crossing_sentences(tokens_file, ann_file, output_file):
    annotations = []
    with open(ann_file, 'r', encoding='utf-8') as f:
        for line in f:
            sp = line.strip().split('\t')
            if len(sp) < 4: continue
            indices = [int(i) for i in sp[2].split(',')]
            ann_start, ann_end = min(indices), max(indices)
            # 儲存 (檔名, 開始, 結束)
            annotations.append((sp[0], ann_start, ann_end))

    with open(tokens_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    fixed_lines = []
    for i in range(len(lines)):
        if lines[i].strip() == "":
            if i > 0 and i < len(lines) - 1:
                prev_sp = lines[i-1].strip().split()
                next_sp = lines[i+1].strip().split()
                
                if len(prev_sp) == 4 and len(next_sp) == 4:
                    prev_doc = prev_sp[1]
                    next_doc = next_sp[1]
                    prev_end = int(prev_sp[3])
                    next_start = int(next_sp[2])

                    # --- 修改處：只有檔名相同，才檢查是否為跨句實體 ---
                    if prev_doc == next_doc:
                        is_inside_entity = False
                        for doc_id, a_start, a_end in annotations:
                            # 同時比對檔名與 Offset
                            if doc_id == prev_doc and a_start <= prev_end and a_end >= next_start:
                                is_inside_entity = True
                                break
                        
                        if is_inside_entity:
                            print(f"發現跨句實體於 {prev_doc}！刪除空行")
                            continue 
            
            # 檔名不同，或是非實體內的空行，一律保留
            fixed_lines.append(lines[i])
        else:
            fixed_lines.append(lines[i])

    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(fixed_lines)

if __name__ == "__main__":
    # 使用你的檔案名稱
    fix_crossing_sentences("train.tokens", "train.ann", "train.tokens.fixed")
    fix_crossing_sentences("test.tokens", "test.ann", "test.tokens.fixed")