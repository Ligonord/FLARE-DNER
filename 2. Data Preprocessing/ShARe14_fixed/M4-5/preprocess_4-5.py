import os
import json

def convert_format(data):
    # �NInput�榡�ഫ��Output�榡
    converted_data = []
    
    for item in data:
        sentence = item['word_list']
        ner_annotations = []
        
        for entity in item.get('entity_list', []):
            # �ഫ tok_span �d�򬰯��ަC��
            index = []
            
            for i in range(0, len(entity['tok_span']), 2):
                start = entity['tok_span'][i]
                end = entity['tok_span'][i+1]
                index.extend(list(range(start, end)))

            ner_annotations.append({
                'index': index,
                'type': entity['type']
            })
        
        converted_data.append({
            'sentence': sentence,
            'ner': ner_annotations
        })
    
    return converted_data

def main():
    # �]�w input �M output ��Ƨ������| (Input�ɮ׬�M3��Output)
    input_folder = 'Input'
    output_folder = 'Output'
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # �B�z�C���ɮ�
    for filename in ['train_data.json', 'valid_data.json', 'test_data.json']:
        input_file_path = os.path.join(input_folder, filename)
        output_file_path = os.path.join(output_folder, filename)
        
        # Ū��Input�� JSON �ɮ�
        with open(input_file_path, 'r', encoding='utf-8') as infile:
            data = json.load(infile)
        
        # �ഫ�榡
        converted_data = convert_format(data)
        
        # �x�sOutput�� JSON �ɮ�
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            json.dump(converted_data, outfile, ensure_ascii=False, indent=4)
        
        print(f'Converted {filename} and saved to {output_file_path}')

if __name__ == '__main__':
    main()
