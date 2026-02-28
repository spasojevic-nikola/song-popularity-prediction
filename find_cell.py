import json

nb = json.load(open('prezentacija.ipynb', 'r', encoding='utf-8'))

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        source_text = ''.join(cell['source'])
        if 'Otkriveni šabloni' in source_text:
            print(f"Ćelija {i}: Pronađena")
            print(f"ID: {cell.get('id', 'N/A')}")
            print(f"Počinje sa: {source_text[:100]}")
