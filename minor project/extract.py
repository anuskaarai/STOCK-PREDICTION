import json
with open(r'c:\Users\KIIT0001\Downloads\18 hourly.ipynb', encoding='utf-8') as f:
    d = json.load(f)
src = [''.join(c['source']) for c in d['cells'] if c['cell_type'] == 'code']
with open('extracted_ml.py', 'w', encoding='utf-8') as f:
    f.write('\n# NEXT CELL #\n'.join(src))
