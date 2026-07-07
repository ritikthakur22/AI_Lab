import json

nb_path = 'Lab5_PyTorch_assignment.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    source = "".join(cell['source'])
    if 'Task 2:' in source or 'Task 3:' in source or 'conclusion' in source or 'Provide your analysis' in source or 'Define your non-linear model here' in source or cell['cell_type'] == 'code':
        print(f"Cell {i} ({cell['cell_type']}):\n{source[:200]}...\n")
