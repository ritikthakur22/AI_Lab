import json

nb_path = 'Lab5_PyTorch_assignment.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("Cell 10:\n", "".join(nb['cells'][10]['source']))
print("------")
print("Cell 12:\n", "".join(nb['cells'][12]['source']))
