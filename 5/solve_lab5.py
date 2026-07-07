import json

nb_path = 'Lab5_PyTorch_assignment.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the cell for Task 1
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if '# Provide your analysis or exploratory code here:' in source:
            cell['source'] = [
                "# Provide your analysis or exploratory code here:\n",
                "\n",
                "print(\"\"\"\n",
                "Analysis:\n",
                "1. linear-regression-data1.csv: YES. This dataset has a linear relationship, so a linear model is appropriate.\n",
                "2. assignment-data.csv: YES. This dataset also has a linear relationship, so the model can learn from it.\n",
                "3. assignment-data2.csv: NO. This dataset contains non-linear data. A simple linear model (without hidden layers and non-linear activation functions) cannot learn the underlying pattern.\n",
                "\"\"\")"
            ]
            break

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

with open('lab5_analysis.txt', 'w') as f:
    for cell in nb['cells']:
        if cell['cell_type'] == 'markdown':
            source = "".join(cell['source'])
            if 'Task' in source:
                f.write(source + '\n\n')
