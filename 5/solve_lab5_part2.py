import json

nb_path = 'Lab5_PyTorch_assignment.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find the cell for Task 2 justification
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if '# Provide your justification here:' in source:
            cell['source'] = [
                "class ModelV0(nn.Module):\n",
                "    def __init__(self):\n",
                "        super().__init__()\n",
                "        # Two linear layers stacked without activation\n",
                "        self.layer_1 = nn.Linear(in_features=2, out_features=5)\n",
                "        self.layer_2 = nn.Linear(in_features=5, out_features=1)\n",
                "\n",
                "    def forward(self, x):\n",
                "        return self.layer_2(self.layer_1(x))\n",
                "\n",
                "# Provide your justification here:\n",
                "print(\"\"\"\n",
                "Justification:\n",
                "ModelV0 consists of two linear layers stacked together, but it does NOT have any non-linear activation functions between them. \n",
                "Mathematically, stacking multiple linear transformations simply results in another linear transformation. \n",
                "Therefore, ModelV0 is functionally equivalent to a single linear layer and will only be able to draw a straight line (or hyperplane). It will fail to learn the complex, non-linear decision boundary required to classify the interlocking half-moons dataset.\n",
                "\"\"\")"
            ]

# Add conclusion text in the last cell or the one after "conclusion" markdown
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown' and 'Key Takeaways' in "".join(cell['source']):
        # If there's a code cell after it, put the conclusion there
        if i + 1 < len(nb['cells']) and nb['cells'][i+1]['cell_type'] == 'code':
            nb['cells'][i+1]['source'] = [
                "print(\"\"\"\n",
                "Conclusion:\n",
                "- Linear Models: Are only effective for linearly separable data. They fail on complex, non-linear datasets (like the moons dataset).\n",
                "- Hidden Layers: Stacking linear layers without activation functions does not increase the model's capacity to learn non-linear patterns; it collapses into a single linear layer.\n",
                "- Activation Functions: Introducing non-linear activation functions (e.g., ReLU) between hidden layers breaks the linearity. This allows the Neural Network to approximate highly complex, non-linear functions and decision boundaries, making it powerful enough for real-world classification tasks.\n",
                "\"\"\")"
            ]

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
