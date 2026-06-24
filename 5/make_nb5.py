import json

notebook = {
 "cells": [],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

def add_md(text):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [text]
    })

def add_code(text):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [text]
    })

# Structure
add_md("# **LAB 5: PyTorch Neural Networks and Activation Functions**")
add_md("## **OBJECTIVE**\nTo understand the architecture of Neural Networks, evaluate model suitability for linear and non-linear datasets, and learn how non-linear activation functions (like ReLU) enable networks to solve complex classification tasks.")
add_md("## **THEORY**\n\n### Linear vs. Non-linear Data\nA linear model can only separate data using a straight line (or hyperplane). If the underlying data distribution is more complex (like concentric circles or interlocking moons), a linear model will fail to capture the patterns.\n\n### Hidden Layers and Activation Functions\nTo model non-linear relationships, Neural Networks introduce **hidden layers** combined with **activation functions** (such as ReLU or Sigmoid). Without activation functions, no matter how many linear layers you stack, the entire network simply behaves like a single linear transformation. Activation functions break this linearity, allowing the network to approximate highly complex functions.")

add_md("## **Task 1: Evaluating Linear Models**\n\nGiven the following basic linear model definition:")
add_code("import torch\nimport torch.nn as nn\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport utils\n\nclass LinearRegressionNN(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.linear = nn.Linear(\n            in_features=1,\n            out_features=1,\n            bias=True\n        )\n\n    def forward(self, x):\n        return self.linear(x)")
add_md("**Question:** Using the model definition above, determine whether the model is capable of learning from each dataset provided in Lab 4:\n* `linear-regression-data1.csv`\n* `assignment-data.csv`\n* `assignment-data2.csv`\n\n*Note: Only determine whether the model can be trained on the datasets and whether its architecture is appropriate. A detailed performance evaluation is not required.*")
add_code("# Provide your analysis or exploratory code here:")

add_md("## **Task 2: Classifying Non-Linear Data**\n\nNow, consider the `binary_classification_moons.csv` dataset, which contains non-linear data shaped like two interlocking half-moons.")
add_code("# Load and inspect the dataset\ndf = pd.read_csv('binary_classification_moons.csv')\nprint(df.head())\n\n# Visualize\nplt.scatter(df['X1'], df['X2'], c=df['y'], cmap=plt.cm.RdYlBu)\nplt.title('Moons Dataset')\nplt.show()")
add_md("**Question 1:** Determine whether the `LinearRegressionNN` model is suitable for this dataset.\n\n**Question 2:** Determine whether the following `ModelV0` is suitable for the same dataset. Justify your answer by explaining why the model is or is not appropriate.")
add_code("class ModelV0(nn.Module):\n    def __init__(self):\n        super().__init__()\n        # Two linear layers stacked without activation\n        self.layer_1 = nn.Linear(in_features=2, out_features=5)\n        self.layer_2 = nn.Linear(in_features=5, out_features=1)\n\n    def forward(self, x):\n        return self.layer_2(self.layer_1(x))\n\n# Provide your justification here:")

add_md("## **Task 3: Adding Non-Linearity**\n\nModify the model by increasing the number of layers/hidden units and adding the ReLU activation function, as demonstrated in `ModelV2`.")
add_code("class ModelV2(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.layer_1 = nn.Linear(in_features=2, out_features=10)\n        self.layer_2 = nn.Linear(in_features=10, out_features=10)\n        self.layer_3 = nn.Linear(in_features=10, out_features=1)\n        self.relu = nn.ReLU() # <- add in ReLU activation function\n\n    def forward(self, x):\n       # Intersperse the ReLU activation function between layers\n       return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))\n\n# Task 3 will be discussed further during the laboratory session.")

add_md("## **DISCUSSION**\nUse this section to discuss the importance of activation functions. How did `ModelV0` perform theoretically compared to `ModelV2`? Why is stacking linear layers without an activation function insufficient for non-linear data like the Moons dataset?")
add_code("# Write your discussion here")

add_md("## **CONCLUSION**\nSummarize the key takeaways from this lab regarding the architecture of Neural Networks and the role of non-linear activation functions in complex classification tasks.")

with open('/home/crdy/testing/AI_lab/5/Lab5_PyTorch_assignment.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)
