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

add_md("# **LAB 4: PyTorch Linear Regression**")
add_md("## **OBJECTIVE**\nTo understand the basics of building, training, and evaluating a Linear Regression model using PyTorch. This includes data preparation, model creation using `nn.Module`, defining loss functions and optimizers, and implementing a training loop.")
add_md("## **THEORY**\n\n### Linear Regression in PyTorch\nLinear regression is a fundamental algorithm in machine learning that models the relationship between a dependent variable and one or more independent variables. In PyTorch, we can implement this by creating a custom model class that inherits from `nn.Module`.\n\n### Training a Model\nThe training process involves:\n1. **Forward Pass:** Passing the input data through the model to get predictions.\n2. **Calculating Loss:** Measuring how far off the predictions are from the actual true values using a loss function (e.g., L1Loss or MSELoss).\n3. **Zeroing Gradients:** Clearing old gradients from the previous step.\n4. **Backward Pass (Backpropagation):** Computing the gradient of the loss with respect to the model parameters.\n5. **Optimizer Step:** Updating the model parameters using an optimizer (e.g., SGD).")

add_code("import torch\nimport matplotlib.pyplot as plt\nfrom torch import nn\nimport pandas as pd\n\nprint(\"PyTorch Version:\", torch.__version__)")
add_code("# Setup device agnostic code\ndevice = \"cuda\" if torch.cuda.is_available() else \"cpu\"\nprint(f\"Using device: {device}\")")
add_code("# Load the assignment data\ndf = pd.read_csv('assignment-data2.csv')\ndf.head()")
add_code("# Convert to torch tensors\nX = torch.tensor(df['x'].values, dtype=torch.float32).unsqueeze(1)\ny = torch.tensor(df['y'].values, dtype=torch.float32).unsqueeze(1)\n\n# Split data\ntrain_split = int(0.8 * len(X))\nX_train, y_train = X[:train_split], y[:train_split]\nX_test, y_test = X[train_split:], y[train_split:]\n\nprint(f\"Training data: {len(X_train)} samples\")\nprint(f\"Testing data: {len(X_test)} samples\")")
add_code("def plot_predictions(train_data=X_train, \n                     train_labels=y_train, \n                     test_data=X_test, \n                     test_labels=y_test, \n                     predictions=None):\n    plt.figure(figsize=(10, 7))\n    plt.scatter(train_data, train_labels, c=\"b\", s=4, label=\"Training data\")\n    plt.scatter(test_data, test_labels, c=\"g\", s=4, label=\"Testing data\")\n    if predictions is not None:\n        plt.scatter(test_data, predictions, c=\"r\", s=4, label=\"Predictions\")\n    plt.legend(prop={\"size\": 14})")
add_code("plot_predictions()")
add_code("class LinearRegressionModel(nn.Module):\n    def __init__(self):\n        super().__init__() \n        self.weights = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)\n        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)\n\n    def forward(self, x: torch.Tensor) -> torch.Tensor:\n        return self.weights * x + self.bias")
add_code("torch.manual_seed(42)\nmodel_0 = LinearRegressionModel()\nlist(model_0.parameters())")
add_code("loss_fn = nn.L1Loss()\noptimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)")
add_code("torch.manual_seed(42)\nepochs = 200\n\ntrain_loss_values = []\ntest_loss_values = []\nepoch_count = []\n\nfor epoch in range(epochs):\n    model_0.train()\n    y_pred = model_0(X_train)\n    loss = loss_fn(y_pred, y_train)\n    optimizer.zero_grad()\n    loss.backward()\n    optimizer.step()\n\n    model_0.eval()\n    with torch.inference_mode():\n        test_pred = model_0(X_test)\n        test_loss = loss_fn(test_pred, y_test.type(torch.float))\n\n        if epoch % 10 == 0:\n            epoch_count.append(epoch)\n            train_loss_values.append(loss.detach().numpy())\n            test_loss_values.append(test_loss.detach().numpy())\n            print(f\"Epoch: {epoch} | MAE Train Loss: {loss:.4f} | MAE Test Loss: {test_loss:.4f}\")")
add_code("plt.plot(epoch_count, train_loss_values, label=\"Train loss\")\nplt.plot(epoch_count, test_loss_values, label=\"Test loss\")\nplt.title(\"Training and test loss curves\")\nplt.ylabel(\"Loss\")\nplt.xlabel(\"Epochs\")\nplt.legend();")
add_code("model_0.eval()\nwith torch.inference_mode():\n    y_preds = model_0(X_test)\nplot_predictions(predictions=y_preds)")

add_md("## **DISCUSSION**\nIn this laboratory exercise, we implemented a Linear Regression model using PyTorch and trained it on `assignment-data2.csv`. We observed the standard workflow for model training: making forward passes, calculating the loss, performing backpropagation, and taking optimization steps. \n\nOne important observation is that the dataset represents a non-linear (parabolic) relationship. Since we used a simple linear model (`y = mx + b`), the model attempts to find the best-fit line, which cannot perfectly capture the underlying quadratic distribution. This highlights a crucial concept in machine learning: a linear model may underfit non-linear data, and more complex models (such as polynomial regression or neural networks with hidden layers) are required for such datasets.")
add_md("## **CONCLUSION**\nThrough this lab, we successfully achieved the objective of building and training a PyTorch model from scratch. We learned how to define a custom model utilizing `nn.Module`, how to incorporate a loss function, and how to use gradient descent via an optimizer to train the parameters over multiple epochs. Although the linear model was not the perfect fit for this specific non-linear dataset, the underlying PyTorch training loop and concepts remain the same and serve as a strong foundation for building more advanced deep learning models.")

with open('/home/crdy/testing/AI_lab/4/Lab4_PyTorch_assignment.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)
