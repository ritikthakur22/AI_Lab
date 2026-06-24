# Lab 5: Neural Networks

## Task 1

```python
# Define model
class LinearRegressionNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(
            in_features=1,
            out_features=1,
            bias=True
        )

    def forward(self, x):
        return self.linear(x)
```

* Using the model definition above, determine whether the model is capable of learning from each dataset provided in Lab 4:

  * [linear-regression-data1.csv](../Lab4-Regression-using-PyTorch/linear-regression-data1.csv) 
  * [assignment-data.csv](../Lab4-Regression-using-PyTorch/assignment-data.csv)
  * [assignment-data2.csv](../Lab4-Regression-using-PyTorch/assignment-data2.csv)

**Note:** Only determine whether the model can be trained on the datasets and whether its architecture is appropriate. A detailed performance evaluation is not required.

## Task 2

1. Determine whether the `LinearRegressionNN` model is suitable for the dataset [binary_classification_moons.csv](../../Old_Syllabus/Lab3-NN/binary_classification_moons.csv).

2. Determine whether the following model `ModelV0` is suitable for the same dataset. Justify your answer by explaining why the model is or is not appropriate.

```python
class ModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=5)
        self.layer_2 = nn.Linear(in_features=5, out_features=1)

    def forward(self, x):
        return self.layer_2(self.layer_1(x))
```

3. Modify the model by increasing the number of layers and/or hidden units. An example is provided in [Lab-5.1-NN.ipynb](./Lab-5.1-NN.ipynb). Evaluate the suitability of the resulting model architecture(s) for the [binary_classification_moons.csv](../../Old_Syllabus/Lab3-NN/binary_classification_moons.csv) dataset and justify your conclusions.

## Task 3
class ModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU() # <- add in ReLU activation function
        # Can also put sigmoid in the model 
        # This would mean you don't need to use it on the predictions
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
      # Intersperse the ReLU activation function between layers
       return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))

Task 3 will be introduced during the laboratory session.
