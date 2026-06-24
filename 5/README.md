# Lab 5: Neural Networks

On a high level, the inspiration for neural networks comes from the biological neurons in the human brain, which communicate through electrical signals.

<!-- ![NN with activation](https://lamarr-institute.org/wp-content/uploads/deepLearn_2_EN.png) -->

A neural network is comprised of:

- Input layer: holds the raw features  
`(X1, X2, X3,..)`.
- Hidden layers: consist of artificial neurons (or nodes) that transform inputs into new representations. Mathematically, hidden layers are expressed as the input features, multiplied by their associated weights and added bias to pass from one layer to the next layer, eventually arriving at the final output layer. This is where the linear transformation between input and output happens. 
- Output layer: After performing the linear transformation in the hidden layer, a nonlinear activation function (tanh, sigmoid, ReLU ) is added to produce the final prediction (such as a number for regression, or a probability distribution for classification).  

## Neural Network Training

Like other machine learning algorithms, neural networks must be trained on data before they can make accurate predictions.

For a single neuron, the computation is:

$$
z = \sum_{i=1}^{n} w_i x_i + b
$$

$$
a = \sigma(z)
$$

Where:

- $x_i$ = input feature
- $w_i$ = weight
- $b$ = bias
- $z$ = weighted sum (linear transformation)
- $\sigma$ = activation function (nonlinear transformation)
- $a$ = output

The neuron first calculates a weighted sum of the inputs and bias. This value is then passed through an activation function, which introduces nonlinearity and helps the network learn complex patterns.

By combining many neurons across multiple layers, the network transforms input features $X$ into an output $Y$, allowing it to perform tasks such as classification or prediction.

## How Neural Networks Learn

The strength of a neural network comes from its ability to learn the correct weights and biases from data.

During training, the network compares its prediction $\hat{Y}$ with the true label $Y$. The difference between them is measured using a **loss function**, which indicates how well the model is performing.

To reduce this error, neural networks use **backpropagation** and an optimization algorithm such as **gradient descent**.

### Training Process

1. **Forward Pass**
   - Input data moves through the network.
   - Each neuron computes weighted sums and applies activation functions.
   - The network produces a prediction.

2. **Loss Calculation**
   - The prediction is compared with the true value.
   - A loss function calculates the prediction error.

3. **Backpropagation**
   - The error is propagated backward through the network.
   - Using the chain rule from calculus, the network determines how much each weight and bias contributed to the error.

4. **Parameter Update**
   - Weights and biases are adjusted slightly to reduce the error.
   - This is typically done using gradient descent or one of its variants.

By repeating these steps over many training examples, the network gradually improves its predictions and learns useful patterns from the data.