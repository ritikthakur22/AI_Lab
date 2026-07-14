# The Complete AI Practical Laboratory Book: A Deep Dive into Machine Learning and PyTorch

Welcome to the definitive guide and textbook for the AI Practical Course (Labs 1 through 7). This book is designed to take you from a complete novice setting up a Python environment to a confident programmer capable of building and training Recurrent Neural Networks and understanding advanced Word Embeddings.

We will cover every theoretical concept in exhaustive depth, provide the actual code you need to run, explain exactly what every single line of that code is doing, and provide **visuals, actual data plots, and extensive real-world code examples with their outputs**.

---

## Chapter 0: Prerequisites and Environment Setup

Before writing any Artificial Intelligence code, we must ensure our computer is properly configured. Modern machine learning relies on heavy mathematical computations, and we use specialized libraries to handle this. We don't want these libraries interfering with our system Python, so we use a Virtual Environment (`venv`).

### 0.1. Setting up a Virtual Environment
A virtual environment is an isolated sandbox. When you install packages in a virtual environment, they stay there and do not break other projects on your computer.

Open your terminal and navigate to your project folder:
```bash
cd ~/testing/AI_lab/
```

Create the virtual environment (we will name it `ai_env`):
```bash
python3 -m venv ai_env
```

Activate the virtual environment:
- On Linux/Mac: `source ai_env/bin/activate`
- On Windows: `.\ai_env\Scripts\activate`

You will know it worked because your terminal prompt will change to show `(ai_env)`.

### 0.2. Installing Requirements
Now that we are inside our sandbox, we need to install the tools of the trade:
1. **PyTorch (`torch`)**: The deep learning engine we will use to build neural networks.
2. **Pandas (`pandas`)**: A library for loading tabular data.
3. **Matplotlib (`matplotlib`)**: A plotting library used to draw graphs.
4. **Scikit-Learn (`scikit-learn`)**: For classical machine learning algorithms.

```bash
pip install torch pandas matplotlib jupyter scikit-learn
```

---

## Chapter 1: Machine Learning & Supervised Binary Classification (Lab 1)

**Location**: `~/testing/AI_lab/1/L1-ML-Logistic-Regression.ipynb`

### 1.1. The AI Roadmap & Taxonomy
![AI Roadmap](1/resources/images/comment_2020_AI_roadmap.png)

- **Artificial Intelligence (AI)** is any technique that mimics human intelligence.
- **Machine Learning (ML)** lets computers figure out the rules by looking at data. 
- **Deep Learning (DL)** uses Artificial Neural Networks (inspired by the brain).

### 1.2. The Machine Learning Pipeline
![ML Pipeline](1/resources/images/ml_pipeline.png)

1. **Data Collection**: Gathering raw data.
2. **Feature Extraction**: Selecting important variables.
3. **Model Training**: The algorithm learns patterns.
4. **Evaluation**: Testing the model on unseen data.
5. **Deployment**: Real-world usage.

### 1.3. Supervised Binary Classification
**Real-World Example**: A Spam Filter.
- **Features (Input Data X)**: [Number of exclamation marks, Contains "FREE" (1 or 0), Sender reputation].
- **Target (Output y)**: Spam (1) or Not Spam (0).

### 1.4. Logistic Regression and the Sigmoid Curve in Code
Logistic Regression predicts a *probability* (between 0.0 and 1.0). 
It uses the **Sigmoid Function**: $$ \sigma(z) = \frac{1}{1 + e^{-z}} $$

Let's see this in actual PyTorch code!

```python
import torch

# Let's say our linear equation z = wX + b outputted these raw numbers:
z = torch.tensor([-5.0, 0.0, 2.0, 10.0])

# We pass them through the Sigmoid function
probabilities = torch.sigmoid(z)

print("Raw Outputs:", z)
print("Probabilities:", probabilities)
```
**Console Output:**
```text
Raw Outputs: tensor([-5.,  0.,  2., 10.])
Probabilities: tensor([0.0067, 0.5000, 0.8808, 0.9999])
```
*Notice how `-5.0` became `0.0067` (almost 0%, Not Spam) and `10.0` became `0.9999` (almost 100%, Spam). A raw `0.0` is exactly 50%.*

---

## Chapter 2: Single Variable Linear Regression (Lab 2)

**Location**: `~/testing/AI_lab/2/L2-LinearRegression.ipynb`

### 2.1. Theoretical Foundation
**Example**: Predicting a house's price (Output `y`) based on its square footage (Input `X`).
$$ y = wX + b $$

- **$w$ (Weight)**: How much the price increases for every 1 extra square foot.
- **$b$ (Bias)**: The base price of a house with 0 square feet.

### 2.2. Cost Function (Mean Squared Error)
$$ MSE = \frac{1}{N} \sum_{i=1}^{N} (y_{actual} - y_{predicted})^2 $$

### 2.3. Data Loading & Visualization
![Lab 2 Linear Regression Plot](images/lab2_linear_regression.png)

**Deep Dive into the Plot:**
* **The Blue Dots**: Real-world observations.
* **The Red Line**: The "Line of Best Fit". The model tuned $w$ and $b$ to slice perfectly through the center of the blue dots.

### 2.4. Linear Regression in Code (Scikit-Learn Example)
Here is how you actually train this line in just 4 lines of code using Scikit-Learn:
```python
from sklearn.linear_model import LinearRegression
import numpy as np

# X must be 2D. Let's say these are square footages (in thousands)
X_train = np.array([[1.0], [2.0], [3.0], [4.0]])
# y is the house price (in hundred thousands)
y_train = np.array([3.0, 5.0, 7.0, 9.0])

model = LinearRegression()
model.fit(X_train, y_train)

print(f"Learned Weight (Slope): {model.coef_[0]:.2f}")
print(f"Learned Bias (Intercept): {model.intercept_:.2f}")

# Predict the price of a 5000 sq ft house
prediction = model.predict([[5.0]])
print(f"Predicted price for X=5.0: {prediction[0]:.2f}")
```
**Console Output:**
```text
Learned Weight (Slope): 2.00
Learned Bias (Intercept): 1.00
Predicted price for X=5.0: 11.00
```

---

## Chapter 3: PyTorch Fundamentals (Lab 3)

**Location**: `~/testing/AI_lab/3/Lab3_PyTorch_assignment.ipynb`

### 3.1. Why PyTorch?
PyTorch allows us to process data on a **GPU (Graphics Card)**, performing matrix multiplication exponentially faster than a CPU.

### 3.2. Matrix Multiplication (Dot Product)
In Deep Learning, we don't just multiply single numbers. We multiply massive matrices. This is called a "Dot Product" or Matrix Multiplication (`torch.matmul`).

```python
import torch

# A 2x3 Matrix
tensor_A = torch.tensor([[1, 2, 3],
                         [4, 5, 6]])

# A 3x2 Matrix
tensor_B = torch.tensor([[7, 8],
                         [9, 10],
                         [11, 12]])

# Matrix Multiplication
result = torch.matmul(tensor_A, tensor_B)

print("Shape of A:", tensor_A.shape)
print("Shape of B:", tensor_B.shape)
print("Result Shape:", result.shape)
```
**Console Output:**
```text
Shape of A: torch.Size([2, 3])
Shape of B: torch.Size([3, 2])
Result Shape: torch.Size([2, 2])
```
*Notice how the inner dimensions `(3)` matched, and the resulting matrix took the outer dimensions `(2x2)`. This is the core operation happening millions of times inside Neural Networks!*

---

## Chapter 4: PyTorch Linear Regression from Scratch (Lab 4)

**Location**: `~/testing/AI_lab/4/Lab4_PyTorch_assignment.ipynb`

### 4.1. Defining the Model Structure
```python
from torch import nn

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__() 
        # Initialize Weight and Bias randomly. 
        self.weights = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias
```

### 4.2. Training vs Testing Data
![PyTorch Lab 4 Plot](images/lab4_pytorch_linear.png)

* **Blue Dots (Training Data, 80%)**: The model uses the 5-step loop to adjust its weights based ONLY on the blue dots.
* **Green Dots (Testing Data, 20%)**: The model has *never* seen these points before. 
* **Red Line**: The model perfectly generalized the rule to fit the green dots!

### 4.3. The Mathematics of Backpropagation (Under the Hood)
When we call `loss.backward()` in PyTorch, it automatically performs calculus (the Chain Rule) to figure out how to adjust the weights. Here is exactly what is happening mathematically under the hood for a single training example:

**Given:**
- Input $x = 3$, True target $t = 10$
- Current random parameters: $w = 2, b = 1$
- Learning rate $\eta = 0.1$

**Step 1: Forward Pass (Make a prediction and calculate error)**
$y = wx + b ightarrow y = 2(3) + 1 = 7$
$Loss = (y - t)^2 ightarrow Loss = (7 - 10)^2 = 9$

**Step 2: Backward Pass (Calculate Gradients using Chain Rule)**
We need to know how much the Loss changes if we tweak $y$, $w$, or $b$.
* Derivative of Loss w.r.t $y$:  $\frac{\partial L}{\partial y} = 2(y - t) = 2(7 - 10) = -6$
* Derivative of $y$ w.r.t $w$:  $\frac{\partial y}{\partial w} = x = 3$
* **Chain Rule for $w$**: $\frac{\partial L}{\partial w} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial w} = (-6)(3) = -18$
* **Chain Rule for $b$**: $\frac{\partial L}{\partial b} = \frac{\partial L}{\partial y} \cdot 1 = -6$

**Step 3: Gradient Descent Update (`optimizer.step()`)**
We update our weights by taking a small step ($\eta$) in the opposite direction of the gradient:
* $w_{new} = w - \eta \frac{\partial L}{\partial w} ightarrow w_{new} = 2 - 0.1(-18) = 3.8$
* $b_{new} = b - \eta \frac{\partial L}{\partial b} ightarrow b_{new} = 1 - 0.1(-6) = 1.6$

PyTorch's `loss.backward()` and `optimizer.step()` perform this exact calculus for millions of parameters automatically!

---

## Chapter 5: Deep Neural Networks & Activation Functions (Lab 5)

**Location**: `~/testing/AI_lab/5/Lab5_PyTorch_assignment.ipynb`

### 5.1. The Fatal Flaw of Straight Lines
![Lab 5 Moons Dataset](images/lab5_moons_data.png)

Linear models are useless for highly complex data like the **Moons Dataset** above. It is mathematically impossible to draw a single straight line separating the red from the blue.

### 5.2. The Magic of ReLU
$$ f(x) = max(0, x) $$
![ReLU Activation Function](images/lab5_relu_function.png)

To give the network the ability to "bend", we use the **ReLU** activation function.
- **Negative side (x < 0)**: Outputs 0 (turns off).
- **Positive side (x > 0)**: Outputs the input exactly.

### 5.3. Tracking Tensor Shapes through a Deep Network
Let's build a Deep Network and pass a batch of 5 data points through it, printing exactly how the shape transforms at each step.

```python
class DeepNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU() 

    def forward(self, x):
       print("Input shape:", x.shape)
       x = self.layer_1(x)
       print("After Layer 1:", x.shape)
       x = self.relu(x)
       x = self.layer_2(x)
       print("After Layer 2 (Output):", x.shape)
       return x

model = DeepNeuralNetwork()
fake_data = torch.randn(5, 2)
predictions = model(fake_data)
```
**Console Output:**
```text
Input shape: torch.Size([5, 2])
After Layer 1: torch.Size([5, 10])
After Layer 2 (Output): torch.Size([5, 1])
```
*The model expanded the 2 features into 10 complex hidden features, ran them through ReLU to bend them, and squished them down to 1 final probability prediction.*

---

## Chapter 6: Recurrent Neural Networks (RNN) (Lab 6)

**Location**: `~/testing/AI_lab/6/6.2-RNN.ipynb`

### 6.1. Processing Sequences & Time
Standard Networks have amnesia. If you are predicting the next word in a sentence ("Sitting by the river..." vs "Depositing money at the..."), you need a network with **Memory**.

### 6.2. How an RNN Works
![RNN Architecture Overview](6/images/RNN-as-simple-NN.png)
![RNN Vector Multiplication](6/images/RNN-Vector Multiplication.png)

An RNN introduces a **Hidden State ($h_t$)**. 
When processing step $T_1$, the RNN generates an output AND a hidden state.
When processing step $T_2$, the RNN takes the new input for $T_2$ **PLUS** the hidden state from $T_1$. 

### 6.3. RNN PyTorch Code Example in Action

```python
import torch.nn as nn

# Define the RNN layer
rnn_layer = nn.RNN(input_size=5, hidden_size=3, batch_first=True)

# Fake sequence of 4 days of weather/dish data for 1 cook (Batch size 1)
# Shape: (1 cook, 4 days, 5 features)
sequence_data = torch.randn(1, 4, 5)

# Initialize the very first hidden memory state with zeros!
initial_hidden_state = torch.zeros(1, 1, 3)

# Pass the sequence through the RNN
outputs, final_hidden_state = rnn_layer(sequence_data, initial_hidden_state)

print("Outputs Shape:", outputs.shape)
print("Final Memory State Shape:", final_hidden_state.shape)
```
**Console Output:**
```text
Outputs Shape: torch.Size([1, 4, 3])
Final Memory State Shape: torch.Size([1, 1, 3])
```

- `Outputs Shape: [1, 4, 3]`: The RNN returned a prediction for *every single one of the 4 days*.
- `Final Memory State Shape: [1, 1, 3]`: This is the RNN's brain after experiencing all 4 days. It is exactly 3 numbers long. We can now pass this final memory state into the next day to continue predicting the future!

![Operations in RNN](6/images/operations in RNN.png)

---

## Chapter 7: Word Embeddings & Encoder-Decoder Models (Lab 7)

**Location**: `~/testing/AI_lab/7/`

When working with Natural Language Processing (NLP), we must convert text into numbers before feeding it into our RNN.

### 7.1. Why One-Hot Vectors Fail in NLP
A One-Hot Vector represents a word using a massive array of zeros and a single `1`.
`hello → [1, 0, 0, 0, 0]`
`world → [0, 1, 0, 0, 0]`

**Why is this bad for Neural Networks?**
1. **Extremely Large**: If your vocabulary has 100,000 words, every single word requires an array of 100,000 numbers (99,999 of them being useless zeros). This wastes massive amounts of RAM.
2. **Zero Understanding of Meaning**: A one-hot vector has no semantic understanding. The mathematical distance between `king` and `queen` is exactly the same as the distance between `king` and `banana`. The model cannot know that `car` and `truck` are related because all one-hot vectors are strictly orthogonal.

### 7.2. The Solution: Word Embeddings
An **Embedding Layer** replaces the massive, sparse one-hot vector with a much smaller, dense vector of floating-point numbers.

Instead of `king` being a 100,000-dimension vector of zeros, it becomes a 256-dimension dense vector:
`king  → [0.21, 0.52, 0.18, 0.76 ...]`
`queen → [0.20, 0.49, 0.22, 0.71 ...]`

**Why is this better?**
1. **Compact**: It compresses 100,000 dimensions down to ~256 dimensions.
2. **Learnable Meaning**: During training, the neural network adjusts these floating-point numbers. Words that appear in similar contexts (like "The dog chased the ball" and "The cat chased the mouse") will have their embedding vectors mathematically pushed closer together. The model naturally learns that `dog ≈ cat`!

### 7.3. The Encoder-Decoder Architecture
![Data Preparation](guide/AI_Practical_ACEM/New_Syllabus/Lab7-Encoder-Decoder/images/DataPreparation.drawio.png)

When translating a sentence from English to French, you cannot translate word-for-word because grammar rules differ. We use an **Encoder-Decoder**:
1. **The Encoder RNN**: Reads the English sentence word-by-word and compresses the *entire meaning of the sentence* into a single final Hidden State (often called the Context Vector).
2. **The Decoder RNN**: Takes that final Context Vector as its initial memory and begins generating the French sentence word-by-word until it outputs an `<END>` token.

### 7.4. The Complete NLP Pipeline
1. **Raw Text**: `"I am Groot"`
2. **Word Indices**: `[12, 45, 891]`
3. **Embedding Layer**: Converts the indices into dense floating-point matrices.
4. **Encoder**: Reads the matrices and produces a Context Vector.
5. **Decoder**: Reads the Context Vector and outputs the translated text.

### 7.5. Encoder PyTorch Code Example in Action

Let's see how the first half of this pipeline (The Encoder) is actually implemented in PyTorch code. We will create an Embedding layer and an RNN layer, and pass a fake sentence of 5 words through it.

```python
import torch
import torch.nn as nn

class EncoderRNN(nn.Module):
    def __init__(self, input_vocab_size, hidden_size):
        super(EncoderRNN, self).__init__()
        # Compresses the sparse one-hot words into dense vectors
        self.embedding = nn.Embedding(input_vocab_size, hidden_size)
        # The Recurrent Neural Network
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)

    def forward(self, input_seq):
        embedded = self.embedding(input_seq)
        output, final_hidden = self.rnn(embedded)
        return output, final_hidden

# Example: Vocabulary of 1000 words, hidden state size of 256
encoder = EncoderRNN(input_vocab_size=1000, hidden_size=256)

# Fake sequence of 5 words (e.g., "I am learning PyTorch now") for 1 user (Batch size 1)
# Each number is the index of the word in our vocabulary
fake_input = torch.tensor([[12, 45, 891, 23, 1]])

# Pass the sequence through the Encoder
outputs, context_vector = encoder(fake_input)

print("Input Sequence Shape:", fake_input.shape)
print("Encoder Outputs Shape:", outputs.shape)
print("Context Vector (Final Hidden State) Shape:", context_vector.shape)
```
**Console Output:**
```text
Input Sequence Shape: torch.Size([1, 5])
Encoder Outputs Shape: torch.Size([1, 5, 256])
Context Vector (Final Hidden State) Shape: torch.Size([1, 1, 256])
```
*Notice how the entire sequence of 5 words has been read, and the RNN's brain compressed all of that meaning into a single `[1, 1, 256]` Context Vector. This exact vector is what gets passed to the Decoder to begin translation!*

---

## Chapter 8: Overcoming the Bottleneck with Attention Mechanisms (Upcoming Lab 8)

**Location**: `~/testing/AI_lab/guide/AI_Practical_ACEM/New_Syllabus/Lab8-Attention/`

While the Encoder-Decoder model from Lab 7 is powerful, it has a fatal flaw: **The Context Vector Bottleneck**. The Encoder is forced to compress the *entire* meaning of a long sentence into a single, fixed-size array. If a sentence has 50 words, the model often "forgets" the beginning of the sentence by the time it reaches the end!

In Lab 8, we will explore **Bahdanau Additive Attention**. Instead of relying on one single final state, an Attention mechanism allows the Decoder to "look back" at *all* the hidden states of the Encoder at every single time step. It dynamically calculates an "attention weight" for each word, allowing it to focus only on the most relevant source words while generating the current translation. This revolutionary idea completely eliminates the bottleneck and paved the way for modern Transformers!

---

## Final Thoughts
By working through these 8 Labs, you have crossed the threshold from standard programming into the realm of Deep Learning. 

You started by learning what AI actually is. You moved on to basic mathematical curve fitting with Linear Regression. You mastered the syntax of Tensors in PyTorch. You built a model from scratch and manually executed a backpropagation training loop, understanding the raw calculus behind it. You discovered how Activation Functions allow models to perceive complex realities. Finally, you stepped into the fourth dimension of time by utilizing Recurrent Neural Networks and Encoder-Decoders to process sequential memory and natural language.
