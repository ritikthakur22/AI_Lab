# The Complete AI Practical Laboratory Book: A Deep Dive into Machine Learning and PyTorch

Welcome to the definitive guide and textbook for the AI Practical Course (Labs 1 through 6). This book is designed to take you from a complete novice setting up a Python environment to a confident programmer capable of building and training Recurrent Neural Networks in PyTorch. 

We will cover every theoretical concept in exhaustive depth, provide the actual code you need to run, and explain exactly what every single line of that code is doing.

---

## Chapter 0: Prerequisites and Environment Setup

Before writing any Artificial Intelligence code, we must ensure our computer is properly configured. Modern machine learning relies on heavy mathematical computations, and we use specialized libraries to handle this. We don't want these libraries interfering with our system Python, so we use a Virtual Environment (`venv`).

### 0.1. Setting up a Virtual Environment
A virtual environment is an isolated sandbox. When you install packages in a virtual environment, they stay there and do not break other projects on your computer.

Open your terminal and navigate to your project folder:
```bash
cd /home/crdy/testing/AI_lab/
```

Create the virtual environment (we will name it `ai_env`):
```bash
python3 -m venv ai_env
```
This command tells Python 3 to use the `venv` module to create a new folder called `ai_env` containing a fresh Python installation.

Activate the virtual environment:
- On Linux/Mac: `source ai_env/bin/activate`
- On Windows: `.\ai_env\Scripts\activate`

You will know it worked because your terminal prompt will change to show `(ai_env)`.

### 0.2. Installing Requirements
Now that we are inside our sandbox, we need to install the tools of the trade. For this course, you need the following primary libraries:
1. **PyTorch (`torch`)**: The deep learning engine we will use to build neural networks.
2. **Pandas (`pandas`)**: A library for loading, manipulating, and analyzing tabular data (like CSV files).
3. **Matplotlib (`matplotlib`)**: A plotting library used to draw graphs and visualize our data.
4. **Jupyter (`jupyter`)**: To run interactive `.ipynb` notebook files.

Install them using pip:
```bash
pip install torch pandas matplotlib jupyter
```

You can save your exact environment setup into a file so others can replicate it:
```bash
pip freeze > requirements.txt
```

Now you are ready to begin Lab 1.

---

## Chapter 1: Machine Learning & Supervised Binary Classification (Lab 1)

**Location**: `/home/crdy/testing/AI_lab/1/L1-ML-Logistic-Regression.ipynb`

### 1.1. The Big Picture: AI vs. ML vs. DL
Before writing algorithms, the professor wants us to understand the hierarchy of the field:
- **Artificial Intelligence (AI)** is the broadest concept. It refers to any technique that enables computers to mimic human intelligence. Even a simple `if/else` statement programmed to play Tic-Tac-Toe is technically AI.
- **Machine Learning (ML)** is a subset of AI. Instead of explicitly programming the rules, we give the computer data and the answers, and let the computer figure out the rules. The system *learns* patterns from the data.
- **Deep Learning (DL)** is a subset of ML based on Artificial Neural Networks, which are inspired by the human brain. DL is incredibly powerful for complex tasks like image recognition and natural language processing.

### 1.2. Supervised Binary Classification
In Supervised Learning, our data comes with "labels". We know the correct answers in advance. Our goal is to train a model to predict the label for new, unseen data.

"Binary Classification" specifically means there are only two possible labels. For example:
- Email: Spam (1) or Not Spam (0)
- Tumor: Malignant (1) or Benign (0)
- Image: Cat (1) or Dog (0)

### 1.3. Logistic Regression
To solve binary classification problems, we often use **Logistic Regression**. Despite having "regression" in the name, it is a classification algorithm.
Instead of predicting a continuous number, Logistic Regression predicts a *probability* (a value between 0.0 and 1.0). 

It does this by taking a linear equation ($z = wX + b$) and passing it through a **Sigmoid Function**:
$$ \sigma(z) = \frac{1}{1 + e^{-z}} $$
The Sigmoid function squishes any number, no matter how large or small, into a range strictly between 0 and 1. If the probability is greater than 0.5, we classify it as Class 1. If it's less than 0.5, it's Class 0.

---

## Chapter 2: Single Variable Linear Regression (Lab 2)

**Location**: `/home/crdy/testing/AI_lab/2/L2-LinearRegression.ipynb`

### 2.1. Theoretical Foundation
While classification separates data into categories, **Regression** predicts a continuous numerical value. For example, predicting the price of a house based on its square footage, or predicting a student's final grade based on hours studied.

Single Variable Linear Regression uses one input feature (X) to predict one output target (y).
The mathematical model is:
$$ y = wX + b $$

- **$y$**: The predicted output.
- **$X$**: The input data (e.g., hours studied).
- **$w$ (Weight)**: The slope of the line. It tells us how much $y$ changes for a one-unit change in $X$.
- **$b$ (Bias)**: The y-intercept. It tells us what $y$ would be if $X$ was exactly zero.

### 2.2. The Goal of Linear Regression
When we start, our model has random values for $w$ and $b$. It draws a random line through our data, which will likely be very wrong.
The goal of the machine learning algorithm is to adjust $w$ and $b$ iteratively until it finds the "Line of Best Fit"—the line that minimizes the total distance (error) between the predicted points on the line and the actual real-world data points.

### 2.3. Cost Function (Mean Squared Error)
To know how "wrong" our line is, we use a Cost Function. The most common one for linear regression is **Mean Squared Error (MSE)**.
It calculates the distance between every actual point and our predicted line, squares that distance (to remove negative signs and punish large errors more severely), and calculates the average.
Our objective is to minimize this MSE value to as close to 0 as possible.

---

## Chapter 3: PyTorch Fundamentals (Lab 3)

**Location**: `/home/crdy/testing/AI_lab/3/Lab3_PyTorch_assignment.ipynb`

### 3.1. Introduction to PyTorch
Up to this point, you could use libraries like Scikit-Learn or just pure mathematical formulas in Python. But as we move toward Deep Learning, we need something vastly more powerful. Enter **PyTorch**.
Developed by Meta AI, PyTorch is a highly flexible, open-source deep learning framework. It is the industry standard for AI research.

Why use PyTorch over standard Python lists or NumPy arrays?
1. **GPU Acceleration**: PyTorch can run computations on a Graphics Processing Unit (GPU), which can process thousands of mathematical operations in parallel, cutting training time from days to minutes.
2. **Automatic Differentiation (Autograd)**: PyTorch automatically calculates the complex calculus derivatives required to update neural network weights.

### 3.2. Tensors: The Heart of PyTorch
The fundamental building block in PyTorch is the **Tensor**. A tensor is simply a container for numbers.

- **0-Dimensional Tensor (Scalar)**: A single number. (e.g., `5`)
- **1-Dimensional Tensor (Vector)**: A 1D array or list. (e.g., `[1, 2, 3]`)
- **2-Dimensional Tensor (Matrix)**: A 2D grid of numbers, like a spreadsheet.
- **3-Dimensional Tensor**: A cube of numbers (e.g., a color image with Height, Width, and 3 Color Channels RGB).
- **N-Dimensional Tensor**: Tensors can have any number of dimensions to hold incredibly complex data.

### 3.3. Code Walkthrough: Creating Tensors
Let's look at the actual code required to create and inspect tensors.

```python
import torch

# 1. Creating a scalar (0D)
scalar = torch.tensor(7)
print("Scalar value:", scalar.item()) # .item() extracts the standard Python number
print("Dimensions:", scalar.ndim)     # Output: 0

# 2. Creating a vector (1D)
vector = torch.tensor([1, 2, 3, 4])
print("Vector shape:", vector.shape)  # Output: torch.Size([4])
print("Dimensions:", vector.ndim)     # Output: 1

# 3. Creating a matrix (2D)
matrix = torch.tensor([[1, 2], 
                       [3, 4]])
print("Matrix shape:", matrix.shape)  # Output: torch.Size([2, 2])
print("Dimensions:", matrix.ndim)     # Output: 2
```

### 3.4. Generating Tensors Automatically
In deep learning, we rarely type out tensors by hand. We generate them.
```python
# Create a 3x3 matrix filled with random numbers between 0 and 1
# Useful for initializing neural network weights randomly
random_tensor = torch.rand(size=(3, 3))

# Create a 2x4 matrix filled with zeros
# Useful for masking or clearing gradients
zeros_tensor = torch.zeros(size=(2, 4))

# Create a sequence of numbers from 0 to 9
range_tensor = torch.arange(start=0, end=10, step=1)
```

### 3.5. Tensor Operations
Deep learning is just millions of mathematical operations on tensors. PyTorch makes this intuitive.
```python
tensor = torch.tensor([1, 2, 3])

# Addition (Element-wise)
# Adds 10 to every element: [11, 12, 13]
added = tensor + 10 

# Multiplication (Element-wise)
# Multiplies every element by 10: [10, 20, 30]
multiplied = tensor * 10 
```

Whenever you run into shape errors in deep learning, always check three things using built-in PyTorch attributes:
1. `tensor.shape`: Are the dimensions what you expect?
2. `tensor.dtype`: Is the data type `torch.float32`? (Neural networks require floats, not integers).
3. `tensor.device`: Is the tensor on the `cpu` or the `cuda` (GPU)? You cannot multiply a tensor on the CPU with a tensor on the GPU!

---

## Chapter 4: PyTorch Linear Regression from Scratch (Lab 4)

**Location**: `/home/crdy/testing/AI_lab/4/Lab4_PyTorch_assignment.ipynb`

### 4.1. Defining a Model in PyTorch
In Lab 2, we learned the theory of linear regression. In Lab 4, we built it entirely from scratch in PyTorch. 

To build any model in PyTorch, you create a Python class that inherits from `nn.Module`. This tells PyTorch, "Hey, this is a neural network, please keep track of its internal state."

Here is the exact code we used to build the model:
```python
from torch import nn

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__() 
        # Initialize the Weight (w) randomly
        # requires_grad=True tells PyTorch to track this variable for backpropagation
        self.weights = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)
        
        # Initialize the Bias (b) randomly
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The core logic: y = wX + b
        return self.weights * x + self.bias
```
**Explanation:**
- `__init__`: This is the constructor. It runs once when the model is created. Here, we define the parameters (weights and biases) and start them off with random numbers using `torch.randn`.
- `forward`: This is the most important method. It defines the "Forward Pass". It dictates exactly what happens to the input data `x` when it passes through the model. We explicitly define the linear equation `w * x + b`.

### 4.2. Loss Function and Optimizer
Before we can train the model, we need two tools:
1. **Loss Function**: To measure how bad the model's predictions are. We used Mean Absolute Error (L1Loss).
2. **Optimizer**: The algorithm that will actively change the weights and biases to reduce the loss. We used Stochastic Gradient Descent (SGD).

```python
# Measure Mean Absolute Error
loss_fn = nn.L1Loss()

# Setup SGD optimizer. 
# params=model_0.parameters() gives the optimizer access to w and b.
# lr=0.01 is the Learning Rate (how large of a step to take when updating parameters).
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)
```

### 4.3. The 5-Step PyTorch Training Loop
This is the holy grail of PyTorch. Whether you are training a simple linear regression line or a trillion-parameter Large Language Model like ChatGPT, the core training loop follows these exact 5 steps. We ran this loop 200 times (epochs) in Lab 4.

```python
epochs = 200

for epoch in range(epochs):
    # Set the model to training mode
    model_0.train()
    
    # STEP 1: Forward Pass
    # Pass the training data (X_train) into the model to get predictions.
    y_pred = model_0(X_train)
    
    # STEP 2: Calculate the Loss
    # Compare predictions (y_pred) to the actual truth (y_train).
    loss = loss_fn(y_pred, y_train)
    
    # STEP 3: Zero Gradients
    # PyTorch accumulates gradients by default. We must reset them to zero before the next pass.
    optimizer.zero_grad()
    
    # STEP 4: Backward Pass (Backpropagation)
    # PyTorch calculates the derivatives (gradients) of the loss with respect to every weight and bias.
    loss.backward()
    
    # STEP 5: Optimizer Step
    # The optimizer uses the calculated gradients to slightly tweak the weights and bias in the right direction.
    optimizer.step()
```
By repeating these 5 steps 200 times, the random weight and bias slowly morphed into the perfect values, allowing the model to accurately draw the line of best fit through the `assignment-data2.csv` data points!

---

## Chapter 5: Deep Neural Networks & Activation Functions (Lab 5)

**Location**: `/home/crdy/testing/AI_lab/5/Lab5_PyTorch_assignment.ipynb`

### 5.1. The Limitation of Linearity
Linear Regression is great, but it has a fatal flaw: it can only draw straight lines. The world is not a straight line.
In Lab 5, we were presented with the `binary_classification_moons.csv` dataset. If you plot this data, it looks like two interlocking crescent moons. You cannot draw a single straight line to separate them perfectly. A linear model will fail spectacularly here.

### 5.2. The Illusion of Stacked Linear Layers
You might think: "If one linear layer isn't enough, I'll just stack two linear layers on top of each other!" 
In Task 2, we looked at `ModelV0`:
```python
class ModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        # Two linear layers stacked without activation
        self.layer_1 = nn.Linear(in_features=2, out_features=5)
        self.layer_2 = nn.Linear(in_features=5, out_features=1)

    def forward(self, x):
        return self.layer_2(self.layer_1(x))
```
**Why this fails:** Mathematically, a linear transformation of a linear transformation is just a single linear transformation. 
If $y = 2x$ and $z = 3y$, then $z = 3(2x) = 6x$. It's still just a straight line! `ModelV0` is utterly incapable of learning the curve of the moon dataset.

### 5.3. The Solution: Non-Linear Activation Functions
To give a neural network the power to learn complex, curved, non-linear patterns, we must inject a non-linear mathematical function between the linear layers. 
The most common activation function is **ReLU (Rectified Linear Unit)**.
ReLU is incredibly simple: 
- If the input is negative, it outputs 0.
- If the input is positive, it outputs the input unchanged.
$$ f(x) = max(0, x) $$

By adding ReLU, we break the linear chain. We explored this in `ModelV2`:
```python
class ModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU() # The magic non-linear function

    def forward(self, x):
       # We wrap the output of each linear layer inside the ReLU function
       # This allows the network to bend and fold its decision boundary!
       return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))
```
**Conclusion:** Activation functions are the true secret behind Deep Learning. They are what allow massive models to learn highly complex representations like human faces, human language, and interlocking moon datasets. Without activation functions, a neural network is just an overpriced linear regression model.

---

## Chapter 6: Recurrent Neural Networks (RNN) (Lab 6)

**Location**: `/home/crdy/testing/AI_lab/6/6.2-RNN.ipynb`

### 6.1. The Need for Memory
Standard Feedforward Neural Networks (like the ones we built in Lab 4 and 5) suffer from amnesia. They process every input completely independently. 
If I feed the network the word "I", it processes it. If I then feed it the word "love", it processes it without having any idea that it just saw the word "I".

This is useless for **Sequential Data**. Time-series data, stock prices, audio waves, and natural language all depend heavily on sequence and context.
To solve this, researchers invented the **Recurrent Neural Network (RNN)**. 

### 6.2. How an RNN Works
An RNN introduces the concept of a **Hidden State ($h_t$)**. 
When the RNN processes step 1, it generates an output AND a hidden state (a vector of numbers representing what it just saw). 
When it processes step 2, it takes the new input PLUS the hidden state from step 1. It combines them! 
This allows the network to maintain a "memory" of the past.

### 6.3. RNN Code Walkthrough
In Lab 6, we built a PyTorch RNN to predict a sequence of dishes a cook would make based on the weather.
Here is the core architecture we explored:

```python
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        
        self.hidden_size = hidden_size
        
        # PyTorch's built-in RNN layer. 
        # batch_first=True just tells PyTorch how our data is formatted.
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        
        # A final linear layer to map the hidden state to the actual output prediction
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        # The RNN layer returns the output sequence and the NEW hidden state
        out, hidden = self.rnn(x, hidden)
        
        # We pass the RNN's output through the linear layer to get the final prediction logits
        out = self.fc(out)
        return out, hidden
```

### 6.4. The Critical Parameter: `hidden_size`
In our lab, the input size was 5 (3 types of dishes one-hot encoded + 2 types of weather). 
We set the `hidden_size` to 3. What does this actually mean?

- **The Capacity of Memory:** The `hidden_size` parameter defines the dimensionality of the hidden state vector. By setting it to 3, we forced the RNN to compress all of its knowledge about the past sequence into an array of exactly 3 floating-point numbers.
- **The Trade-off:** If you make `hidden_size` too small (e.g., 1), the network doesn't have enough capacity to remember complex rules (like "If it rained yesterday and I made Dish A, make Dish B today"). It will suffer from amnesia.
- **The Danger of Large Sizes:** If you make `hidden_size` too large (e.g., 1000), the network will have an incredible memory, but it will require millions of parameters (weights). This takes massive amounts of RAM and computation time, and it risks "overfitting" (memorizing the exact training data instead of learning the general rule).
- **The Sweet Spot:** For our simple cook/weather problem, a `hidden_size` of 3 was perfectly optimal. It provided exactly enough dimensions to map back to the 3 possible dish predictions in the final linear layer without wasting computational resources.

### 6.5. Evaluating the RNN
We trained this RNN for 300 epochs using the same 5-step training loop we learned in Lab 4! 
By the end of the 300 epochs, the loss had converged close to zero. We then evaluated it by passing in all 6 possible test cases (combining every dish with every weather condition). 
Because of the RNN's hidden state, it successfully learned the transition rules and predicted the next dish with 100% accuracy.

---

## Final Thoughts
Congratulations! By working through these 6 Labs, you have crossed the threshold from standard programming into the realm of Deep Learning. 

You started by learning what AI actually is. You moved on to basic mathematical curve fitting with Linear Regression. You mastered the syntax of Tensors in PyTorch. You built a model from scratch and manually executed a backpropagation training loop. You discovered how Activation Functions allow models to perceive complex realities. Finally, you stepped into the fourth dimension of time by utilizing Recurrent Neural Networks to process sequential memory.

The repository containing all your files, code, and executed Jupyter Notebooks represents a complete and profound foundational journey into modern Artificial Intelligence.




