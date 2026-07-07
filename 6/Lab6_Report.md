# Lab 6 (B1): Recurrent Neural Networks (RNN)

## 1. Objective
The objective of this lab is to understand the architecture and working mechanism of Recurrent Neural Networks (RNNs) by running and analyzing a simple sequence-prediction model.

## 2. Methodology
The provided PyTorch code (`6.2-RNN.ipynb`) simulates a sequence prediction problem. The scenario involves predicting a cook's next dish (A, B, or C) based on two inputs: the current dish and the current weather (Sunny or Rainy). 

* **Data Generation:** A sequence dataset of length 2000 is generated. The underlying rule is: if it is Sunny, the cook makes the same dish. If it is Rainy, the cook transitions to the next dish in the sequence (A -> B -> C -> A).
* **Encoding:** The inputs are one-hot encoded. The dish is represented by 3 dimensions and the weather by 2 dimensions, resulting in a combined input size of 5.
* **Model Architecture:** A simple Vanilla RNN (`nn.RNN`) is constructed with `input_size=5` and `hidden_size=3`. The output of the RNN is passed to a fully connected linear layer (`nn.Linear`) to output the prediction logits.
* **Training:** The model is trained for 300 epochs using `CrossEntropyLoss` and the Adam optimizer (learning rate = 0.01).

## 3. Results
After training for 300 epochs, the loss converges and the model successfully learns the underlying transitions. When evaluated against all 6 possible test cases (combinations of the Current Dish and Weather), the model predicts the next dish correctly, proving it has learned the sequence dependencies.

## 4. Discussion
**What does `hidden_size` mean in an RNN?**

In an RNN, the `hidden_size` determines the dimensions of the hidden state vector ($h_t$).

* **The "Memory" of the Network:** You can think of the hidden state as the network's short-term memory. The `hidden_size` dictates how much information the network can "remember" from past time steps and carry over to the next time step.
* **Complexity vs. Performance:** A larger `hidden_size` allows the network to learn more complex patterns and remember longer sequences, but it also increases the number of weights (parameters) the network has to train, making it slower and more prone to overfitting.
* **In this specific code:** The `hidden_size` is set to `3`. This means at any given time step, the RNN compresses its understanding of the past sequence into an array of exactly 3 numbers. This perfectly aligns with the output because those 3 hidden numbers are then multiplied by a 3 × 3 weight matrix to predict the 3 possible dishes (A, B, C).

## 5. Conclusion
Through this experiment, one can practically learn that RNN is very efficient in handling sequential information. In contrast to feed-forward neural nets, the hidden state of RNN enables the network to keep the memory from the earlier input. From the experiment, we learned that setting the right size of the `hidden_size` would allow the net to have just enough "memory" capacity.