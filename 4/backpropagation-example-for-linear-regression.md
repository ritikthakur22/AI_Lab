# Backpropagation for Linear Regression (Simple Example)

Suppose we have a linear regression model:

$$
y = wx + b
$$

and the Mean Squared Error (MSE) loss for a single training example:

$$
L = (y - t)^2
$$

where:

- $x$ = input
- $t$ = true target
- $y$ = prediction
- $w$ = weight
- $b$ = bias

---

## Given

Training example:

$$
x = 3,\quad t = 10
$$

Current parameters:

$$
w = 2,\quad b = 1
$$

---

## Step 1: Forward Pass

Compute the prediction:

$$
y = wx + b
$$

$$
y = 2(3) + 1 = 7
$$

Compute the loss:

$$
L = (y - t)^2
$$

$$
L = (7 - 10)^2
$$

$$
L = 9
$$

Forward flow:

```text
     x=3
      ↓
  y=wx+b = 7
      ↓
  L=(y-t)² = 9
```

---

## Step 2: Compute Gradient of Loss w.r.t. Prediction

Loss:

$$
L = (y - t)^2
$$

Differentiate with respect to $y$:

$$
\frac{\partial L}{\partial y}
=
2(y - t)
$$

Substitute values:

$$
\frac{\partial L}{\partial y}
=
2(7 - 10)
=
-6
$$

So:

$$
\frac{\partial L}{\partial y} = -6
$$

Interpretation:

A small increase in $y$ would reduce the loss.

---

## Step 3: Compute Gradient w.r.t. Weight $w$

Prediction:

$$
y = wx + b
$$

Differentiate:

$$
\frac{\partial y}{\partial w}
=
x
=
3
$$

Apply the chain rule:

$$
\frac{\partial L}{\partial w}
=
\frac{\partial L}{\partial y}
\cdot
\frac{\partial y}{\partial w}
$$

$$
=
(-6)(3)
=
-18
$$

So:

$$
\frac{\partial L}{\partial w}
=
-18
$$

---

## Step 4: Compute Gradient w.r.t. Bias $b$

Differentiate:

$$
\frac{\partial y}{\partial b}
=
1
$$

Apply the chain rule:

$$
\frac{\partial L}{\partial b}
=
\frac{\partial L}{\partial y}
\cdot
\frac{\partial y}{\partial b}
$$

$$
=
(-6)(1)
=
-6
$$

So:

$$
\frac{\partial L}{\partial b}
=
-6
$$

---

## Final Gradients

$$
\frac{\partial L}{\partial w}
=
-18
$$

$$
\frac{\partial L}{\partial b}
=
-6
$$

These tell us how the loss changes when $w$ and $b$ change.

---

## Step 5: Gradient Descent Update

Let the learning rate be:

$$
\eta = 0.1
$$

Update rule:

$$
w \leftarrow w - \eta \frac{\partial L}{\partial w}
$$

$$
b \leftarrow b - \eta \frac{\partial L}{\partial b}
$$

Update $w$:

$$
w
=
2 - 0.1(-18)
=
3.8
$$

Update $b$:

$$
b
=
1 - 0.1(-6)
=
1.6
$$

New parameters:

$$
w = 3.8
$$

$$
b = 1.6
$$

---

## Why This Is Called Backpropagation

Forward pass:

```text
x,w,b
   ↓
   y
   ↓
 loss
```

Backward pass:

```text
loss
  ↑
 dL/dy
  ↑
 dL/dw, dL/db
```

The error starts at the loss and is propagated backward through the computation using the chain rule.

For linear regression, backpropagation is simply:

1. Compute the prediction.
2. Compute the loss.
3. Compute $\frac{\partial L}{\partial y}$.
4. Use the chain rule to compute:
   - $\frac{\partial L}{\partial w}$
   - $\frac{\partial L}{\partial b}$
5. Update the parameters using gradient descent.