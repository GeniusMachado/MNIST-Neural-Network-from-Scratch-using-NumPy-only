## Two-Layer Neural Network from Scratch (NumPy Only)

This project builds a **two-layer neural network from scratch using only NumPy**.  
No TensorFlow and no Keras are used.

The model is trained on the **MNIST handwritten digit dataset (0–9)**.

---

## 1. Loading and Preparing the Data

```python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')
```

### Dataset Description

Each row contains:

- The label (0–9)
- 784 pixel values (28 × 28 image flattened)

---

## Convert to NumPy and Shuffle

```python
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)
```

- `m` = number of training examples  
- `n` = number of columns (1 label + 784 pixels)

---

## Create Development and Training Sets

### Development Set

```python
data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.
```

Important:

- `.T` transposes the matrix  
- Each column becomes **one example**  
- Shape of `X` becomes **(784, number_of_examples)**  
- Division by 255 normalizes values to **[0, 1]**

---

### Training Set

```python
data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
```

---

## 2. Model Architecture

- **Input layer:** 784 units  
- **Hidden layer:** 10 units (ReLU)  
- **Output layer:** 10 units (Softmax)

Architecture:

```
784 → 10 → 10
```

---

## 3. Parameter Initialization

```python
def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2
```

Shapes:

- `W1`: (10 × 784)  
- `b1`: (10 × 1)  
- `W2`: (10 × 10)  
- `b2`: (10 × 1)

Weights are initialized between **-0.5 and 0.5**.

---

## 4. Activation Functions

### ReLU

```python
def ReLU(Z):
    return np.maximum(Z, 0)
```

ReLU(x) = max(0, x)

---

### Softmax

```python
def softmax(Z):
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    return A
```

Softmax converts raw scores into probabilities.  
Each column represents a probability distribution across digits 0–9.

---

## 5. Forward Propagation

```python
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2
```

Mathematically:

Z1 = W1X + b1  
A1 = ReLU(Z1)  
Z2 = W2A1 + b2  
A2 = Softmax(Z2)

---

## 6. One-Hot Encoding

```python
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y
```

Example:

```
3 → [0 0 0 1 0 0 0 0 0 0]
```

---

## 7. Backward Propagation

### Output Layer Error

```python
dZ2 = A2 - one_hot_Y
```

This comes from the derivative of cross-entropy loss with softmax.

---

### Gradients for W2 and b2

```python
dW2 = 1 / m * dZ2.dot(A1.T)
db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
```

---

### Hidden Layer Backpropagation

```python
def ReLU_deriv(Z):
    return Z > 0

dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
```

---

### Gradients for W1 and b1

```python
dW1 = 1 / m * dZ1.dot(X.T)
db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
```

All operations are fully vectorized.

---

## 8. Parameter Update

```python
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2
```

Gradient descent rule:

θ = θ − α ∇J(θ)

---

## 9. Training Loop

```python
def gradient_descent(X, Y, alpha, iterations):
```

Steps:

1. Initialize parameters  
2. Forward propagation  
3. Backward propagation  
4. Update parameters  
5. Print accuracy every 10 iterations  

Uses **full-batch gradient descent**.

---

## 10. Making Predictions

```python
def get_predictions(A2):
    return np.argmax(A2, 0)
```

Selects the digit with highest probability.

---

## 11. Evaluation

```python
dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
get_accuracy(dev_predictions, Y_dev)
```

Accuracy: **~84–85%**

Strong performance for a small network built entirely from scratch.

---

## Key Concepts

- Fully vectorized implementation  
- Each column represents one example  
- Softmax + Cross-Entropy gradient simplifies to:

```
A2 − Y_one_hot
```

- ReLU introduces non-linearity  
- Manual forward and backward propagation  
- Full-batch gradient descent  

---

## Possible Improvements

- Use He initialization  
- Increase hidden layer size (64 or 128 units)  
- Use mini-batch gradient descent  
- Train longer  
- Add additional hidden layers  
- Implement learning rate scheduling  
