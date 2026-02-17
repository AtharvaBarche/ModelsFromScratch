# Linear Regression from Scratch

This project implements a **Linear Regression** model from scratch using Python, without relying on machine learning libraries like scikit-learn. The implementation demonstrates the fundamental concepts of linear regression including cost function, gradient computation, and gradient descent optimization.

## 📁 Project Structure

- `linearRegresson.ipynb` - Jupyter notebook containing the complete implementation
- `Salary Data.csv` - Dataset containing years of experience and corresponding salaries
- `README.md` - This file

## 📊 Dataset

The project uses a salary prediction dataset with two features:
- **YearsExperience**: Number of years of professional experience
- **Salary**: Corresponding salary amount

## 🔧 Implementation Details

### 1. **Cost Function (Mean Squared Error)**

The cost function measures how well our model fits the data by calculating the average squared difference between predicted and actual values:

```python
def cost_funtion(x, y, w, b):
    m = len(x)
    cost_sum = 0
    for i in range(m):
        f = w * x[i] + b
        cost = (f - y[i]**2)
        cost_sum += cost
    total_cost = (1/(2*m)) * cost_sum
    return total_cost
```

**Formula**: J(w,b) = (1/2m) Σ(f(x) - y)²

Where:
- `m` = number of training examples
- `f(x) = w*x + b` = predicted value
- `y` = actual value

### 2. **Gradient Function**

Computes the partial derivatives of the cost function with respect to parameters `w` (weight) and `b` (bias):

```python
def gradient_function(x, y, w, b):
    m = len(x)
    dc_dw = 0
    dc_db = 0
    for i in range(m):
        f = w * x[i] + b
        dc_dw += (f - y[i]) * x[i]
        dc_db += (f - y[i])
    dc_dw = (1/m) * dc_dw
    dc_db = (1/m) * dc_db
    return dc_dw, dc_db
```

**Formulas**:
- ∂J/∂w = (1/m) Σ(f(x) - y) * x
- ∂J/∂b = (1/m) Σ(f(x) - y)

### 3. **Gradient Descent**

Iteratively updates the parameters to minimize the cost function:

```python
def gradient_decent(x, y, alpha, iterations):
    w = 0 
    b = 0
    for i in range(iterations):
        dc_dw, dc_db = gradient_function(x, y, w, b) 
        w = w - alpha * dc_dw
        b = b - alpha * dc_db
        print(f"Iteration {i}: Cost{cost_funtion(x, y, w, b)}")
    return w, b
```

**Update Rules**:
- w = w - α * (∂J/∂w)
- b = b - α * (∂J/∂b)

Where `α` (alpha) is the learning rate.

## 🚀 How to Use

1. **Load the Data**:
   ```python
   training_set = pd.read_csv("Salary Data.csv")
   x_train = training_set['YearsExperience'].values
   y_train = training_set['Salary'].values
   ```

2. **Train the Model**:
   ```python
   final_w, final_b = gradient_decent(x_train, y_train, alpha=0.01, iterations=1000)
   ```

3. **Visualize Results**:
   ```python
   plt.scatter(x_train, y_train, label='Data Points')
   x_vals = np.linspace(min(x_train), max(x_train), 100)
   y_vals = final_w * x_vals + final_b
   plt.plot(x_vals, y_vals, color='red', label="Regression Line")
   plt.show()
   ```

## 📈 Key Concepts

### Linear Regression Model
The hypothesis function is: **h(x) = wx + b**

Where:
- `w` = weight (slope of the line)
- `b` = bias (y-intercept)
- `x` = input feature
- `h(x)` = predicted output

### Gradient Descent Algorithm
1. Initialize parameters (w, b) to zero
2. Calculate cost function
3. Compute gradients (partial derivatives)
4. Update parameters using learning rate
5. Repeat steps 2-4 for specified iterations

### Hyperparameters
- **Learning Rate (α)**: Controls step size during optimization (typically 0.001 - 0.1)
- **Iterations**: Number of times to update parameters (typically 100 - 10000)

## 📚 Dependencies

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

## 🎯 Results

The model successfully learns the linear relationship between years of experience and salary by:
- Minimizing the cost function through gradient descent
- Finding optimal values for weight (w) and bias (b)
- Plotting a regression line that best fits the training data

## 🔍 Learning Outcomes

This implementation helps understand:
- How linear regression works under the hood
- The mathematics behind gradient descent
- Cost function optimization
- Parameter initialization and updates
- Vectorization vs. iterative approaches

## 📝 Notes

- This is an educational implementation focused on understanding fundamentals
- For production use, consider using optimized libraries like scikit-learn
- The implementation uses a simple iterative approach rather than vectorized operations for clarity

## 🤝 Contributing

Feel free to fork this project and experiment with:
- Different learning rates
- Various numbers of iterations
- Feature scaling/normalization
- Multiple features (multiple linear regression)
- Regularization techniques

---

**Author**: Implementation from scratch for educational purposes  
**Date**: 2026
