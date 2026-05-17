# ML From Scratch

Learning ML from mathematical foundations up. Derivatives by hand before touching PyTorch, computation graphs drawn on paper before implementing them in code.

## Status

**Started:** May 2026  
**Current state:** Work in progress.

---

## Mathematical foundations

### derivatives
- Partial derivatives by hand
- Numerical gradient estimation
- Contour plot of:
  
  f(x,y) = x² + y²

- Chain rule on composed functions:
  
  f(g(x))

- Verified df/dx analytically and numerically using finite differences
- Interpreted chain rule as propagation of change through layers
- Connected repeated multiplication of local derivatives to vanishing gradients in neural networks

### computation graphs
- Computation graph drawn by hand
- Forward and backward pass intuition
- Numerical gradient verification for:
  
  e = (a + b) × c

## Numpy 
### created arrays using:
  - lists
  - `zeros`
  - `ones`
  - `arange`
  - `linspace`

### gave each array a different dtypes. 
- Checked how much memory each uses with array.nbytes. 
- Multiplied a float64 array by a float32 array and answered what dtype is the result and why.

### create a 5x5 matrix and extract the diagonal without using numpy.diagonal
- Tried 2 methods one with integer array indexing and the other using np.arange 

### extracte all elements greater than 10 using boolean indexing. 
- Did it with no intermediate variable, so the boolean array never gets stored anywhere

### reversed Every Row Without a Loop

## Micrograd from scratch 
### built computation graph
- Built a computational graph with th expression from ml-from-scratch/mathematical_foundations
/computation_graphs
- When you write e = (a + b) * c, most languages just compute 20 and forget everything. Your Value class remembers: the result, what operation made it, which values were its inputs

### wrote the backward pass for addition and multiplication
 - when called, it tells a and b how much they contributed to the output.

### implemented full backward pass
- topological sort to find all nodes in correct order
- backpropagation through the full expression e = (a + b) * c
- verified all gradients match manual calculation
- implemented tanh activation function with forward and backward pass

### implemented Neuron class 
- takes n inputs with random weights and bias
- computes weighted sum, adds bias, passes through tanh
- gradients flow automatically through weights and bias
---