# ML From Scratch

Learning ML from mathematical foundations up. Derivatives by hand before touching PyTorch, computation graphs drawn on paper before implementing them in code.

## Status

**Started:** May 2026  
**Current state:** Work in progress.

---

## mathematical foundations

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
### Created arrays using:
  - lists
  - `zeros`
  - `ones`
  - `arange`
  - `linspace`

## Gave each array a different dtypes. 
- Checked how much memory each uses with array.nbytes. 
- Multiplied a float64 array by a float32 array and answered what dtype is the result and why.
## micrograd from scratch 

---