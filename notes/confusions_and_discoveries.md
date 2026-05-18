# Confusions and Discoveries

---

## mathematical_foundations

### derivatives

* Syntax error from an extra `=` in the numerical derivative line.
* Case mismatch between `x` and `X` broke meshgrid and plotting.
* Numerical differentiation is always approximate because `h` can never be 0, only very small.

* Chain rule initially felt like “multiplying formulas”, but it is actually how change travels through composed functions step by step.
* Vanishing gradient intuition clarified: repeated multiplication of numbers < 1 shrinks the signal toward zero, starving early layers of learning signal.
* Gradients are propagated backward layer by layer, each contributing a local derivative.

---

### computation_graphs

* The term "nodes" was confusing at first until I drew the computation graph by hand.

e = (a + b) × c, where a=2, b=3, c=4

FORWARD PASS:
a + b = 5  
e = 20  

BACKWARD PASS:
de/da = 4  
de/db = 4  
de/dc = 5  

d(a+b)/da = 1  
d(a+b)/db = 1  
d(e)/d(a+b) = c = 4  
d(e)/dc = (a + b) = 5  

* Implementing the computation graph matched the manual backward pass exactly.

---

## numpy_foundations

### arrays and core intuition

* Learned:
  - `arange` -> step size controls spacing
  - `linspace` -> number of points controls resolution
  - They are different ways of defining a range, not interchangeable
  - Arrays are fixed-type, shape-driven structures designed for vectorized computation
  - When we multiply a float64 array by a float32 array numpy promotes the operation to the safer, more precise dtype: float64.

### indexing and slicing

*Learned:
- when you pass two arrays of equal shape, numpy pairs them up element by element and fetches one value per pair

## micrograd_from_scratch
### value_class

*Learned:
- defaults go in the signature _op='' not in the body
- self + other causes infinite recursion, you need self.data + other.data
- __add__ only takes other, nothing else
- __repr__ makes your objects print nicely
- every operation owns its own _backward that knows its specific chain rule
- addition's rule is simple: both inputs get exactly out.grad because nudging either by 1 changes the output by 1
- += vs = — gradients accumulate, not overwrite
- out.grad not out.data, gradient flows back, not the value
- each input's gradient is the other input's value times out.grad
- build_topo visits every node exactly once using visited set to avoid duplicates
- topo builds in forward order [a, b, tmp, c, e], reversed() flips it for backprop
- self.grad = 1 because the output's gradient with respect to itself is always 1
- v not self inside build_topo because v changes each recursion, self is always the same object
- _backward = lambda: None is replaced by the real gradient function inside each operation
- Why tuple → set: convention for passing, set for storing
- Why _prev not _children: just internal naming
- The _ prefix: means "private, don't touch from outside"
- Why not define children as a set from the start: because (a, b) is cleaner to pass than {a, b}
- using self.grad instead of out.grad, the gradient flows back from out, not from self
- tanh backward is (1 - out.data**2) * out.grad — same chain rule pattern, just a different derivative
- out.data is reused because tanh was already computed in the forward pass
- __pow__ backward uses power rule: n * x^(n-1) * out.grad
- __truediv__ is just multiplying by the inverse: self * other**-1

### neuron.py
- weights must be Value objects not plain numbers, otherwise gradients can't flow through them
- weights start random to break symmetry, if all start at 0 every neuron learns the same thing
- x.tanh() not tanh(x): tanh is a method on Value, not a standalone function
- zip pairs two lists element by element
- sum(acts, self.bias) starts the sum from bias instead of 0

### layer.py
-A Layer is just multiple neurons all receiving the same input but with different weights
- neuron(x) works because Neuron has __call__ defined

### mlp.py
- sizes = [nin] + nouts just combines all layer sizes into one list
- zip(sizes, sizes[1:]) creates consecutive pairs — each pair becomes one Layer
- x = layer(x) replaces x each iteration — output of one layer becomes input of next

### training_loop.ipynb
- list comprehension [1 if y > 2*x+3 else 0 for x, y in points] generates all labels in one line
- zip(points, labels) pairs each point with its label for plotting
- MSE = average of squared differences between predictions and labels
- squaring errors punishes big mistakes more than small ones
- p[0] needed because MLP returns a list, not a single Value
