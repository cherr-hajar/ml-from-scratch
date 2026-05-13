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