class Value:
    def __init__(self, data, _children = (), _op=''):
        self.data= data
        self._op= _op
        self._prev = set(_children)
        self.grad = 0
    

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad
        out._backward = _backward
        return out
        
    
    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += out.grad * other.data
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
a = Value(2)
b = Value(3)
e = a * b

e.grad = 1
e._backward()

print(a.grad) 
print(b.grad)


