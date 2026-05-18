import math #for the tanh method
class Value():
    def __init__(self, data, children=(), _op=''):
        self.data = data 
        self._prev = set(children) #convert children to set to avoid duplicates
        self._op = _op
        self.grad = 0
        self._backward = lambda: None
    
    
    def __repr__(self):
        return f"Value(data={self.data})"

    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward =_backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward =_backward
        return out

    def __rmul__(self, other):
        return self * other
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v) 
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            v._backward()
        
    def tanh(self):
        out = Value(math.tanh(self.data), (self,), 'tanh')
        def _backward():
            self.grad += (1 - out.data**2) * out.grad
        out._backward = _backward
        return out
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)

    def __pow__(self, other):
        out = Value(self.data ** other, (self,), f'**{other}')
        def _backward():
            self.grad += other * (self.data ** (other-1)) * out.grad
        out._backward = _backward
        return out
    def __truediv__(self, other):
        return self * other**-1


a = Value(0.5)
a_tanh = a.tanh()
a_tanh.backward()
print(a.grad)