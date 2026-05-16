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
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward =_backward
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward =_backward
        return out
    
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

a = Value(2)
b = Value(3)
c = Value(4)
e = (a + b) * c

e.backward()

print(a.grad) 
print(b.grad)  
print(c.grad)  