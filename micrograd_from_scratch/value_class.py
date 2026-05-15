class Value:
    def __init__(self, data, _children = (), _op=''):
        self.data= data
        self._op= _op
        self._prev = set(_children)
    
    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        return out
    
    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        return out
    
a = Value(2)
b = Value(3)
c = Value(4)
e = (a + b) * c
print(e.data) 
print(e._op) 
print(e._prev) 


