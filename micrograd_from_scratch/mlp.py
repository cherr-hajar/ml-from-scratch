from neuron import Neuron
from layer import Layer
from value_class import Value
class MLP():
    def __init__(self, nin, nouts):
        sizes = [nin] + nouts
        self.layers = [Layer(nin, nout) for nin, nout in zip(sizes, sizes[1:])]
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
mlp = MLP(2, [3, 1])
x = [Value(1.0), Value(0.5)]
print(mlp(x))