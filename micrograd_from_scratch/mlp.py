from neuron import Neuron
from layer import Layer
from value_class import Value
class MLP():
    #nin is number of inputs and nout is list of layer sizes 
    def __init__(self, nin, nouts):
        sizes = [nin] + nouts # all sizes combined 
        # list of layer objectss built from thsoe sizes 
        self.layers = [Layer(nin, nout) for nin, nout in zip(sizes, sizes[1:])]
    
    def __call__(self, x):
        #input data gets replaced by each layers output as it passes through
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

mlp = MLP(2, [3, 1])
x = [Value(1.0), Value(0.5)]
print(mlp(x))