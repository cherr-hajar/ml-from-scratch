from neuron import Neuron
from value_class import Value
class Layer():
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        return [neuron(x) for neuron in self.neurons]
    
    
    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
layer = Layer(2, 3)
x = [Value(1.0), Value(0.5)]
print(layer(x))

