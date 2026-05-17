from value_class import Value
import random
class Neuron():
    #only takes number of input because training will adjust the weight and other hyperparameters
    def __init__(self, nin):
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(nin)] #list of weight per input
        self.bias = Value(random.uniform(-1, 1))

    def __call__(self, x):
        acts = [xi * wi for xi, wi in zip(x, self.weights)]
        
    def __call__(self, x):
        acts = [xi * wi for xi, wi in zip(x, self.weights)]
        return sum(acts, self.bias).tanh()
    
n = Neuron(2)
x = [Value(1.0), Value(0.5)]
out = n(x)
out.backward()

print(n.weights[0].grad)
print(n.weights[1].grad)
print(n.bias.grad)