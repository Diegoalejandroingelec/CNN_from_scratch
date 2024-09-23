from layers.layer import Layer
import numpy as np

class Activation(Layer):
    def __init__(self,activation_function, prime_activation_function):
        self.activation_function = activation_function
        self.prime_activation_function = prime_activation_function

    def forward(self,X):
        self.X = X
        return self.activation_function(X)

    def backward(self,back_propagated_gradient, lr):
        
        return np.multiply(back_propagated_gradient,self.prime_activation_function(self.X))
