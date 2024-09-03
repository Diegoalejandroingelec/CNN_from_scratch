from activation_layer import Activation
import numpy as np
class Sigmoid(Activation):
    def __init__(self):
        activation_function= lambda x: 1/(1+np.exp(-x))
        prime_activation_function = lambda x: activation_function(x)*(1-activation_function(x))
        super().__init__(activation_function, prime_activation_function)