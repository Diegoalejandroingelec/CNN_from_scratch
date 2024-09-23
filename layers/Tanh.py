from layers.activation_layer import Activation
import numpy as np
class Tanh(Activation):
    def __init__(self):
        activation_function= lambda x: np.tanh(x)
        prime_activation_function = lambda x: 1 - np.tanh(x)**2
        super().__init__(activation_function, prime_activation_function)