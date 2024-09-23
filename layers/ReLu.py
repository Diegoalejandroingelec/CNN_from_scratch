from layers.activation_layer import Activation
import numpy as np
class ReLu(Activation):
    def __init__(self):
        activation_function= lambda x: np.maximum(0, x)
        prime_activation_function = lambda x: np.where(x>=0,1,0)
        super().__init__(activation_function, prime_activation_function)