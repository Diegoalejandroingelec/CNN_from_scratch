# Import the base Activation layer and numpy for numerical operations
from layers.activation_layer import Activation
import numpy as np

# Sigmoid class for applying the sigmoid activation function, inherits from the Activation class
class Sigmoid(Activation):
    def __init__(self):
        """
        Initialize the Sigmoid activation layer.
        It defines the sigmoid function and its derivative (for backpropagation).
        """
        # Sigmoid activation function: 1 / (1 + exp(-x))
        activation_function = lambda x: 1 / (1 + np.exp(-x))
        
        # Derivative of the sigmoid function for backpropagation:
        # sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        prime_activation_function = lambda x: activation_function(x) * (1 - activation_function(x))
        
        # Call the constructor of the parent class (Activation) with the sigmoid and its derivative
        super().__init__(activation_function, prime_activation_function)
