# Import the base Layer class and numpy for numerical operations
from layers.layer import Layer
import numpy as np

# Activation class for applying an activation function and its derivative, inherits from Layer
class Activation(Layer):
    def __init__(self, activation_function, prime_activation_function):
        """
        Initialize the Activation layer with the activation function and its derivative (for backpropagation).
        
        Args:
            activation_function: A function that applies a non-linear transformation to the input (e.g., sigmoid, ReLU).
            prime_activation_function: The derivative of the activation function, used for backpropagation.
        """
        self.activation_function = activation_function  # Store the activation function
        self.prime_activation_function = prime_activation_function  # Store the derivative of the activation function

    # Forward pass of the Activation layer
    def forward(self, X):
        """
        Perform the forward pass by applying the activation function to the input.
        
        Args:
            X: Input data (output from the previous layer).
        
        Returns:
            The output of the activation function applied to the input.
        """
        self.X = X  # Store the input for use in the backward pass
        return self.activation_function(X)  # Apply the activation function to the input

    # Backward pass for the Activation layer (used during backpropagation)
    def backward(self, back_propagated_gradient, lr):
        """
        Perform the backward pass by multiplying the backpropagated gradient with the derivative of the activation function.
        
        Args:
            back_propagated_gradient: Gradient of the loss with respect to the output of this layer.
            lr: Learning rate (not used here, as Activation layers don't have learnable parameters).
        
        Returns:
            The gradient of the loss with respect to the input of this layer.
        """
        # Multiply the backpropagated gradient by the derivative of the activation function applied to the input
        return np.multiply(back_propagated_gradient, self.prime_activation_function(self.X))
