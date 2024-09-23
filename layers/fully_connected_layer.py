# Import the base Layer class and numpy for numerical operations
from layers.layer import Layer
import numpy as np

# FullyConnected layer class for a densely connected neural network layer, inherits from Layer
class FullyConnected(Layer):
    def __init__(self, input_size, output_size, W=None, B=None):
        """
        Initialize a FullyConnected layer with input and output sizes.
        
        Args:
            input_size: Size of the input vector (number of input neurons).
            output_size: Size of the output vector (number of output neurons).
            W: (Optional) Predefined weight matrix. If not provided, it is initialized randomly.
            B: (Optional) Predefined bias vector. If not provided, it is initialized randomly.
        """
        self.input_shape = input_size  # Store the input size (number of input neurons)
        self.output_shape = output_size  # Store the output size (number of output neurons)
        
        # Initialize weights (W) either from the provided values or randomly if not given
        if W is None:
            self.W = np.random.randn(output_size, input_size)  # Randomly initialize weights (output_size x input_size)
        else:
            self.W = W  # Use the provided weight matrix
        
        # Initialize biases (B) either from the provided values or randomly if not given
        if B is None:
            self.B = np.random.randn(output_size, 1)  # Randomly initialize biases (output_size x 1)
        else:
            self.B = B  # Use the provided bias vector

    # Method to set the weights and biases manually
    def set_weights(self, W, B):
        """
        Set custom weights and biases for the FullyConnected layer.
        
        Args:
            W: The new weight matrix to set.
            B: The new bias vector to set.
        """
        self.W = W  # Set the weight matrix
        self.B = B  # Set the bias vector

    # Forward pass for the FullyConnected layer
    def forward(self, X):
        """
        Perform the forward pass by applying the linear transformation (W * X + B).
        
        Args:
            X: Input data (vector from the previous layer or input).
        
        Returns:
            Output of the layer after applying the weights and biases.
        """
        self.X = X  # Store the input for use in the backward pass
        return np.dot(self.W, self.X) + self.B  # Linear transformation (output = W * X + B)

    # Backward pass for the FullyConnected layer (for backpropagation)
    def backward(self, back_propagated_gradient, lr):
        """
        Perform the backward pass, computing the gradients and updating the weights and biases.
        
        Args:
            back_propagated_gradient: Gradient of the loss with respect to the output of this layer.
            lr: Learning rate for updating the weights and biases.
        
        Returns:
            Gradient of the loss with respect to the input of this layer.
        """
        W = self.W.copy()  # Make a copy of the current weights for use in computing the input gradient
        
        # Update the weights using the gradient of the loss with respect to the weights
        self.W -= lr * np.dot(back_propagated_gradient, self.X.T)
        
        # Update the biases using the gradient of the loss with respect to the biases
        self.B -= lr * back_propagated_gradient
        
        # Return the gradient of the loss with respect to the input of this layer (W.T * output_gradient)
        return np.dot(W.T, back_propagated_gradient)
