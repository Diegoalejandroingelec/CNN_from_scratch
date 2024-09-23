# Import the base Layer class and numpy for numerical operations
from layers.layer import Layer
import numpy as np

# Flatten layer class for reshaping multi-dimensional data into 2D (vector) form, inherits from Layer
class Flatten(Layer):
    def __init__(self, input_shape, output_shape):
        """
        Initialize the Flatten layer with input and output shapes.
        
        Args:
            input_shape: Shape of the input data (e.g., (channels, height, width)).
            output_shape: Shape of the output data after flattening (e.g., (flattened_size, 1)).
        """
        self.input_shape = input_shape  # Store the input shape (e.g., (channels, height, width))
        self.output_shape = output_shape  # Store the output shape (e.g., (flattened_size, 1))

    # Forward pass of the Flatten layer
    def forward(self, X):
        """
        Reshape the input data into the output shape (flatten the input).
        
        Args:
            X: Input data, a multi-dimensional array (e.g., 3D image).
        
        Returns:
            Reshaped data in the form of a flattened vector.
        """
        return np.reshape(X, self.output_shape)  # Flatten the input data into the specified output shape

    # Backward pass of the Flatten layer (for backpropagation)
    def backward(self, output_gradient, lr):
        """
        Reshape the gradient from the output shape back to the original input shape.
        
        Args:
            output_gradient: Gradient of the loss with respect to the output (flattened shape).
            lr: Learning rate (not used here, as Flatten has no learnable parameters).
        
        Returns:
            Reshaped gradient, matching the original input shape of the layer.
        """
        return np.reshape(output_gradient, self.input_shape)  # Reshape the gradient back to the input shape






