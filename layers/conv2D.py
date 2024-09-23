# Import the base Layer class and necessary libraries
from layers.layer import Layer  # Base Layer class
from scipy import signal  # For 2D cross-correlation and convolution operations
import numpy as np  # Numpy for numerical operations

# Conv2D class for performing 2D convolution operations, inheriting from Layer
class Conv2D(Layer):
    def __init__(self, n_kernels, kernel_shape, input_shape):
        """
        Initialize a Conv2D layer with multiple kernels (filters).
        
        Args:
            n_kernels: Number of convolutional kernels (filters) in the layer.
            kernel_shape: Shape of each kernel (height, width).
            input_shape: Shape of the input image (channels, height, width).
        """
        self.n_channels = input_shape[0]  # Number of input channels (e.g., 1 for grayscale, 3 for RGB)
        self.n_kernels = n_kernels  # Number of convolutional filters (kernels)
        self.input_shape = input_shape  # Input shape (channels, height, width)

        # Output shape based on the input dimensions and kernel size
        self.output_shape = (n_kernels, 
                             input_shape[1] - kernel_shape[0] + 1,  # Output height
                             input_shape[2] - kernel_shape[1] + 1)  # Output width

        # Randomly initialize the kernels (filters) with shape (n_kernels, n_channels, kernel_height, kernel_width)
        self.K = np.random.randn(n_kernels, self.n_channels, kernel_shape[0], kernel_shape[1])

        # Randomly initialize the biases (same shape as the output)
        self.B = np.random.randn(*self.output_shape)

    # Forward pass of the Conv2D layer
    def forward(self, X):
        """
        Perform the forward pass by applying 2D convolution to the input.
        
        Args:
            X: Input image (numpy array of shape (channels, height, width)).
        
        Returns:
            Y: Output of the convolution layer (feature maps) after adding bias.
        """
        self.X = X  # Store the input for use in the backward pass
        Y = self.B.copy()  # Start with the bias as the base for the output
        cross_corr = np.zeros(self.output_shape)  # Initialize an array for cross-correlation results

        # Loop over each kernel (filter)
        for i in range(self.n_kernels):
            # For each kernel, loop over each input channel
            for j in range(self.n_channels):
                # Perform 2D cross-correlation between the input channel and the corresponding kernel slice
                cross_corr[i] += signal.correlate2d(X[j], self.K[i][j], mode='valid')

            # Add the result to the output Y (after adding bias)
            Y[i] += cross_corr[i]

        return Y  # Return the feature maps after convolution and bias addition

    # Backward pass for the Conv2D layer (used during backpropagation)
    def backward(self, output_gradient, lr):
        """
        Perform the backward pass (backpropagation) to compute gradients and update weights.
        
        Args:
            output_gradient: Gradient of the loss with respect to the output of this layer.
            lr: Learning rate for updating the kernels and biases.
        
        Returns:
            gradient_X: Gradient of the loss with respect to the input of this layer.
        """
        K_o = self.K.copy()  # Make a copy of the current kernels to use in the backward pass
        gradient_X = np.zeros(self.input_shape)  # Initialize gradient with respect to the input

        # Loop over each kernel to update the biases and kernels
        for i in range(self.n_kernels):
            # Update the bias by subtracting the learning rate times the gradient
            self.B[i] -= lr * output_gradient[i]
            
            # Loop over each input channel
            for j in range(self.n_channels):
                # Update the kernel by correlating the input with the output gradient
                self.K[i][j] -= lr * signal.correlate2d(self.X[j], output_gradient[i], mode='valid')

                # Compute the gradient with respect to the input by convolving the output gradient with the original kernel
                gradient_X[j] += signal.convolve2d(output_gradient[i], K_o[i][j], mode='full')

        return gradient_X  # Return the gradient with respect to the input of this layer





