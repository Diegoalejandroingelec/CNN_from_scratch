# Base class for layers in a neural network
class Layer():
    def __init__(self):
        """
        Initialize the base Layer class. This class acts as a blueprint for other layers.
        """
        self.input = None  # Placeholder to store the input to the layer during the forward pass
        self.output = None  # Placeholder to store the output from the layer after the forward pass

    # Method for the forward pass through the layer (to be implemented in subclasses)
    def forward(self, input):
        """
        Forward pass method that takes input and produces output.
        To be overridden by subclasses (e.g., FullyConnected, Conv2D, etc.).
        
        Args:
            input: Input to the layer (e.g., data or output from the previous layer).
        """
        pass

    # Method for the backward pass through the layer (to be implemented in subclasses)
    def backward(self, output_gradient, lr):
        """
        Backward pass method that takes the gradient of the loss with respect to the output
        and computes the gradient of the loss with respect to the input. Updates layer parameters if necessary.
        To be overridden by subclasses.
        
        Args:
            output_gradient: Gradient of the loss with respect to the output of the layer.
            learning_rate: The learning rate used to update the layer's parameters (if applicable).
        """
        pass
