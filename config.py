# Import necessary layer classes 
from layers.fully_connected_layer import FullyConnected  # Fully connected (dense) layer
from layers.flatten_layer import Flatten  # Flatten layer to reshape input
from layers.conv2D import Conv2D  # 2D Convolutional layer
from layers.sigmoid import Sigmoid  # Sigmoid activation function layer

# Define the architecture of the neural network model
model = [
    Conv2D(6, (3, 3), (1, 28, 28)),  # 2D Convolution layer with 6 filters of size 3x3, input shape (1, 28, 28)
    Sigmoid(),                       # Sigmoid activation function after the Conv2D layer
    Flatten((6, 26, 26), (26 * 26 * 6, 1)),  # Flatten the 6 channels of size 26x26 into a single vector (for fully connected layers)
    FullyConnected(26 * 26 * 6, 256),  # Fully connected layer with input size matching flattened vector, and 256 output neurons
    Sigmoid(),                         # Sigmoid activation for non-linearity
    FullyConnected(256, 2),            # Fully connected layer with 256 inputs and 2 output neurons (for binary classification)
    Sigmoid(),                         # Sigmoid activation for the final layer to output probabilities (or scores between 0 and 1)
]

# Set the number of epochs to train the model
epochs = 50  # Train for 50 epochs

# Set the learning rate for gradient descent optimization
lr = 0.1  # Learning rate for backpropagation

# Flags to determine the mode of operation
train_and_test = False  # Set to True if the model should both train and test
train_only = False      # Set to True if the model should only train
test_only = True        # Set to True if the model should only test (skip training)
