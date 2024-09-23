import numpy as np  # Import numpy for efficient numerical operations

# Function to calculate Mean Squared Error (MSE) loss
def mse(Y, Y_p):
    """
    Compute the Mean Squared Error (MSE) between the ground truth labels Y and predictions Y_p.
    
    Args:
        Y: Ground truth labels (numpy array).
        Y_p: Predicted labels (numpy array).
    
    Returns:
        Mean of the squared differences between Y and Y_p.
    """
    return np.mean(np.power((Y - Y_p), 2))  # Calculate the mean of the squared differences

# Function to compute the derivative (gradient) of the MSE loss with respect to predictions
def mse_prime(Y, Y_p):
    """
    Compute the derivative of Mean Squared Error (MSE) loss with respect to the predictions.
    
    Args:
        Y: Ground truth labels (numpy array).
        Y_p: Predicted labels (numpy array).
    
    Returns:
        The gradient of the MSE loss.
    """
    return 2 * (Y - Y_p) / np.size(Y)  # Derivative of MSE with respect to predictions

# Function to calculate Binary Cross-Entropy loss
def binary_cross_entropy(Y, Y_p):
    """
    Compute the Binary Cross-Entropy loss between ground truth labels Y and predictions Y_p.
    
    Args:
        Y: Ground truth labels (numpy array).
        Y_p: Predicted labels (numpy array).
    
    Returns:
        The binary cross-entropy loss value.
    """
    return -np.mean(Y_p * np.log(Y) + (1 - Y_p) * np.log(1 - Y))  # Binary Cross-Entropy formula

# Function to compute the derivative (gradient) of the Binary Cross-Entropy loss
def binary_cross_entropy_prime(Y, Y_p):
    """
    Compute the derivative of the Binary Cross-Entropy loss with respect to the predictions.
    
    Args:
        Y: Ground truth labels (numpy array).
        Y_p: Predicted labels (numpy array).
    
    Returns:
        The gradient of the Binary Cross-Entropy loss.
    """
    return (1 / np.size(Y)) * ((1 - Y_p) / (1 - Y) - (Y_p / Y))  # Derivative of Binary Cross-Entropy loss
