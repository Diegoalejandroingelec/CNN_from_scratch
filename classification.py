# Import necessary modules and functions for loss calculation, model building, and utilities
from loss_functions.losses import binary_cross_entropy, binary_cross_entropy_prime  # Loss functions
from neural_network import NeuralNetwork  # NeuralNetwork class for building and training the model
from keras.datasets import mnist  # MNIST dataset from Keras
import numpy as np  # Numpy for numerical operations
from utils import (print_performance_metrics, plot_training_curves, plot_confusion_matrix, 
                   data_preprocessing, plot_image, save_model, load_model)  # Utility functions
from config import model, epochs, lr, train_and_test, train_only, test_only  # Configurations

# Load MNIST dataset and split into training and testing sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the training data (reshape, normalize, etc.)
x_train, y_train = data_preprocessing(x_train, y_train, 1000)  # Preprocess with 1000 samples for training
print(x_train.shape)  # Print the shape of the preprocessed training data
print(y_train.shape)  # Print the shape of the preprocessed training labels

# Preprocess the testing data (reshape, normalize, etc.)
x_test, y_test = data_preprocessing(x_test, y_test, 1000)  # Preprocess with 1000 samples for testing
print(x_test.shape)  # Print the shape of the preprocessed testing data
print(y_test.shape)  # Print the shape of the preprocessed testing labels

# ######################### TRAINING #########################
if train_only or train_and_test:  # If either train_only or train_and_test flag is True, proceed with training
    
    # Initialize the Neural Network with the model, loss functions, and learning rate
    CNN = NeuralNetwork(model, binary_cross_entropy, binary_cross_entropy_prime, lr)
    
    # Train the network and retrieve training and testing loss
    training_loss, testing_loss = CNN.fit(x_train, y_train, x_test, y_test, epochs, 1)
    
    # Plot the training and testing loss curves over epochs
    plot_training_curves(training_loss, testing_loss)
    
    # Save the trained model to a file
    save_model(CNN, "cnn.pkl")

# #################### TESTING #########################
if test_only or train_and_test:  # If either test_only or train_and_test flag is True, proceed with testing
    
    # Load the trained model from the saved file
    CNN = load_model("cnn.pkl")
    
    # Print a summary of the network architecture
    CNN.summary()

    # Initialize lists to store predicted and ground truth class labels
    predicted_class = []
    ground_truth_class = []

    # Counters for correctly and incorrectly predicted classes (1 and 0)
    count_incorrect_1 = 0
    count_incorrect_0 = 0
    count_correct_1 = 0
    count_correct_0 = 0

    # Loop through the test images and their corresponding ground truth labels
    for image, ground_truth in zip(x_test, y_test):
        # Predict the class for the current test image
        prediction = CNN.predict(image)
        pred_index = np.argmax(prediction)  # Get the predicted class (index with highest value)
        gt_index = np.argmax(ground_truth)  # Get the ground truth class (index with highest value)
        
        # Append the predicted and ground truth class to the respective lists
        predicted_class.append(pred_index)
        ground_truth_class.append(gt_index)

        # Plot incorrectly predicted images (with label 1) up to 3 samples
        if gt_index != pred_index and gt_index == 1 and count_incorrect_1 < 3:
            plot_image(image, pred_index, gt_index)  # Display the image along with predicted and actual labels
            count_incorrect_1 += 1  # Increment the count of incorrect class 1 predictions

        # Plot incorrectly predicted images (with label 0) up to 3 samples
        if gt_index != pred_index and gt_index == 0 and count_incorrect_0 < 3:
            plot_image(image, pred_index, gt_index)  # Display the image along with predicted and actual labels
            count_incorrect_0 += 1  # Increment the count of incorrect class 0 predictions

        # Plot correctly predicted images (with label 1) up to 3 samples
        if gt_index == pred_index and gt_index == 1 and count_correct_1 < 3:
            plot_image(image, pred_index, gt_index)  # Display the image along with predicted and actual labels
            count_correct_1 += 1  # Increment the count of correct class 1 predictions

        # Plot correctly predicted images (with label 0) up to 3 samples
        if gt_index == pred_index and gt_index == 0 and count_correct_0 < 3:
            plot_image(image, pred_index, gt_index)  # Display the image along with predicted and actual labels
            count_correct_0 += 1  # Increment the count of correct class 0 predictions

    # Plot the confusion matrix to visualize the classification results
    plot_confusion_matrix(ground_truth_class, predicted_class)
    
    # Print performance metrics (e.g., accuracy, precision, recall)
    print_performance_metrics(ground_truth_class, predicted_class)
