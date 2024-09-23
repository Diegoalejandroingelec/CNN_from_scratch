# Import necessary libraries for plotting and metrics
import matplotlib.pyplot as plt  # For plotting graphs and images
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  # For confusion matrix and display
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # For performance metrics
import numpy as np  # Numpy for numerical operations
import dill as pickle  # Dill for saving/loading models (alternative to pickle)

# Function to print performance metrics like accuracy, precision, recall, and F1 score
def print_performance_metrics(ground_truth_class, predicted_class):

    # Calculate accuracy
    accuracy = accuracy_score(ground_truth_class, predicted_class)

    # Calculate precision
    precision = precision_score(ground_truth_class, predicted_class)

    # Calculate recall
    recall = recall_score(ground_truth_class, predicted_class)

    # Calculate F1-score
    f1 = f1_score(ground_truth_class, predicted_class)

    # Print the calculated performance metrics
    print(f'Accuracy: {accuracy:.5f}')
    print(f'Precision: {precision:.5f}')
    print(f'Recall: {recall:.5f}')
    print(f'F1-score: {f1:.5f}')

# Function to plot training and testing loss over epochs
def plot_training_curves(training_loss, testing_loss):

    # Create an x-axis representing the number of epochs
    epochs_axis = range(1, len(training_loss) + 1)

    # Create a figure and axis object for plotting
    fig, ax1 = plt.subplots()

    # Plot the training loss on the primary y-axis
    ax1.plot(epochs_axis, training_loss, 'g-', label='Training Loss')
    ax1.set_xlabel('Epochs')  # Label for the x-axis
    ax1.set_ylabel('Training Loss', color='g')  # Label for the y-axis
    ax1.tick_params(axis='y', labelcolor='g')  # Color the ticks to match the graph

    # Create a second y-axis for testing loss
    ax2 = ax1.twinx()
    ax2.plot(epochs_axis, testing_loss, 'b-', label='Testing Loss')
    ax2.set_ylabel('Testing Loss', color='b')  # Label for the second y-axis
    ax2.tick_params(axis='y', labelcolor='b')

    # Add a title to the plot
    plt.title('Training and Testing Loss Over Epochs')

    # Add legends for the training and testing loss
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Display the plot
    plt.show()

# Function to plot the confusion matrix
def plot_confusion_matrix(ground_truth_class, predicted_class):

    # Generate the confusion matrix
    cm = confusion_matrix(ground_truth_class, predicted_class)
    
    # Create a display object for the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    
    # Plot the confusion matrix with a blue color map
    disp.plot(cmap=plt.cm.Blues)

    # Add a title to the confusion matrix plot
    plt.title('Confusion Matrix')
    
    # Show the plot
    plt.show()

# Function for data preprocessing, including normalization and one-hot encoding
def data_preprocessing(X, Y, max_data):

    # Select 'max_data' samples from the data with class 1
    y_index_1 = np.where(Y == 1)[0][0:max_data]
    
    # Select 'max_data' samples from the data with class 0
    y_index_0 = np.where(Y == 0)[0][0:max_data]

    # Combine the selected data points
    Y = Y[np.concatenate((y_index_0, y_index_1))]
    X = X[np.concatenate((y_index_0, y_index_1))]

    # Normalize the image data to range [0, 1]
    X = X.astype(float) / 255

    # Initialize an empty array for one-hot encoding of labels
    Y_one_hot = np.zeros((Y.size, len(set(Y))))
    
    # Perform one-hot encoding on the labels
    Y_one_hot[np.arange(Y.size), Y] = 1

    # Reshape the labels and images for model input
    reshaped_Y = Y_one_hot.reshape(Y_one_hot.shape[0], Y_one_hot.shape[1], 1)
    reshaped_X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
    
    return reshaped_X, reshaped_Y

# Function to plot a single image along with the predicted and ground truth labels
def plot_image(img, pred, gt):

    # Remove any singleton dimensions
    img = img.squeeze()

    # Plot the image using grayscale color map
    plt.imshow(img, cmap='gray')
    
    # Hide the axis for a cleaner plot
    plt.axis('off')
    
    # Add a title displaying the predicted and ground truth class
    plt.title(f'Predicted Class: {pred} Ground Truth: {gt}')
    
    # Show the plot
    plt.show()

# Function to save the model to a file using dill
def save_model(model, filename):

    # Open the file in write-binary mode and save the model
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    
    # Confirm that the model was saved successfully
    print(f"Model saved to {filename} successfully!")

# Function to load the model from a file using dill
def load_model(filename):

    # Open the file in read-binary mode and load the model
    with open(filename, 'rb') as file:
        instance = pickle.load(file)
    
    # Confirm that the model was loaded successfully
    print(f"Model loaded from {filename} successfully!")
    
    return instance
