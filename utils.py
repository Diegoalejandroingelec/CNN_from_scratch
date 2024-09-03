import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import dill as pickle

def print_performance_metrics(ground_truth_class,predicted_class):

    # Calculate accuracy
    accuracy = accuracy_score(ground_truth_class, predicted_class)

    # Calculate precision
    precision = precision_score(ground_truth_class, predicted_class)

    # Calculate recall
    recall = recall_score(ground_truth_class, predicted_class)

    # Calculate F1-score
    f1 = f1_score(ground_truth_class, predicted_class)

    # Print the results
    print(f'Accuracy: {accuracy:.5f}')
    print(f'Precision: {precision:.5f}')
    print(f'Recall: {recall:.5f}')
    print(f'F1-score: {f1:.5f}')

def plot_training_curves(training_loss,testing_loss):
    # Number of epochs (assuming each element in training_loss and testing_loss corresponds to one epoch)
    epochs_axis = range(1, len(training_loss) + 1)

    # Create a figure and a set of subplots
    fig, ax1 = plt.subplots()

    # Plot training loss on the first y-axis
    ax1.plot(epochs_axis, training_loss, 'g-', label='Training Loss')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Training Loss', color='g')
    ax1.tick_params(axis='y', labelcolor='g')

    # Create a second y-axis sharing the same x-axis
    ax2 = ax1.twinx()
    ax2.plot(epochs_axis, testing_loss, 'b-', label='Testing Loss')
    ax2.set_ylabel('Testing Loss', color='b')
    ax2.tick_params(axis='y', labelcolor='b')

    # Add a title
    plt.title('Training and Testing Loss Over Epochs')

    # Add legends for both axes
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Display the plot
    plt.show()


def plot_confusion_matrix(ground_truth_class,predicted_class):

    cm = confusion_matrix(ground_truth_class, predicted_class)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)

    # Adding title and labels
    plt.title('Confusion Matrix')
    plt.show()

def data_preprocessing(X,Y,max_data):

    y_index_1=np.where(Y==1)[0][0:max_data]
    y_index_0=np.where(Y==0)[0][0:max_data]

    Y = Y[np.concatenate((y_index_0,y_index_1))]
    X = X[np.concatenate((y_index_0,y_index_1))]

    X = X.astype(float)/255
    #flattened_X = X.reshape(X.shape[0], X.shape[1] * X.shape[1],1)/np.max(X)
    
    
    Y_one_hot = np.zeros((Y.size, len(set(Y))))
    Y_one_hot[np.arange(Y.size), Y] = 1

    
    reshaped_Y = Y_one_hot.reshape(Y_one_hot.shape[0],Y_one_hot.shape[1],1)
    reshaped_X = X.reshape(X.shape[0],1,X.shape[1],X.shape[2])
    return reshaped_X,reshaped_Y

def plot_image(img,pred,gt):

    img = img.squeeze()

    plt.imshow(img, cmap='gray')  # Use 'gray' colormap for grayscale images
    plt.axis('off') 
    plt.title(f'Predicted Class:{pred} Groud Truth:{gt}')  # Add a title to the image
    plt.show()

def save_model(model, filename):

    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved to {filename} successfully!")


def load_model(filename):

    with open(filename, 'rb') as file:
        instance = pickle.load(file)
    print(f"Model loaded from {filename} successfully!")
    return instance