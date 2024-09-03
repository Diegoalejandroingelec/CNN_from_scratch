from losses import binary_cross_entropy, binary_cross_entropy_prime
from neural_network import NeuralNetwork
from keras.datasets import mnist
import  numpy as np
from utils import print_performance_metrics,plot_training_curves,plot_confusion_matrix,data_preprocessing,plot_image,save_model,load_model
from config  import model, epochs, lr, train_and_test, train_only, test_only




(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train=data_preprocessing(x_train,y_train,1000)
print(x_train.shape)
print(y_train.shape)

x_test, y_test=data_preprocessing(x_test,y_test,1000)
print(x_test.shape)
print(y_test.shape)


if(train_only or train_and_test):


    CNN = NeuralNetwork(model,binary_cross_entropy,binary_cross_entropy_prime,lr)


    training_loss,testing_loss=CNN.fit(x_train,y_train,x_test,y_test,epochs,1)


    plot_training_curves(training_loss,testing_loss)

    save_model(CNN,"cnn.pkl")


#################### TESTING #########################
if(test_only or train_and_test):
    
    CNN=load_model("cnn.pkl")
    W,B=CNN.get_weights()
    # print(f"Weights shape {W}")
    # print(f"Biases shape {B}")
    predicted_class=[]
    ground_truth_class =[]
    count_incorrect_1 = 0
    count_incorrect_0 = 0
    count_correct_1= 0
    count_correct_0 = 0
    for image, ground_truth in  zip(x_test, y_test):
        prediction=CNN.predict(image)
        pred_index = np.argmax(prediction)
        gt_index = np.argmax(ground_truth)
        predicted_class.append(pred_index)
        ground_truth_class.append(gt_index)

        if (gt_index!=pred_index and gt_index==1 and count_incorrect_1<3):
            plot_image(image,pred_index,gt_index)
            count_incorrect_1+=1

        if (gt_index!=pred_index and gt_index==0 and count_incorrect_0<3):
            plot_image(image,pred_index,gt_index)
            count_incorrect_0+=1


        if(gt_index==pred_index and gt_index==1 and count_correct_1<3):
            plot_image(image,pred_index,gt_index)
            count_correct_1+=1

        if(gt_index==pred_index and gt_index==0 and count_correct_0<3):
            plot_image(image,pred_index,gt_index)
            count_correct_0+=1



    plot_confusion_matrix(ground_truth_class,predicted_class)
    print_performance_metrics(ground_truth_class,predicted_class)

