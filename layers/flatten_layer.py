from layers.layer import Layer
import numpy as np

class Flatten(Layer):
    def __init__(self,input_shape,output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self,X):
        return np.reshape(X, self.output_shape)

    def backward(self,output_gradient, lr):
        return np.reshape(output_gradient, self.input_shape)





