from layers.layer import Layer
import numpy as np

class FullyConnected(Layer):
    def __init__(self,input_size, output_size,W=None,B=None):
        self.input_shape = input_size
        self.output_shape = output_size
        if(W==None):
            self.W = np.random.randn(output_size,input_size)
        else:
            self.W = W
        if(B==None):
            self.B = np.random.randn(output_size,1)
        else:
            self.B=B
    
    def set_weights(self,W,B):
        self.W=W
        self.B=B
    
    def forward(self,X):
        self.X = X
        return np.dot(self.W, self.X) + self.B

    def backward(self,back_propagated_gradient, lr):

        W = self.W.copy()
        self.W-=lr*np.dot(back_propagated_gradient,self.X.T)
        self.B-=lr*back_propagated_gradient
        return np.dot(W.T,back_propagated_gradient)