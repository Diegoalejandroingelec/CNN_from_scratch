from layer import Layer
from scipy import signal
import numpy as np

class Conv2D(Layer):
    def __init__(self,n_kernels, kernel_shape,input_shape):
        self.n_channels = input_shape[0]
        self.n_kernels = n_kernels
        self.input_shape = input_shape
        
        self.output_shape = (n_kernels,input_shape[1]-kernel_shape[0]+1,input_shape[2]-kernel_shape[1]+1)
        self.K = np.random.randn(n_kernels,self.n_channels,kernel_shape[0],kernel_shape[1])
        self.B = np.random.randn(*self.output_shape)

    def forward(self,X):
        self.X = X
        Y = self.B.copy()
        cross_corr = np.zeros(self.output_shape)

        for i in range(self.n_kernels):
            for j in range(self.n_channels):
                cross_corr[i] += signal.correlate2d(X[j], self.K[i][j], mode='valid')

            Y[i] += cross_corr[i]  

        return Y
    

    def backward(self,output_gradient, lr):
        K_o = self.K.copy()
        gradient_X = np.zeros(self.input_shape)

        for i in range(self.n_kernels):
            self.B[i]-=lr*output_gradient[i]
            for j in range(self.n_channels):
                self.K[i][j] -= lr*signal.correlate2d(self.X[j], output_gradient[i], mode='valid')
            
            gradient_X[j] += signal.convolve2d(output_gradient[i],K_o[i][j],mode='full')  

        return gradient_X




