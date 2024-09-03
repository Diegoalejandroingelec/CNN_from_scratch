import numpy as np

def mse(Y,Y_p):
    return np.mean((np.power((Y-Y_p),2)))

def mse_prime(Y,Y_p):
    return 2*(Y-Y_p)/np.size(Y)


def binary_cross_entropy(Y,Y_p):
    return -np.mean(Y_p*np.log(Y)+(1-Y_p)*np.log(1-Y))

def binary_cross_entropy_prime(Y,Y_p):
    return (1/np.size(Y))*((1-Y_p)/(1-Y)-(Y_p/Y))