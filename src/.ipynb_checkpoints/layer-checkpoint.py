import numpy as np

class FullyConnectedLayer(object):
    def __init__(self, input_dim, output_dim):
        self.__weight = 0.01 * np.random.rand(output_dim,input_dim)
        self.__bias = 0.01 * np.random.rand(output_dim,1)
    
    @property
    def weight(self):
        return self.__weight
    
    @weight.setter
    def weight(self, new_weight):
        self.__weight = new_weight
        
    @property
    def bias(self):
        return self.__bias
    
    @bias.setter
    def bias(self, new_bias):
        self.__bias = new_bias
       
        
    def forward(self, x):
        return np.dot(self.__weight, x) + self.__bias

