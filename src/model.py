import numpy as np
from layer import FullyConnectedLayer
import activation

class DAEModel(object):
    def __init__(self,layers=[200,120,200]):
        self.layers = layers
        
        self.layer1 = FullyConnectedLayer(self.layers[0], self.layers[1])
        self.layer2 = FullyConnectedLayer(self.layers[1], self.layers[2])
        
    def forward(self,inputs):
        c1 = self.layer1.forward(inputs)
        x2 = activation.sigmoid(c1)
        c2 = self.layer2.forward(x2)
        outputs = activation.sigmoid(c2)
        return x2, outputs