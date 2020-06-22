import numpy as np

class LeastSquaresError(object):
    def calc_loss(self,pred,ans):
        return np.mean((pred-ans)**2)