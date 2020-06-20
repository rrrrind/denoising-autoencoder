import numpy as np

class LeastSquaresError(object):
    def calc_loss(self,pred,ans):
        return np.mean(np.linalg.norm(pred-ans, ord=2, axis=0))