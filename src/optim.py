import numpy as np

class GradientDescent(object):
    def __init__(self,model,data_num,wave_len):
        self.model = model
        self.data_num = data_num
        self.wave_len = wave_len
        
        self.grad_b2 = None
        self.grad_w2 = None
        self.grad_b1 = None
        self.grad_w1 = None
        
        self.grad_b1_temp = None
        
    def calc_grad(self,inputs,middles,outputs,corrects):
        self.grad_b2 = np.zeros((len(outputs),len(inputs[0])))
        self.grad_w2 = np.zeros((len(outputs),len(middles),len(inputs[0])))
        self.grad_b1 = np.zeros((len(middles),len(inputs[0])))
        self.grad_w1 = np.zeros((len(middles),len(inputs),len(inputs[0])))
        self.grad_b1_temp = np.zeros((len(middles),len(inputs)))
        
        for a in range(self.data_num):
            # 出力層のバイアスの勾配の導出
            self.grad_b2[:,a] = (2/self.wave_len) * outputs[:,a] * (1 - outputs[:,a]) * (outputs[:,a] - corrects[:,a])

            # 出力層の重みの勾配の導出
            for l in range(len(middles)):
                self.grad_w2[:,l,a] = middles[l,a] * self.grad_b2[:,a]
        
            # 中間層のバイアスの勾配の導出
            for m in range(len(inputs)):
                self.grad_b1_temp[:,m] = self.grad_b2[m,a] * self.model.layer2.weight[m,:]
            self.grad_b1[:,a] = middles[:,a] * (1 - middles[:,a]) * np.sum(self.grad_b1_temp,axis=1)
        
            # 中間層の重みの勾配の導出
            for n in range(len(inputs)):
                self.grad_w1[:,n,a] = inputs[n,a] * self.grad_b1[:,a]
        
        # データ数で除算する
        self.grad_b2 = np.sum(self.grad_b2,axis=1) / self.data_num
        self.grad_w2 = np.sum(self.grad_w2,axis=2) / self.data_num
        self.grad_b1 = np.sum(self.grad_b1,axis=1) / self.data_num
        self.grad_w1 = np.sum(self.grad_w1,axis=2) / self.data_num
        
        self.grad_b2 = np.reshape(self.grad_b2, [len(self.grad_b2),1])
        self.grad_b1 = np.reshape(self.grad_b1, [len(self.grad_b1),1])
        
    def update(self,inputs,middles,outputs,corrects,gamma=0.01):
        self.calc_grad(inputs,middles,outputs,corrects)
        
        self.model.layer2.bias = self.model.layer2.bias - gamma * self.grad_b2
        
        self.model.layer2.weight = self.model.layer2.weight - gamma * self.grad_w2

        self.model.layer1.bias = self.model.layer1.bias - gamma * self.grad_b1

        self.model.layer1.weight = self.model.layer1.weight - gamma * self.grad_w1

        