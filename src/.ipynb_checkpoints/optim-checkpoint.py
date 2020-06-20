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
        
    def calc_grad(self,inputs,middles,outputs,corrects):
        self.grad_b2 = np.zeros((len(outputs),1))
        self.grad_w2 = np.zeros((len(outputs),len(middles)))
        self.grad_b1 = np.zeros((len(middles),1))
        self.grad_w1 = np.zeros((len(middles),len(inputs)))
        
        for a in range(self.data_num):
            # 出力層のバイアスの勾配の導出
            self.grad_b2[a] = (2/self.wave_len) * outputs * (1 - outputs) * (outputs - corrects)
            print(np.shape(self.grad_b2))
            # 出力層の重みの勾配の導出
            for l, x2 in enumerate(middles):
                self.grad_w2[a][l] = x2 * self.grad_b2
        
            # 中間層のバイアスの勾配の導出
            for m in range(len(middles)):
                self.grad_b1[m] = middles * (1 - middles) * np.sum(self.grad_b2 * self.model.layer2.weight()[:,m])
        
            # 中間層の重みの勾配の導出
            for n, x1 in enumerate(inputs):
                self.grad_w1[n] = x1 * self.grad_b1
            
        # データ数で除算する
        self.grad_b2 = self.grad_b2 / self.data_num
        self.grad_w2 = self.grad_w2 / self.data_num
        self.grad_b1 = self.grad_b1 / self.data_num
        self.grad_w1 = self.grad_w1 / self.data_num
        
    def update(self,inputs,middles,outputs,corrects,gamma=0.01):
        self.calc_grad(inputs,middles,outputs,corrects)
        
        new_bias2 = self.model.layer2.bias() - gamma * self.grad_b2
        self.model.layer2.bias(new_bias2)
        
        new_weight2 = self.model.layer2.weight() - gamma * self.grad_w2
        self.model.layer2.bias(new_weight2)
        
        new_bias1 = self.model.layer1.bias() - gamma * self.grad_b1
        self.model.layer2.bias(new_bias1)
        
        new_weight1 = self.model.layer1.weight() - gamma * self.grad_w1
        self.model.layer2.bias(new_weight1)
        