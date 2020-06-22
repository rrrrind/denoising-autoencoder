import numpy as np

class Wave(object):
    def __init__(self,data_num,wave_len=200,noise_power=0.05):
        self.wave_len = wave_len
        self.data_num = data_num
        self.noise_power = noise_power
        
        self.interval = np.arange(0,1,1/self.wave_len)
 
    def generate(self):
        # オリジナルの波形
        xorg = np.zeros((self.wave_len,self.data_num))
        # ノイズ込みの波形
        xnois = np.zeros((self.wave_len,self.data_num))
        
        for i in range(self.data_num):
            Ampl = 0.3 * np.random.rand()
            freq = 20 * np.random.rand()
            norm = 0.1 * np.random.rand() + 0.5
            
            xorg[:,i] = Ampl * np.sin(freq * self.interval) + norm
            xnois[:,i] = xorg[:,i] + self.noise_power * np.random.randn(self.wave_len)
        
        return xorg, xnois