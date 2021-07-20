import numpy as np
class DataPreprocessor:
    def __init__(self,data):
        self.x = data
        self.n = len(data)
        
    def add_bias(self,x):
        bias=np.ones((x.shape[0],1))
        x=np.hstack((bias,x))
        return x
    
    def normalization_parameters(self):
        min_=np.min(self.x,axis=0)
        max_=np.max(self.x,axis=0)
        return min_,max_
        
        
    def standardization_parammeters(self):
        mu =np.mean(self.x,axis=0)
        std =np.std(self.x,axis=0)
        return mu,std
    
    def normalize(self,x):
        min_,max_ = self.normalization_parameters()
        x_n=(x-min_)/(max_-min_) #avoid zero division
        return x_n
    def standardize(self,x):
        mu , std = self.standardization_parammeters()
        x_s = (x-mu)/(std+1e-4)
        return x_s
    
    def fit_transform(self,x,mode):
        if mode=='n':
            x_n = self.normalize(x)
            x_n_aug = self.add_bias(x_n)
        elif mode=='s':
            x_n = self.standardize(x)
            x_n_aug = self.add_bias(x_n)
        return x_n_aug
