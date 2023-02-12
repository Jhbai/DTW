import numpy as np

class Generator:
    def __init__(self, size1, size2, mu, sigma):
        self.paras = {'size1':size1,'size2':size2, 'mu':mu, 'sigma':sigma}

    def make(self):
        size1,size2,mu,sigma = self.paras['size1'], self.paras['size2'], self.paras['mu'], self.paras['sigma']
        start1, start2 = np.random.normal(size = (1, )), np.random.normal(size = (1, ))
        epsilon1, epsilon2 = np.random.normal(0, 1/size1, size = (size1, )), np.random.normal(0, 1/size2, size = (size2, ))
        X1 = start1 + np.cumsum(start1*mu*(1/size1) + start1*sigma*epsilon1)
        X2 = start2 + np.cumsum(start2*mu*(1/size2) + start2*sigma*epsilon2)
        return (X1, X2)
