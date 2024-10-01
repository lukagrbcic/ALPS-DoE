import numpy as np
from scipy.stats import qmc


class DOE:
    def __init__(self, n_samples, method, lb, ub, seed=None):
        
        self.n_samples = n_samples
        self.method = method
        self.lb = lb
        self.ub = ub
        self.seed = seed
    
    def lhs(self):
        
        n = self.n_samples
        d = len(self.ub)
        if self.seed == None:
            sampler = qmc.LatinHypercube(d=d)
        else: 
            sampler = qmc.LatinHypercube(d=d, seed=self.seed)   
            
        X = qmc.scale(sampler.random(n=n), self.lb, self.ub)
      
        return X
    
    def sobol(self):
        
        n = self.n_samples
        d = len(self.ub)
        if self.seed == None:
            sampler = qmc.Sobol(d=d)
        else: 
            sampler = qmc.Sobol(d=d, seed=self.seed)   
            
        X = qmc.scale(sampler.random(n=n), self.lb, self.ub)
      
        return X
                
    
    def halton(self):
        
        n = self.n_samples
        d = len(self.ub)
        if self.seed == None:
            sampler = qmc.Halton(d=d)
        else: 
            sampler = qmc.Halton(d=d, seed=self.seed)   
            
        X = qmc.scale(sampler.random(n=n), self.lb, self.ub)
      
        return X
        
    

    
    
    def grid(self):
        
        d = len(self.ub)
        
        X = []
        for i in range(d):
            arr = np.linspace(self.lb[i], self.ub[i], self.n_samples)
            X.append(arr)
        
        X = np.transpose(np.array(X))
      
        return X
        
    
    def random(self):
        
        X = np.random.uniform(self.lb, self.ub, size=(self.n_samples, len(self.lb)))
        
        return X
    
    
    def generate_samples(self):
        
        
        if self.method == 'grid':
            
            return self.grid()
        
        if self.method == 'lhs':
            
            return self.lhs()
        
        if self.method == 'sobol':
            
            return self.sobol()
        
        if self.method == 'halton':
            
            return self.halton()
        
        if self.method == 'random':
            
            return self.random()
    
    

method = 'grid'   
n_samples = 1000
lb = np.array([0.2, 10, 15])
ub = np.array([1.3, 700, 28])     

experiment = DOE(n_samples, method, lb, ub)
print (experiment.generate_samples())
        
        
    
    
    