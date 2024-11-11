import numpy as np
from scipy.stats import qmc
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

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
    
    def greedyfp(self, scale=10):

        M = self.n_samples * scale  
        
        candidates = np.random.uniform(self.lb, self.ub, size=(M, len(self.lb)))    
        
        
        selected_indices = []
        available_indices = list(range(M))
    
        first_idx = np.random.choice(available_indices)
        selected_indices.append(first_idx)
        available_indices.remove(first_idx)
    
        for _ in range(self.n_samples - 1):
            selected_samples = candidates[selected_indices]  
    
            remaining_candidates = candidates[available_indices]  
    
            distances = cdist(remaining_candidates, selected_samples, metric='euclidean')
    
            min_distances = np.min(distances, axis=1)
    
            idx_in_remaining = np.argmax(min_distances)
            idx_max_min_dist = available_indices[idx_in_remaining]
    
            selected_indices.append(idx_max_min_dist)
            available_indices.remove(idx_max_min_dist)
    
        selected_samples = candidates[selected_indices]
        
        return selected_samples


    def bc(self, scale=10, maxCand=250):
        """
        Generate N samples in a d-dimensional space that are far from each other
        using the Best Candidate algorithm.
    
        Parameters:
        - N: int, number of samples to generate.
        - d: int, dimensionality of the space.
        - scale: int, scale factor used to compute the number of candidates.
        - maxCand: int, maximum number of candidate samples allowed.
        - lower_bounds: array-like of shape (d,), lower bounds for each dimension.
        - upper_bounds: array-like of shape (d,), upper bounds for each dimension.
    
        Returns:
        - selected_samples: numpy array of shape (N, d), the selected samples.
        """

        
        first_sample = np.random.uniform(self.lb, self.ub, size=(1, len(self.lb)))
        selected_samples = [first_sample]  
    
        for i in range(2, self.n_samples + 1):

            nCand = min(scale * i, maxCand)
    
            candidates = np.random.uniform(self.lb, self.ub, size=(nCand, len(self.lb)))

            selected_array = np.vstack(selected_samples)
    
            distances = cdist(candidates, selected_array, metric='euclidean')  # Shape: (nCand, i-1)
    
            min_distances = np.min(distances, axis=1)
    
            idx_best_candidate = np.argmax(min_distances)
            best_candidate_sample = candidates[idx_best_candidate]
            
            selected_samples.append(best_candidate_sample.reshape(1, len(self.lb)))
    
        selected_samples_array = np.vstack(selected_samples)
        return selected_samples_array
    
    
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
        
        if self.method == 'greedyFP':
            
            return self.greedyfp()
        
        if self.method == 'BC':
            
            return self.bc()
    

# method = 'BC'   
# n_samples = 1000
# lb = np.array([0.2, 10, 15])
# ub = np.array([1.3, 700, 28])     

# experiment = DOE(n_samples, method, lb, ub)
# samples = experiment.generate_samples()
        
        
    
    
    