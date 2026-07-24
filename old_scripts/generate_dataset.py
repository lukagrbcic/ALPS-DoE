import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from alps_doe import DOE
import sys

sys.path.insert(0, '../InverseBench/src/')

from benchmarks import *

bench = 'inconel' #(random forests)
name = 'inconel_benchmark'
model = load_model(name).load()
f = benchmark_functions(name, model)

def nmax_ae(y_true, y_pred):
    
    nmaxae = np.max(np.abs(y_true - y_pred), axis=0)/np.max(np.abs(y_true - np.mean(y_true, axis=0)))
    
    return np.mean(nmaxae)
   


methods = ['random', 'lhs', 'halton', 'BC', 'greedyFP']

# methods = ['random', 'lhs', 'halton']


for m in methods:
    method = m 
    n_samples = 1000
    lb = np.array([0, 100, 10])
    ub = np.array([1, 1000, 100])     
    
    # lb = np.array([0.2, 100, 15])
    # ub = np.array([1.3, 700, 28])     
    
    experiment = DOE(n_samples, method, lb, ub)

    t = np.linspace(0, 1, 200)
    
    
    def synthetic_function(x):
        
    
        y = np.array([0.5 * (np.sin(2 * np.pi * (x[i][0] * t) + x[i][1] / 100 + x[i][2] / 20) + 1) for i in range(len(x))])

        return y
    
    # def synthetic_function(x):
        
    #     y = f.evaluate(x)
    
    #     return y

    r2_mean = []
    rmse_mean = []
    mape_mean = []
    nmaxae_mean = []

    
    for i in range(10):
    
        x_ = experiment.generate_samples()
        
        y_ = synthetic_function(x_)
        
        
        X_train, X_test, y_train, y_test = train_test_split(x_, y_, test_size=0.2)
        
        model = RandomForestRegressor().fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = mean_absolute_percentage_error(y_test, y_pred)
        nmaxae = nmax_ae(y_test, y_pred)

        r2_mean.append(r2)
        rmse_mean.append(rmse)
        mape_mean.append(mape)
        nmaxae_mean.append(nmaxae)
    
    print (method)
    print (f'R2: {np.mean(r2_mean):.3f}, {np.std(r2_mean):.3f}')
    print (f'RMSE: {np.mean(rmse_mean):.3f}, {np.std(rmse_mean):.3f}')
    print (f'NMAE: {np.mean(nmaxae_mean):.3f}, {np.std(nmaxae_mean):.3f}')


