import numpy as np
import pandas as pd
import datetime
import time
from scipy.optimize import minimize
from numba import jit, prange    
from IPython.display import display
from sklearn.metrics import mean_squared_error
from math import sqrt
pd.options.display.float_format = '{:,.3f}'.format

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

@jit(nopython=True, parallel=True)
def likelihood(c, data, draws, verbose=False):
    #print("evaluation likelihood")
    n_r = draws.shape[0]
    n_q = draws.shape[1]
#   n_k = draws.shape[2]
    
    #matrix with all simulations
    simulations = np.zeros((n_q, n_r))
    
    #iterate over households
    for q in prange(n_q):
        rows = data[np.where(data[:,0] == q+1)]
        n_rows = len(rows)
        
        #iterate over draws per household
        for r in prange(n_r):
            probabilities= np.zeros(n_rows)
            
            #iterate over oberservations per househould
            for t in prange(n_rows):
                choices = np.zeros(4)
                
                #itetate over probability of choices per observation
                for j in prange(4):
                    utility = 0; #start with alpha
                    if j < 3: utility = c[j]
                    x = [rows[t][1+j], rows[t][5+j], rows[t][9+j]]
                    mu =  c[3:6]
                    sigma = np.exp(c[6:])

                    for l in prange(3):
                          utility += mu[l] * x[l] + sigma[l] * draws[r][q][l] * x[l]

                    choices[j] = np.exp(utility)
            
                probabilities[t] = choices[int(rows[t][13])] / np.sum(choices)
                
            simulations[q,r] = np.exp(np.log(probabilities).sum())
            
    estimates = np.zeros(n_q)
    for q in prange(n_q):
        estimates[q] = np.sum(simulations[q,:]) / n_r #.mean()
    
    res = -np.log(estimates).sum()
    
    if verbose == 2: print(res)
        
    return res


def mixedlogit(data, drawtype, n_draws, true_c, c_0=False, method='BFGS', verbose=0):
        n_q = len(data.id.unique())

    #     coefficients = [#alpha heinz41
    #                     #alpha heinz32 
    #                     #alpha heinz28
    #                     #mu    display
    #                     #mu    feat
    #                     #mu    price
    #                     #sigma dispay 
    #                     #sigma feat 
    #                     #sigma price 
    #                     ]

        #generate inital values if neccesary
        if isinstance(c_0, bool) and c_0 == False:
                #genereate random starting coefficients
                c_0 = np.random.rand(9)
        elif len(c_0) != 9:
            raise Exception("Incorrect initial coefficients")

        #generate draws
        if drawtype == 'pseudo':
            draws = np.random.randn(n_draws, n_q, 3); 
        else:
             raise Exception("Incorrect Drawtype: "+drawtype)


        
        iterations = pd.DataFrame(columns=['a h41', 'a h32', 'a h28', 'm disp', 'm feat', 'm price', 's disp', 's feat', 's price', 'MAPE', 'RMSE'])
        
        def logging(xk):
            if(verbose >0):
                #calulating values
                sigma = np.exp(xk[6:])
                c_i = np.concatenate((xk[:6], sigma))
                rmse = sqrt(mean_squared_error(true_c, c_i))
                mape = mean_absolute_percentage_error(true_c, c_i)
                                     
                #loggint output
                iterations.loc[len(iterations)] = np.concatenate((c_i, mape, rmse), axis=None)
                display(iterations.tail(1))
               

        start = time.time()
        try:
            res =  minimize(likelihood, c_0, args=(data.drop(columns='choice').values, draws, verbose), method=method, callback=logging)
        except:
            res =  minimize(likelihood, c_0, args=(data.drop(columns='choice').values, draws, verbose), method=method, callback=logging)
        end = time.time()
        duration = end-start 
        
        res['duration'] = duration
        res['x'][6:] = np.exp(res['x'][6:])
        if verbose > 0: 
            print("Optimization done, time elapsed: %s" % str(datetime.timedelta(seconds=round(duration))))
            display(res)
            print('\n')

        res['iterations'] = iterations
        return res