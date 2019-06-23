import numpy as np
from scipy.stats import norm
import ghalton

def makedraws(drawtype, n_r, n_q, n_k, verbose):
        print("Making %s draws of shape (%d, %d, %d)" % (drawtype, n_r, n_q, n_k))
        #generate draws
        if drawtype == 'pseudo':
            draws = np.random.randn(n_r, n_q, n_k);
        elif drawtype == 'halton':
            
#             #initialize sequencer with K dimensions
#             sequencer = ghalton.Halton(n_k)
            
#             #make the draws and discard the 10 inital values to prevent issues
#             draws = sequencer.get(n_q * n_r + 10)[10:]
            
#             #convert draws to standard variates by inverse cdf
#             draws = norm.ppf(draws)
            
#             #shape the draws into the right shape
#             draws = np.array(np.split(draws, n_r))
                
            #initialize sequencer with K dimensions
            sequencer = ghalton.Halton(n_k)
            
            #make the draws and discard the 10 inital values to prevent issues
            halton = np.array(sequencer.get(n_q * n_r + 10)[10:])

            #make the result array
            draws = np.zeros((n_r, n_q, n_k))

            #assign the right probablilities
            for q in range(n_q):
                for r in range(n_r):
                    draws[r, q, :] = halton[n_r * q + r,:]

            #convert to standard variates
            draws = norm.ppf(draws)
            
        else:
            raise Exception("Incorrect Drawtype: "+drawtype)    
            
        
        if(verbose > 2): print(draws)
        return draws
    