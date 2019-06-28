import numpy as np
from scipy.stats import norm
import ghalton
import sobol_seq



def makedraws(drawtype, n_r, n_q, n_k, verbose):
        #define a function to transform made draws in to appropriate standard normals
        def transform(madedraws):
            #make the result array
            draws = np.zeros((n_r, n_q, n_k))

            #assign the right probablilities
            for q in range(n_q):
                 for r in range(n_r):
                    draws[r, q, :] = madedraws[n_r * q + r,:] 
            
            #convert to standard variates
            return norm.ppf(draws)
            
        print("Making %s draws of shape (%d, %d, %d)" % (drawtype, n_r, n_q, n_k))
        #generate draws
        if drawtype == 'pseudo':
            draws = np.random.randn(n_r, n_q, n_k);
        elif drawtype == 'halton':
            #initialize sequencer with K dimensions
            sequencer = ghalton.Halton(n_k)
            
            #make the draws and discard the 10 inital values to prevent issues
            halton = np.array(sequencer.get(n_q * n_r + 10)[10:])

            return transform(halton)
        
        elif drawtype == 'golden':
             #generalized definition of the golden ration
            def r2draws(n, d, seed=0.5): 
                def phi(d): 
                    x=2
                    for i in range(100): 
                        x = pow(1+x,1/(d+1)) 
                    return x

                g = phi(d) 
                alpha = np.zeros(d) 
                for j in range(d): 
                    alpha[j] = pow(1/g,j+1)  %1 # mod 1 

                r2draws = np.zeros((n, d))
                for i in range(n): 
                      r2draws[i] = (seed + alpha*(i+1)) % 1 # mod 1 

                return r2draws

            #make the draws and discard the 10 inital values to prevent issues
            goldenr2 = np.array(r2draws(n_q * n_r+10, n_k))[10:]

            return transform(goldenr2)
            
        elif drawtype == 'sobol':
            sobol = sobol_seq.i4_sobol_generate(n_k, n_r*n_q+10)[10:]
            
            return transform(sobol)
            
        else:
            raise Exception("Incorrect Drawtype: "+drawtype)    
            
        
        if(verbose > 2): print(draws)
        return draws
    