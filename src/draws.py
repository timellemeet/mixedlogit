import numpy as np

def makedraws(drawtype, n_r, n_q, n_k):
        print("Making %s draws of shape (%d, %d, %d)" % (drawtype, n_r, n_q, n_k))
        #generate draws
        if drawtype == 'pseudo':
            draws = np.random.randn(n_r, n_q, n_k);
        elif drawtype == 'halton':
            raise Exception("halton selected")
        else:
             raise Exception("Incorrect Drawtype: "+drawtype)
                
        return draws
    