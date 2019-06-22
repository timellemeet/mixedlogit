import numpy as np

def makedraws(drawtype, n_r, n_q, n_k):
        print("making draws")
        #generate draws
        if drawtype == 'pseudo':
            draws = np.random.randn(n_r, n_q, n_k); 
        else:
             raise Exception("Incorrect Drawtype: "+drawtype)
                
        return draws
    