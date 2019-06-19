from src.optimize import mixedlogit
from multiprocessing import Pool, cpu_count
from pathvalidate import sanitize_filename
import os
import pickle
import time

def datasetloader(dataset):
    return pickle.load(open("datasets/"+dataset,"rb"))

def resultsloader(picklelist):
    results = []
    
    for file in picklelist:
        with open("Results/"+file, "rb") as f:
            while True:
                try:
                    results.append(pickle.load(f))
                except EOFError:
                    break
    
    return results

class Mixedlogit:
        def __init__(self, dataset, dgp_name):
            self.coefficients = dataset['c']
            self.n_data = dataset['n']
            self.seed = dataset['seed']
            self.beta = dataset['beta']
            self.data = dataset['data']
            self.dgp = dgp_name
            self.result = {}
            
        def fit(self, drawtype, n_draws, start=0, end=0, c_0=False, save=False, method='BFGS', verbose=0):
            if end == 0:
                end = len(self.data)
            n_runs = end - start 
            print("Fitting %d datasets \n" % n_runs)
            
            if save != False:
                os.makedirs("Results", exist_ok=True)
                timestamp = time.strftime("%Y-%m-%d-%H%M%S")
                filename = "Results/"+sanitize_filename(save
                                                        +" - dgp "+self.dgp
                                                        +"- start "+str(start)+" end "+str(end-1)
                                                        +" - drawtype "+drawtype
                                                        + " - ndraws "+str(n_draws)
                                                        +"- timestamp "+timestamp)
                
                pickle_out = open(filename+".pickle","wb")

            for i in range(n_runs):
                print("Fitting model %d of %d \n" % ((i+1), n_runs))
                self.result[i] = mixedlogit(self.data[i], drawtype, n_draws, true_c = self.coefficients, dgp= self.dgp, dgp_i = (start+i), dgp_n=len(self.data), verbose = verbose)
    
                if save != False:
                    print("Saving run %d of %d \n" % ((i+1), n_runs))
                    pickle.dump(self.result[i], pickle_out)
                
            if save != False:
                pickle_out.close()
                print("All results saved to pickle: "+filename)
            
           
            