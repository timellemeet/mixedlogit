from src.optimize import mixedlogit
from multiprocessing import Pool, cpu_count
from pathvalidate import sanitize_filename
import os
import pickle
import time
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
from joblib import Parallel, delayed
import datetime


def datasetloader(dataset):
    return pickle.load(open("datasets/"+dataset,"rb"))

class Mixedlogit:
        def __init__(self, dataset, dgp_name):
            self.coefficients = dataset['c']
            self.n_data = dataset['n']
            self.seed = dataset['seed']
            self.beta = dataset['beta']
            self.data = dataset['data']
            self.dgp = dgp_name
            self.result = {}
            
        def fit(self, drawtype, n_draws, start=0, end=0, c_0=False, save=False, method='BFGS', verbose=0, parallel=False):
            if end == 0:
                end = len(self.data)
            n_runs = end - start 
            print("Fitting %d datasets \n" % n_runs)
            
            if save != False:
                
                timestamp = time.strftime("%Y-%m-%d-%H%M%S")
                path = []
                for s in save.split('/'):
                    path.append(sanitize_filename(s))
                path = '/'.join(path)
                
                dirs = ["Results"]
                if(len(path.split('/')) > 1):
                        dirs = np.concatenate((dirs,path.split('/')[:-1]))
                        
                os.makedirs('/'.join(dirs), exist_ok=True)
                
                filename = "Results/"+path+sanitize_filename(" - dgp "+self.dgp
                                                        +"- start "+str(start)+" end "+str(end-1)
                                                        +" - drawtype "+drawtype
                                                        + " - ndraws "+str(n_draws)
                                                        +"- timestamp "+timestamp)
                
                

            
            #execture code in single thread
            if parallel == False:
                if save != False: pickle_out = open(filename+".pickle","wb")
                    
                for i in range(n_runs):
                    print("Fitting model %d of %d \n" % ((i+1), n_runs))
                    self.result[i] = mixedlogit(self.data[start+i], drawtype, n_draws, c_true = self.coefficients, dgp= self.dgp, dgp_i = (start+i), dgp_n=len(self.data), verbose = verbose)

                    if save != False:
                        print("Saving run %d of %d \n" % ((i+1), n_runs))
                        pickle.dump(self.result[i], pickle_out)

                if save != False:
                    pickle_out.close()
                    print("All results saved to pickle: "+filename)
                    
            #execute code in parallel
            else:
                print("Executing code in parallel \n")
                def mxl(i):
                    try:
                        return mixedlogit(self.data[start+i], drawtype, n_draws, c_true = self.coefficients, dgp=self.dgp, dgp_i = (start+i), dgp_n=len(self.data), verbose = verbose)
                    except:
                        return False
                
                starttime = time.time()
                self.result = Parallel(n_jobs=-1, verbose=10)(delayed(mxl)(i) for i in range(n_runs))
                endtime = time.time()
                
                duration = str(datetime.timedelta(seconds=round(endtime-starttime)))
                avg = str(datetime.timedelta(seconds=round((endtime-starttime)/n_runs))) 
                print("Finished parallel excuting of %d jobs in %s, time per job %s\n" % (n_runs, duration, avg))
                
                if save != False:
                    print("Initializing save")
                    pickle_out = open(filename+".pickle","wb")
                    for i in range(n_runs):
                        if verbose > 2: print("Saving run %d of %d \n" % ((i+1), n_runs))
                        pickle.dump(self.result[i], pickle_out)
                    pickle_out.close()
                    print("All results saved to pickle: "+filename)
                
class Analyzer:
    def __init__(self, folders):
        results = []
        
        if isinstance(folders, str): folders = [folders]
        
        for folder in folders:
            picklelist = [f for f in listdir(folder) if isfile(join(folder, f))]

            for file in picklelist:
                with open(folder+"/"+file, "rb") as f:
                    while True:
                        try:
                            res = pickle.load(f)
                            
                            if isinstance(res, bool):
                                print("Bad result found, is a boolean of value: ")
                                print(res)
                            else:
                                res['folder'] = folder
                                res['file'] = file
                                results.append(res)
                        except EOFError:
                            break

        self.data = results
        
        
        self.df = pd.DataFrame(self.data)
        