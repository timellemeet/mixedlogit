from src.optimize import mixedlogit

class Mixedlogit:
        def __init__(self, dataset):
            self.coefficients = dataset['c']
            self.n_data = dataset['n']
            self.seed = dataset['seed']
            self.beta = dataset['beta']
            self.data = dataset['data']
            self.result = {}
            
        def fit(self, drawtype, n_draws, subset=1, c_0=False, method='BFGS', verbose=0):
            n_fit = int(len(self.data) * subset)
            print("Fitting %d datasets \n" % n_fit)
            
            
            for i in range(n_fit):
                print("Fitting model %d \n" % (i+1))
                self.result[i] = mixedlogit(self.data[i], drawtype, n_draws, true_c = self.coefficients, verbose = verbose)
                
                
            
           
            