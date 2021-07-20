import numpy as np
import time

class Timer:
    def __init__(self,gpu_model,cpu_model):
        self.gpu_model = gpu_model
        self.cpu_model = cpu_model
        
        
    def experiment(self,n_experiments,x,y,model_params):
        mode = model_params['mode']
        t_max = model_params['t_max']
        alpha = model_params['alpha']
        gamma = model_params['gamma']
        epsilon = model_params['epsilon']
        
        if mode=='mbgd':
            batch_size = model_params['batch_size']
            train_times = np.zeros(n_experiments)
            fit_times = np.zeros(n_experiments)
            scores = np.zeros(n_experiments)
            gpu_results = {}
            # gpu experiment
            if self.gpu_model is not None:
                for i in range(n_experiments):
                    start_time = time.time()
                    theta,J = self.gpu_model.train(mode=mode,t_max=t_max,alpha=alpha,gamma=gamma,batch_size=batch_size,epsilon=epsilon)
                    end_time = time.time()
                    train_times[i] = end_time-start_time
                    start_time = time.time()
                    score = self.gpu_model.score(x,y)
                    end_time = time.time()
                    fit_times[i] = end_time-start_time
                    scores[i] = score

                gpu_results = {'device':'gpu','n_experiments':n_experiments,'average_training_time':np.mean(train_times),
                              'average_fit_time':np.mean(fit_times),'average_score':scores.mean()}
            
            
            train_times = np.zeros(n_experiments)
            fit_times = np.zeros(n_experiments)
            scores = np.zeros(n_experiments)
            cpu_results = {}
            # cpu experiment
            if self.cpu_model is not None:
                for i in range(n_experiments):
                    start_time = time.time()
                    theta,J = self.cpu_model.train(mode=mode,t_max=t_max,alpha=alpha,gamma=gamma,batch_size=batch_size,epsilon=epsilon)
                    end_time = time.time()
                    train_times[i] = end_time-start_time
                    start_time = time.time()
                    score = self.cpu_model.score(x,y)
                    end_time = time.time()
                    fit_times[i] = end_time-start_time
                    scores[i] = score

                cpu_results = {'device':'cpu','n_experiments':n_experiments,'average_training_time':np.mean(train_times),
                              'average_fit_time':np.mean(fit_times),'average_score':scores.mean()}

            return gpu_results,cpu_results

        else:
            train_times = np.zeros(n_experiments)
            fit_times = np.zeros(n_experiments)
            scores = np.zeros(n_experiments)
            gpu_results = {}
            if self.gpu_model is not None:
                # gpu experiment
                for i in range(n_experiments):
                    start_time = time.time()
                    theta,J = self.gpu_model.train(mode=mode,t_max=t_max,alpha=alpha,gamma=gamma,epsilon=epsilon)
                    end_time = time.time()
                    train_times[i] = end_time-start_time
                    start_time = time.time()
                    score = self.gpu_model.score(x,y)
                    end_time = time.time()
                    fit_times[i] = end_time-start_time
                    scores[i] = score

                gpu_results = {'device':'gpu','n_experiments':n_experiments,'average_training_time':np.mean(train_times),
                              'average_fit_time':np.mean(fit_times),'average_score':scores.mean()}
            
            cpu_results = {}
            train_times = np.zeros(n_experiments)
            fit_times = np.zeros(n_experiments)
            scores = np.zeros(n_experiments)
            # cpu experiment
            if self.cpu_model is not None:
                for i in range(n_experiments):
                    start_time = time.time()
                    theta,J = self.cpu_model.train(mode=mode,t_max=t_max,alpha=alpha,gamma=gamma,epsilon=epsilon)
                    end_time = time.time()
                    train_times[i] = end_time-start_time
                    start_time = time.time()
                    score = self.cpu_model.score(x,y)
                    end_time = time.time()
                    fit_times[i] = end_time-start_time
                    scores[i] = score

                cpu_results = {'device':'cpu','n_experiments':n_experiments,'average_training_time':np.mean(train_times),
                              'average_fit_time':np.mean(fit_times),'average_score':scores.mean()}

            return gpu_results,cpu_results
        