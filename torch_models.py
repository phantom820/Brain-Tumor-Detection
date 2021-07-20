import torch
import numpy as np

# torch implemenation that aims to use gpu
class LogisticRegression:
    def __init__(self,data,labels):
        if torch.cuda.is_available():
            self.x_train = torch.cuda.FloatTensor(data)
            self.y_train = torch.cuda.IntTensor(labels)
            self.n = len(data[0])
            self.N = len(data)
            self.theta = torch.randn((self.n)).cuda()
            
        else:
            raise Exception('Cuda enable gpu not unavailable')
          
    # sigmoid
    def sigmoid(self,x,theta):
        z = (x*theta).sum(axis=1)
        y = 1/(1+torch.exp(-z))
        return y
    
    # probability
    def probability(self,x):
        return self.sigmoid(x,self.theta)
    
    # BINARY CROSS ENTROPY cost function
    def cost_function(self,theta,gamma=0):
        x = self.x_train
        y = self.y_train
        h = self.sigmoid
        entropy = -y*torch.log(h(x,theta)+10e-5)-(1-y)*torch.log(1-h(x,theta)+1e-5)+0.5*gamma*torch.dot(theta,theta)
        J = torch.mean(entropy,axis=0).cpu().numpy()
        return J

    # mini batch gradient descent learning
    def mini_batch_gradient_descent(self,alpha = 0.01,epsilon=1e-3,gamma=1e-3,t_max=100,batch_size=128):  
        J = []
        t = 0
        N = self.N
        theta = torch.randn((self.n)).cuda()
        h = self.sigmoid
        for t in range(t_max):
            permutation = torch.randint(N,(1,batch_size)).flatten()
            x = self.x_train[permutation]
            y = self.y_train[permutation]
            error = h(x,theta)-y
            x_error = x*error[:,None]
            J.append(self.cost_function(theta,gamma))
            grad = torch.mean(x_error,axis=0)
            theta_prior = theta.clone()
            theta[0] = theta[0]-alpha*grad[0]
            theta[1:] = theta[1:]*(1-alpha*gamma/N)-alpha*grad[1:]
            residual = theta-theta_prior
            if torch.norm(residual)<epsilon:
                return theta,J
        return theta,J

         
    # batch gradient descent learning
    def batch_gradient_descent(self,alpha=0.01,epsilon=1e-3,gamma=1e-3,t_max=100):  
        J = []
        t = 0
        N = self.N
        theta = torch.randn((self.n)).cuda()
        h = self.sigmoid
        for t in range(t_max):
            error = h(self.x_train,theta)-self.y_train
            x_error = self.x_train*error[:,None]
            J.append(self.cost_function(theta,gamma))
            grad = torch.mean(x_error,axis=0)
            theta_prior = theta.clone()
            theta[0] = theta[0]-alpha*grad[0]
            theta[1:] = theta[1:]*(1-alpha*gamma/N)-alpha*grad[1:]
            residual = theta-theta_prior
            if torch.norm(residual)<epsilon:
                return theta,J
        return theta,J
    
    # train the model
    def train(self,mode,alpha=0.011,epsilon=1e-3,gamma=1e-3,batch_size=128,t_max=200,w=0.6,N=10):
        if mode=="bgd":
            self.theta , J = self.batch_gradient_descent(alpha=alpha,epsilon=epsilon,gamma=gamma,t_max=t_max)
            torch.cuda.empty_cache()
            return self.theta.cpu().numpy(),J
        elif mode=="mbgd":
            self.theta , J = self.mini_batch_gradient_descent(alpha=alpha,epsilon=epsilon,gamma=gamma,t_max=t_max,batch_size=batch_size)
            torch.cuda.empty_cache()
            return self.theta.cpu().numpy(),J
        
    # prediction
    def predict(self,X):
        theta = self.theta
        X = torch.cuda.FloatTensor(X)
        y_pred = self.sigmoid(X,theta)
        y_pred[y_pred>0.5] = 1
        y_pred[y_pred<=0.5] = 0
        return y_pred.cpu().numpy().astype(int)
    
    # evaluate the model
    def score(self,X,y):
        y_pred = self.predict(X)
        accuracy = np.count_nonzero(y_pred==y)/len(y_pred)
        return 100*accuracy
        