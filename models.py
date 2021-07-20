import numpy as np

# numpy implemenation that uses cpu
class LogisticRegression:
    def __init__(self,data,labels):
        self.x_train = data.copy()
        self.y_train = labels.copy()
        self.n = len(data[0])
        self.N = len(data)
        self.theta = np.random.randn((self.n))
        
    # sigmoid
    def sigmoid(self,x,theta):
        z = (x*theta).sum(axis=1)
        y = 1/(1+np.exp(-z))
        return y
    
    # probability
    def probability(self,x):
        return self.sigmoid(x,self.theta)
    
    # BINARY CROSS ENTROPY cost function
    def cost_function(self,theta,gamma=0):
        x = self.x_train
        y = self.y_train
        h = self.sigmoid
        entropy = -y*np.log(h(x,theta)+10e-5)-(1-y)*np.log(1-h(x,theta)+1e-5)+0.5*gamma*np.dot(theta,theta)
        J = np.mean(entropy,axis=0)
        return J

    # mini batch gradient descent learning
    def mini_batch_gradient_descent(self,alpha = 0.01,epsilon=1e-3,gamma=1e-3,t_max=100,batch_size=128):  
        J = []
        t = 0
        N = self.N
        theta = np.random.randn((self.n))
        h = self.sigmoid
        for t in range(t_max):
            permutation = np.random.randint(0,N,batch_size).flatten()
            x = self.x_train[permutation]
            y = self.y_train[permutation]
            error = h(x,theta)-y
            x_error = x*error[:,None]
            J.append(self.cost_function(theta,gamma))
            grad = np.mean(x_error,axis=0)
            theta_prior = theta.copy()
            theta[0] = theta[0]-alpha*grad[0]
            theta[1:] = theta[1:]*(1-alpha*gamma/N)-alpha*grad[1:]
            residual = theta-theta_prior
            if np.linalg.norm(residual)<epsilon:
                return theta,J
        return theta,J

    # batch gradient descent learning
    def batch_gradient_descent(self,alpha=0.01,epsilon=1e-3,gamma=1e-3,t_max=100):  
        J = []
        t = 0
        N = self.N
        theta = np.random.randn((self.n))
        h = self.sigmoid
        for t in range(t_max):
            error = h(self.x_train,theta)-self.y_train
            x_error = self.x_train*error[:,None]
            J.append(self.cost_function(theta,gamma))
            grad = np.mean(x_error,axis=0)
            theta_prior = theta.copy()
            theta[0] = theta[0]-alpha*grad[0]
            theta[1:] = theta[1:]*(1-alpha*gamma/N)-alpha*grad[1:]
            residual = theta-theta_prior
            if np.linalg.norm(residual)<epsilon:
                return theta,J
        return theta,J
    
    # train the model
    def train(self,mode,alpha=0.011,epsilon=1e-3,gamma=1e-3,batch_size=128,t_max=200,w=0.6,N=10):
        if mode=="bgd":
            self.theta , J = self.batch_gradient_descent(alpha=alpha,epsilon=epsilon,gamma=gamma,t_max=t_max)
            return self.theta,J
        elif mode=="mbgd":
            self.theta , J = self.mini_batch_gradient_descent(alpha=alpha,epsilon=epsilon,gamma=gamma,t_max=t_max,batch_size=batch_size)
            return self.theta,J
        
    # prediction
    def predict(self,X):
        theta = self.theta
        y_pred = self.sigmoid(X,theta)
        y_pred[y_pred>0.5] = 1
        y_pred[y_pred<=0.5] = 0
        return y_pred.astype(int)
    
    # evaluate the model
    def score(self,X,y):
        y_pred = self.predict(X)
        accuracy = np.count_nonzero(y_pred==y)/len(y_pred)
        return 100*accuracy
        