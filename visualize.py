import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix as cm
import seaborn as sns


def confusion_matrix(y_test,y_predict):
        matrix = cm(y_test,y_predict)
        ax = plt.subplot()
        sns.heatmap(matrix, annot=True, ax = ax,fmt='g'); #annot=True to annotate cells
        # labels, title and ticks
        ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
        ax.set_title('Confusion Matrix'); 
        ax.xaxis.set_ticklabels(['Normal', 'Brain Tumor']); ax.yaxis.set_ticklabels(['Normal', 'Brain Tumor'])
        return ax  
    
def cost_curve(J,title,xlabel,c):
    fig = plt.figure()
    plt.plot(np.arange(len(J)), J,c=c)
    plt.xlabel(xlabel)
    plt.ylabel(r'$J(\Theta)$')
    plt.grid()
    plt.title(title)
    plt.savefig("figures/cost_curve.png") # if wanna save plot