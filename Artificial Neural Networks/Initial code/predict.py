import numpy as np

from sigmoid import sigmoid

def predict(Theta1, Theta2, X):
#PREDICT Predict the label of an input given a trained neural network
#   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
# Useful values
        #number of examples
#   trained weights of a neural network (Theta1, Theta2)
    m = np.shape(X)[0]         
    ones = np.ones((m,1))
    inp_X = np.hstack((ones,X))
    

    
# You need to return the following variables correctly 
    
    p = np.zeros(m);

# ====================== YOUR CODE HERE ======================
# Instructions: Complete the following code to make predictions using
#               your learned neural network. You should set p to a 
#               vector containing labels between 1 to num_labels.
#
    a = inp_X
    z2 = np.dot(a, Theta1.transpose())
    
    a2 = sigmoid(z2)
    ones = np.ones((np.shape(a2)[0],1))
    a2 = np.hstack((ones, a2))
    
    
    z3 = np.dot(a2, Theta2.transpose())
    a3 = sigmoid(z3)
    ones = np.ones((np.shape(a3)[0],1))
    a2 = np.hstack((ones, a3))

   
    
    p = a3
    p = np.argmax(p,axis=1)

    return p

# =========================================================================
