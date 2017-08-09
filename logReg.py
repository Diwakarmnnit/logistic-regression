import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def sigmoid(X):
    return 1/(1+np.exp(-X))

def costFunction(theta,X,Y):
    h = sigmoid(X.dot(theta));
    m = len(Y);
    J = (1/m) * sum(np.matmul(-Y.T,np.log(h)) - np.matmul((1-Y).T,np.log(1-h)) );
    return J;

def gradient(theta,X,Y):
   h = sigmoid(X.dot(theta.reshape(-1,1)));
   m = len(Y)
   grad =(1/m) *(X.T.dot(h-Y))
   return (grad.flatten())

def predict(theta, X):
    p = sigmoid(X.dot(theta)) >=0.5
    return p;

#Reading the data
data = pd.read_csv("ex2data1.txt",sep=",",header=None,names=["score1", "score2", "result"])

x1 = data[["score1"]].values
x2 = data[["score2"]].values
Y = data[["result"]].values

# Adding X0(ones)
X_dataframe = data;
del X_dataframe['result']
X_dataframe.insert(0,'ones',1)

#Converting to matrix
X = X_dataframe.as_matrix();

#Plotting the data
for i in range(0,len(Y)):

    if Y[i]==0:
       a = plt.scatter(x1[i], x2[i],color="r")
        
    else:
       b = plt.scatter(x1[i],x2[i],color="b",marker='+')
       
plt.legend((a,b),('Not admitted','Admitted'))

plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")

#Initial theta
[a,n]=X.shape;
theta = np.zeros((n,1));


#Cost function for initial theta
print("Cost function for initial theta is:",costFunction(theta,X,Y));

#Gradient for initial theta
print("Gradient at initial theta: ", gradient(theta,X,Y))


#Optimal theta
res = minimize(costFunction, theta, args=(X,Y), method=None, jac=gradient, options={'maxiter':400})

print("Cost function for optimal theta is:",costFunction(res.x.T,X,Y));
print("Gradient at optimal theta: ", gradient(res.x.T,X,Y));


#Predicting the result
#Exam score 1 : 45, Exam score 2 : 85
testSample = np.array([1,45,85]);
optimalTheta = res.x.T;
prediction = sigmoid(testSample.dot(optimalTheta));

plt.scatter(45,85,color='black',marker=',');


print("Prediction for student:", prediction)


#Training accuracy 
p = predict(res.x, X)
rightPredictions=0;
for i in range(0,len(p)):
    if p[i]==Y[i]:
      rightPredictions+=1;

print("Training accuracy", rightPredictions/len(p)*100, "%")

