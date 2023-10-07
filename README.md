# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1.Prepare your data - Clean and format your data - Split your data into training and testing sets

2.Define your model - Use a sigmoid function to map inputs to outputs - Initialize weights and bias terms

3.Define your cost function - Use binary cross-entropy loss function - Penalize the model for incorrect predictions

4.Define your learning rate - Determines how quickly weights are updated during gradient descent

5.Train your model - Adjust weights and bias terms using gradient descent - Iterate until convergence or for a fixed number of iterations

6.Evaluate your model - Test performance on testing data - Use metrics such as accuracy, precision, recall, and F1 score

7.Tune hyperparameters - Experiment with different learning rates and regularization techniques

8.Deploy your model - Use trained model to make predictions on new data in a real-world application.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: DELLI PRIYA L
RegisterNumber: 212222230029 
*/
```
```
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data = np.loadtxt("/content/ex2data2.txt",delimiter = ',')
x= data[:,[0,1]]
y= data[:,2]
print('Array Value of x:')
x[:5]

print('Array Value of y:')
y[:5]

print('Exam 1-Score graph')
plt.figure()
plt.scatter(x[y == 1][:,0],x[y == 1][:,1],label='Admitted')
plt.scatter(x[y == 0][:,0],x[y == 0][:,1],label=' Not Admitted')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend()
plt.show()


def sigmoid(z):
  return 1/(1+np.exp(-z))
  
print('Sigmoid function graph: ')
plt.plot()
x_plot = np.linspace(-10,10,100)
plt.plot(x_plot,sigmoid(x_plot))
plt.show()


def costFunction(theta,x,y):
  h = sigmoid(np.dot(x,theta))
  j = -(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/x.shape[0]
  grad = np.dot(x.T,h-y)/x.shape[0]
  return j,grad


print('X_train_grad_value: ')
x_train = np.hstack((np.ones((x.shape[0],1)),x))
theta = np.array([0,0,0])
j,grad = costFunction(theta,x_train,y)
print(j)
print(grad)


print('y_train_grad_value: ')
x_train=np.hstack((np.ones((x.shape[0],1)),x))
theta=np.array([-24,0.2,0.2])
j,grad=costFunction(theta,x_train,y)
print(j)
print(grad)

def cost(theta,x,y):
  h = sigmoid(np.dot(x,theta))
  j = -(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/x.shape[0]
  return j


def cost(theta,x,y):
  h = sigmoid(np.dot(x,theta))
  j = -(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/x.shape[0]
  return j

print('res.x:')
x_train = np.hstack((np.ones((x.shape[0],1)),x))
theta = np.array([0,0,0])
res = optimize.minimize(fun=cost,x0=theta,args=(x_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)


def plotDecisionBoundary(theta,x,y):
  x_min,x_max = x[:,0].min()-1,x[:,0].max()+1
  y_min,y_max = x[:,1].min()-1,x[:,1].max()+1
  xx,yy = np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
  x_plot = np.c_[xx.ravel(),yy.ravel()]
  x_plot = np.hstack((np.ones((x_plot.shape[0],1)),x_plot))
  y_plot = np.dot(x_plot,theta).reshape(xx.shape)
  plt.figure()
  plt.scatter(x[y == 1][:,0],x[y == 1][:,1],label='Admitted')
  plt.scatter(x[y == 0][:,0],x[y == 0][:,1],label='Not Admitted')
  plt.contour(xx,yy,y_plot,levels=[0])
  plt.xlabel('Exam  1 score')
  plt.ylabel('Exam 2 score')
  plt.legend()
  plt.show()

print('DecisionBoundary-graph for exam score: ')
plotDecisionBoundary(res.x,x,y)

print('Proability value: ')
prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)


def predict(theta,x):
  x_train = np.hstack((np.ones((x.shape[0],1)),x))
  prob = sigmoid(np.dot(x_train,theta))
  return (prob >=0.5).astype(int)


print('Prediction value of mean:')
np.mean(predict(res.x,x)==y)
```

## Output:
### Array Value of x:
![image](https://github.com/Priya-Loganathan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121166075/13958cfe-acfa-4bef-be02-8e63b5ba23e0)
### Array Value of y:
![image](https://github.com/Priya-Loganathan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121166075/a78e98e4-665c-4274-8e51-af590c355202)
### Score Graph:
![image](https://github.com/Priya-Loganathan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121166075/d96d4d9d-ea76-476f-95a1-1972193f2fce)
### Sigmoid function graph:
![image](https://github.com/Priya-Loganathan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121166075/f2d149ba-35ab-4ab5-bf5b-ef0507c89226)
### X_train_grad value:
![image](https://github.com/Priya-Loganathan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121166075/cdfb71e2-ca56-4d7a-b8e3-7866562421d3)
### Y_train_grad value:
![image](https://github.com/Priya-Loganathan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121166075/8be5a08a-5391-4993-ba3e-d2ea883904f0)
### res.X:
![image](https://github.com/Priya-Loganathan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121166075/0dc0b5a3-9872-42fd-ad21-fa2b626bdced)
### Decision Boundary:
![image](https://github.com/Priya-Loganathan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121166075/cbc04f51-dfdd-4b1d-b3fa-dd2b677df60d)
### Probability value:
![image](https://github.com/Priya-Loganathan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121166075/6b44222b-1adc-4298-96cd-4e889ce0dd70)
### Prediction value of mean:
![image](https://github.com/Priya-Loganathan/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/121166075/43953b43-1d9f-405b-8c2b-5527c7ea2cc6)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

