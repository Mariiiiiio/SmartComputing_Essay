import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse_sk
from sklearn.metrics import accuracy_score as accur
import matplotlib.pyplot as plt
import sys
# from sklearn.metrics import mean_absolute_error as mse
# from sklearn.metrics import acc
sys.path.append(r'C:\Users\USER\Desktop\University\Project\SmartComputing_Essay\AI_model') #for windows

from data_process import data_col

# Loading dataset
data1, data2, data1_1, data2_2, data3, data4, data5, data6 = data_col()

 

data_ar = np.array(data1_1.drop('製造業', axis=1)).astype('int64')


X = data_ar
# print(type(data_ar))
data = load_iris()
# X = data.data
# y = data.target
# print(data5)
target_ori = np.array(data5).astype('int64')

# print(type(target_ori))

y = target_ori.reshape((len(target_ori), 1))
# print(y)
# print(X)
# Split dataset into training and test sets
#

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
print('-'+"shape")
print(X_train.shape)
print(y_train.shape)


# Hyperparameters
learning_rate = 0.1
iterations = 5000
N = y_train.size

input_size = 6
hidden_size = 2*input_size
output_size = 3
 
np.random.seed(10)
W1 = np.random.normal(scale=0.5, size=(input_size, hidden_size))
W2 = np.random.normal(scale=0.5, size=(hidden_size, output_size))
 
# Helper functions
 
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
 
def mean_squared_error(y_pred, y_true):
    # One-hot encode y_true (i.e., convert [0, 1, 2] into [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    eye = np.eye(output_size)
    print(eye.shape)
    print(y_true.shape)
    y_true_one_hot = np.eye(output_size)[y_true]
    
     
    # Reshape y_true_one_hot to match y_pred shape
    y_true_reshaped = y_true_one_hot.reshape(y_pred.shape)
     
    # Compute the mean squared error between y_pred and y_true_reshaped
    error = ((y_pred - y_true_reshaped)**2).sum() / (2*y_pred.size)
 
    return error
 
def accuracy(y_pred, y_true):
    acc = y_pred.argmax(axis=1) ==  y_true.argmax(axis=1)
    return acc.mean()
 
results = pd.DataFrame(columns=["mse", "accuracy"])
 
# Training loop
 
for itr in range(iterations):
    # Feedforward propagation
    Z1 = np.dot(X_train, W1)
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1, W2)
    A2 = sigmoid(Z2)
    print('-'*10)
    
    # print(A2.shape, y_train.shape)
    # Calculate error
    mse = mean_squared_error(A2, y_train)
    # print(np.eye(output_size)[y_train])
    acc = accuracy(np.eye(output_size)[y_train], A2)
    new_row = pd.DataFrame({"mse": [mse], "accuracy": [acc]})
    results = pd.concat([results, new_row], ignore_index=True)
 
    # Backpropagation
    E1 = A2 - np.eye(output_size)[y_train]
    dW1 = E1 * A2 * (1 - A2)
    E2 = np.dot(dW1, W2.T)
    dW2 = E2 * A1 * (1 - A1)
 
    # Update weights
    W2_update = np.dot(A1.T, dW1) / N
    W1_update = np.dot(X_train.T, dW2) / N
    W2 = W2 - learning_rate * W2_update
    W1 = W1 - learning_rate * W1_update
 
# Visualizing the results
 
results.mse.plot(title="Mean Squared Error")
plt.show()
 
results.accuracy.plot(title="Accuracy")
plt.show()
 
# Test the model
 
Z1 = np.dot(X_test, W1)
A1 = sigmoid(Z1)
Z2 = np.dot(A1, W2)
A2 = sigmoid(Z2)
test_acc = accuracy(np.eye(output_size)[y_test], A2)
print("Test accuracy: {}".format(test_acc))