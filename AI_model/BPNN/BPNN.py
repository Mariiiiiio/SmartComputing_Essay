import matplotlib.pyplot as plt
import numpy as np
import time
import sys  
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mse
import pandas as pd
# sys.path.append('/Users/mariio/專題/論文專題/AI_model')  #for mac

sys.path.append(r'C:\Users\USER\Desktop\University\Project\SmartComputing_Essay\AI_model') #for windows

from data_process import data_col

''' Documents : 
    data1 : 原始值-變數
    data2 : 年增率-變數
    data1_1 : 原始值(不包含礦業與土石採取業)-變數
    data2_2 : 年增率(不包含礦業與土石採取業)-變數
    data3 : 原始值-目標
    data4 : 年增率-目標
    data5 : 原始值（不包含礦業與土石採取業) -目標
    data6 : 年增率（不包含礦業與土石採取業) -目標
'''

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def mean_squared_error(y_pred, y_true):
    # One-hot encode y_true (i.e., convert [0, 1, 2] into [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    y_true_one_hot = np.eye(output_size)[y_true]
    
    # Reshape y_true_one_hot to match y_pred shape
    y_true_reshaped = y_true_one_hot.reshape(y_pred.shape)
    
    # Compute the mean squared error between y_pred and y_true_reshaped
    error = ((y_pred - y_true_reshaped)**2).sum() / (2*y_pred.size)

    return error

def accuracy(y_pred, y_true):
    acc = y_pred.argmax(axis=1) == y_true.argmax(axis=1)
    return acc.mean()




if __name__ == '__main__':
    start = time.time()
    # Data loading
    data1, data2, data1_1, data2_2, data3, data4, data5, data6 = data_col()
    
    # 原始值-變數-轉換矩陣型態
    data1_ar = np.array(data1)
    data1_1ar = np.array(data1_1)
    # print(data1_1ar)

    # 年增率-變數-轉換矩陣型態
    data2_ar = np.array(data2)
    data2_2ar = np.array(data2_2)
    # print(data2_2ar)

    #Target-setting-To array
    target_ori = np.array(data5)
    target_Year = np.array(data6)
    y = pd.get_dummies(data5).values
    print(y[:3])
    #Data prepare   
    X_train, X_test, y_train, y_test = train_test_split(data1_1ar, y, test_size=0.2)
    print(y_train.size)
    print(X_train.size)
    print(X_train.shape[1])
    
    # Hyperparameters
    learning_rate = 0.1
    iterations = 5000
    N = y_train.size
    input_size = X_train.shape[1]
    hidden_size = 2*(input_size)
    output_size = 1
    
    np.random.seed(10)
    W1 = np.random.normal(scale=0.5, size=(input_size, hidden_size))
    W2 = np.random.normal(scale=0.5, size=(hidden_size, output_size))
    
    # Helper functions
    
    
    
    results = pd.DataFrame(columns=["mse", "accuracy"])
    
    # Training loop
    
    for itr in range(iterations):
        # Feedforward propagation
        Z1 = np.dot(X_train, W1)
        A1 = sigmoid(Z1)
        Z2 = np.dot(A1, W2)
        A2 = sigmoid(Z2)
    
        # Calculate error
        mse = mean_squared_error(A2, y_train)
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
