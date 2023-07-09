import matplotlib.pyplot as plt
import numpy as np
import time
import sys  
from sklearn.decomposition import PCA
from sklearn import preprocessing
import pandas as pd

sys.path.append('/Users/mariio/專題/論文專題/AI_model')  #for mac

# sys.path.append(r'C:\Users\USER\Desktop\University\Project\SmartComputing_Essay\AI_model') #for windows

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

def pca(X,k):#k is the components you want
  #mean of each feature
  n_samples, n_features = X.shape

  mean=np.array([np.mean(X[:,i]) for i in range(n_features)])
  #normalization
  norm_X=X-mean
  print(norm_X.shape)
  #scatter matrix
  scatter_matrix=np.dot(np.transpose(norm_X),norm_X)
  print(scatter_matrix)
  print('-'*50)
  #Calculate the eigenvectors and eigenvalues
  eig_val, eig_vec = np.linalg.eig(scatter_matrix)

  print(eig_val)
  print('-'*50+'Up Eig val')
  print('-'*50+'Up Eig pairs')
  eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(n_features)]
  print(eig_pairs)
  # sort eig_vec based on eig_val from highest to lowest
  eig_pairs.sort(reverse=True)
  # select the top k eig_vec
  feature=np.array([ele[1] for ele in eig_pairs[:k]])
  print('-'*50)
  print(feature)
  #get new data
  data=np.dot(norm_X,np.transpose(feature))
  print('-'*50)
  print(data.shape)
  return data



if __name__ == '__main__':

    start = time.time()
    # Data loading
    data1, data2, data1_1, data2_2, data3, data4, data5, data6 = data_col()
    data_column = ['金屬機電工業', '資訊電子工業',
        '化學工業', '民生工業', '電力及燃氣供應業', '用水供應業']

    #Data Re-Organize
    # print(data2)
    # print(data3)
    
    # 原始值-變數-轉換矩陣型態
    data1_ar = np.array(data1)
    
    # print(data1_1.head(10))
    # data1_1ar = np.array(data1_1.drop('製造業',axis=1))
    data1_1ar = np.array(data1_1)
    # print(data1_1ar)

    # 年增率-變數-轉換矩陣型態
    data2_ar = np.array(data2)
    data2_2ar = np.array(data2_2)
    # print(data2_2ar)

    #Target-setting-To array
    target_ori = np.array(data5)
    target_Year = np.array(data6)
    
    pca(data1_1ar, 2)

