from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
import numpy as np
import time
import sys  
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
    
    #StandardScaler 
    from sklearn.preprocessing import StandardScaler as ss
    
    scaled_data = ss().fit_transform(data1_1)

    print(scaled_data.shape)
    fast_ica = FastICA(n_components=6)
    S_ = fast_ica.fit(scaled_data).fit_transform(scaled_data)





    


    
    #ICA draw grphic 
    # plt.figure(figsize=(16,6))
    # plt.subplot(3, 1, 1)
    # plt.plot(S_)
    # plt.title("Recovered ICA")

    # plt.subplot(3, 1, 2)
    # plt.plot(np.hsplit(S_,[1])[0])
    # plt.title("RecoveredI s1")

    # plt.subplot(3, 1, 3)
    # plt.plot(np.hsplit(S_,[1])[1])
    # plt.title("Recovered s2")

    # plt.subplot(6, 1, 4)
    # plt.plot(np.hsplit(S_,[1])[2])
    # plt.title("Recovered ICA")

    # plt.subplot(6, 1, 5)
    # plt.plot(np.hsplit(S_,[1])[3])
    # plt.title("RecoveredI s1")

    # plt.subplot(6, 1, 6)
    # plt.plot(np.hsplit(S_,[1])[4])
    # plt.title("Recovered s2")

    # plt.tight_layout()
    # plt.show()


