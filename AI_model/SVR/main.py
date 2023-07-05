from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as mse
import matplotlib.pyplot as plt
import numpy as np
import time
import sys  

# sys.path.append('/Users/mariio/專題/論文專題/AI_model')  #for mac

sys.path.append(r'C:\Users\USER\Desktop\University\Project\SmartComputing_Essay\AI_model') #for windows

from data_process import data_col


if __name__ == '__main__':

    start = time.time()
    # Data loading
    data1, data2, data1_1, data2_2, data3, data4, data5, data6 = data_col()
    '''
    data1 : 原始值-變數
    data2 : 年增率-變數
    data1_1 : 原始值(不包含礦業與土石採取業)-變數
    data2_2 : 年增率(不包含礦業與土石採取業)-變數
    data3 : 原始值-目標
    data4 : 年增率-目標
    data5 : 原始值（不包含礦業與土石採取業) -目標
    data6 : 年增率（不包含礦業與土石採取業) -目標
    '''

    #Data Re-Organize
    # print(data2)
    # print(data3)
    
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

    # print(data1.shape) 
    # print(data1_1ar.shape) 
    # print(data2_ar.shape) 
    # print(data2_2ar.shape) 
    # print(target_ori.shape) 
    # print(target_Year.shape) 

    #Data prepare
    x_train, x_test, y_train, y_test = train_test_split(data1_1ar, target_ori, test_size=0.2)
    
    # print(x_train.shape)
    # print(y_train.shape)
    #SVR model create
    print('-'*50+'SVR Started--')
    polyModel=SVR(C=6, kernel='poly', gamma='auto', max_iter=-1, verbose=0)
    polyModel.fit(x_train, y_train)
    y_hat=polyModel.predict(x_test)
    # print(y_hat)
    print("得分:", r2_score(y_test, y_hat))
    # print(y_test)
    mse_score = mse(y_test, y_hat)
    print("MSE_Score : ", mse_score)
    print("RMSE_Score : ", np.sqrt(mse_score))
    
    r = len(x_test) + 1
    plt.plot(np.arange(1,r), y_hat, 'go-', label="predict")
    plt.plot(np.arange(1,r), y_test, 'co-', label="real")
    plt.legend()
    end = time.time()
    print("執行時間：%f 秒" % (end - start))
    plt.show()





