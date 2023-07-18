from sklearn.svm import SVR
# from thundersvm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import FastICA
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as mse
import matplotlib.pyplot as plt
import numpy as np
import time
import sys  
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pmdarima as pm
import pylab as plt
from scipy.signal import argrelextrema
import scipy.interpolate as spi
import pandas as pd

sys.path.append('/Users/mariio/專題/論文專題/AI_model')  #for mac

# sys.path.append(r'C:\Users\USER\Desktop\University\Project\SmartComputing_Essay\AI_model') #for windows
from data_process import data_col, lessData


def sifting(data):
    index = list(range(len(data)))
 
 
    max_peaks = list(argrelextrema(data, np.greater)[0])
    min_peaks = list(argrelextrema(data, np.less)[0])
 
 
    ipo3_max = spi.splrep(max_peaks, data[max_peaks],k=3) #样本点导入，生成参数
    iy3_max = spi.splev(index, ipo3_max) #根据观测点和样条参数，生成插值
 
 
    ipo3_min = spi.splrep(min_peaks, data[min_peaks],k=3) #样本点导入，生成参数
    iy3_min = spi.splev(index, ipo3_min) #根据观测点和样条参数，生成插值
 
 
    iy3_mean = (iy3_max+iy3_min)/2
    return data-iy3_mean
 
 
 
 
def hasPeaks(data):
    max_peaks = list(argrelextrema(data, np.greater)[0])
    min_peaks = list(argrelextrema(data, np.less)[0])
 
 
    if len(max_peaks)>3 and len(min_peaks)>3:
        return True
    else:
        return False
 
 
 
 
# 判断IMFs
def isIMFs(data):
    max_peaks = list(argrelextrema(data, np.greater)[0])
    min_peaks = list(argrelextrema(data, np.less)[0])
 
 
    if min(data[max_peaks]) < 0 or max(data[min_peaks])>0:
        return False
    else:
        return True
 
 
 
 
def getIMFs(data):
    while(not isIMFs(data)):
        data = sifting(data)
    return data
 
 
 
 
# EMD函数
def EMD(data):
    IMFs = []
    while hasPeaks(data):
        data_imf = getIMFs(data)
        data = data-data_imf
        IMFs.append(data_imf)
    return IMFs





if __name__== "__main__":
    # data1, data2, data1_1, data2_2, data3, data4, target, data6 = data_col()
    data1, target =  lessData()
    data = np.array(target)
    # print(len(data))
    df = np.log(target)

    # draw year graph
    
    df.plot()
    # plt.show()
    
    ''' EMD part (Over)
    #EMD Part 

    # 获取极大值
    max_peaks = argrelextrema(data, np.greater)
    #获取极小值
    min_peaks = argrelextrema(data, np.less)

    # 绘制极值点图像
    plt.figure(figsize = (18,6))
    plt.plot(target)
    plt.scatter(max_peaks, data[max_peaks], c='r', label='Max Peaks')
    plt.scatter(min_peaks, data[min_peaks], c='b', label='min Peaks')
    plt.legend()
    plt.xlabel('time (s)')
    plt.ylabel('Amplitude')
    plt.title("Find Peaks")
    # plt.show()


    index = list(range(len(data)))  
    # 获取极值点
    max_peaks = list(argrelextrema(data, np.greater)[0])
    min_peaks = list(argrelextrema(data, np.less)[0])
    
    
    # 将极值点拟合为曲线
    ipo3_max = spi.splrep(max_peaks, data[max_peaks],k=3) #样本点导入，生成参数
    iy3_max = spi.splev(index, ipo3_max) #根据观测点和样条参数，生成插值
    
    
    ipo3_min = spi.splrep(min_peaks, data[min_peaks],k=3) #样本点导入，生成参数
    iy3_min = spi.splev(index, ipo3_min) #根据观测点和样条参数，生成插值
    
    
    # 计算平均包络线
    iy3_mean = (iy3_max+iy3_min)/2
    
    
    # 绘制图像
    plt.figure(figsize = (18,6))
    plt.plot(data, label='Original')
    plt.plot(iy3_max, label='Maximun Peaks')
    plt.plot(iy3_min, label='Minimun Peaks')
    plt.plot(iy3_mean, label='Mean')
    plt.legend()
    plt.xlabel('time (s)')
    plt.ylabel('microvolts (uV)')
    plt.title("Cubic Spline Interpolation")
    # plt.show()


    IMFs = EMD(data)
    n = len(IMFs)+1
    
    


    # 原始信号
    plt.figure(figsize = (18,15))
    plt.subplot(n, 1, 1)
    plt.plot(data, label='Origin')
    plt.title("Origin ")
    
    
    # 若干条IMFs曲线
    for i in range(0,len(IMFs)):
        plt.subplot(n, 1, i+2)
        plt.plot(IMFs[i])
        plt.ylabel('Amplitude')
        plt.title("IMFs "+str(i+1))
    print()

    print(type(IMFs))
    print(len(IMFs[0]))

    
    i = 0
    IMF_arr = np.stack((IMFs[i], IMFs[i+1], IMFs[i+2], IMFs[i+3]),axis=0) 
    IMF_arr = np.stack((IMFs[i], IMFs[i+1], IMFs[i+2], IMFs[i+3]),axis=0) 
    # print(IMF_arr)
    IMF_arr = IMF_arr.T
    # print(IMF_arr.shape)
    print(IMF_arr)

    # daf = pd.DataFrame(IMF_arr, columns=['1','2','3','4'])
    

    plt.legend()
    plt.xlabel('time (s)')
    plt.ylabel('Amplitude')
    # plt.show()
    '''
    

    
    for i in range(2, 49):
        print(f'--------------------------{i} Round')
        # fast_ica = FastICA(n_components=i)
        # S_ = fast_ica.fit(IMF_arr).fit_transform(IMF_arr)

        # print(S_.shape)
        # print(S_[:][0].shape)
        # print(S_)


        
        n = 276 #number of testing data
        df_train = df[:][:n].copy()
        df_test = df[:][n:].copy()

        df_train = df_train
        df_test = df_test
        # print(df_train)
        print(f'train : {df_train.shape}')
        print(f'test : {df_test.shape}')
        
        # df_test.replace([np.inf, -np.inf], 0, inplace=True)



        # Draw ACF PACF graph
        acf_original = plot_acf(df_train)

        pacf_original = plot_pacf(df_train)
        # plt.show()



        from statsmodels.tsa.stattools import adfuller

        adf_test = adfuller(df_train)
        print(f'p-value: {adf_test[1]}')
        
        # df_train_diff = df_train.diff().dropna()
        # df_train_diff.plot()    
        
        # acf_diff = plot_acf(df_train_diff)
        # pacf_diff = plot_pacf(df_train_diff)
        # plt.show()
        

        # auto_arima  = pm.auto_arima(df_train, stepwise=False, seasonal=False)
        auto_arima  = pm.auto_arima(df_train, stepwise=False, seasonal=True, m = i )
        # model_fit = auto_arima.fit()


        # print(auto_arima)
        print(auto_arima.summary())

        import matplotlib.pyplot as plt
        
        forecast_test_auto = auto_arima.predict(n_periods=len(df_test))
        
        print('-'*50)
        # print(forecast_test_auto)
        
        
        from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

        mae = mean_absolute_error(df_test, forecast_test_auto)
        mape = mean_absolute_percentage_error(df_test, forecast_test_auto)
        rmse = np.sqrt(mean_squared_error(df_test, forecast_test_auto))
        r2_rec = r2_score(df_test, forecast_test_auto)


        print(f'mae - auto: {mae}')
        print(f'mape - auto: {mape}')
        print(f'rmse - auto: {rmse}')
        print(f'R2 score : {r2_rec}')
        


        