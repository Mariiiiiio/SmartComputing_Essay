from sklearn.svm import SVR
# from thundersvm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as mse
import matplotlib.pyplot as plt
import numpy as np
import time
import sys  
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox as lb_test

sys.path.append('/Users/mariio/專題/論文專題/AI_model')  #for mac

sys.path.append(r'C:\Users\USER\Desktop\University\Project\SmartComputing_Essay\AI_model') #for windows

from data_process import data_col

def stableCheck(origin, time_diff):
    fig = plt.figure(figsize=(12, 8))
    orig = plt.plot(origin, color='blue', label='Original')
    diff = plt.plot(time_diff, color='red', label='After Differencing')
    plt.legend(loc='best')
    plt.title('Before & After Differencing')
    plt.show()
    
def whiteNoiseCheck(data):
    
    result = lb_test(data, lags=1)
    # temp = result[1]
    print('白噪聲檢驗结果：', result)
    # 如果temp小于0.05，则可以以95%的概率拒绝原假设，认为该序列为非白噪声序列；否则，为白噪声序列，认为没有分析意义
    print(result.shape)
    return result

# if __name__== "__main__":
def newdata_generate(opt):
    # data1, data2, data1_1, data2_2, data3, data4, data5, data6 = data_col()
    data3 = pd.DataFrame(pd.read_csv('..\..\OriginalValue_copy.csv',encoding='cp950', index_col=0))
    # print(data3.head())
    data3.index = pd.to_datetime(data3.index)
    data3_og = data3['總指數(不含土石採取業)']
    # print(data3.head(10))
    
    
    data_329 = pd.DataFrame(pd.read_csv('..\..\OriginalValue(329)_copy.csv',encoding='cp950', index_col=0))
    # print(data_329.head())
    data_329.index = pd.to_datetime(data_329.index)
    # print(data_329.head(10))
    data_329_og = data_329['總指數']
    # print(data_329_og.head(10))
    
    
    '''
    from chart_studio.plotly import plot_mpl
    from plotly.offline import plot_mpl
    result = seasonal_decompose(data_329['總指數'], model='multiplicative')
    # result = seasonal_decompose(data3['總指數(不含土石採取業)'], model='multiplicative')
    # result = seasonal_decompose(data, model=’multiplicative’)
    result.plot()
    plt.show()
    '''

    '''
    from statsmodels.tsa.stattools import adfuller   

    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(data3_og, autolag='AIC')
    # 对检验结果进行语义描述
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print('ADF檢驗結果:')
    print(dfoutput)
    '''
    
    #opt = 1 => less data
    #opt = 2 => full data => num : 329
    
    if opt == 1:
        time_series_diff1 = data3_og.diff(1)
    elif opt == 2:
        time_series_diff1 = data_329_og.diff(1)
        
    # print(time_series_diff1[time_series_diff1.isnull().values==True], time_series_diff1.shape)
    time_series_diff1 = time_series_diff1.dropna()
    
    # print('-'*100)
    time_series_diff2 = time_series_diff1.diff(12)
    # print(time_series_diff2[time_series_diff2.isnull().values==True], time_series_diff2.shape)
    
    time_series_diff2 = time_series_diff2.dropna()
    # print(time_series_diff2)
    
    return time_series_diff2
    '''
    print('-'*100)
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(time_series_diff2, autolag='AIC')
    # 对检验结果进行语义描述
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print('ADF檢驗結果:')
    print(dfoutput)
    
    stableCheck(data_329_og ,time_series_diff2)
    
    print('-'*100)
    
    ifwhiteNoise = whiteNoiseCheck(time_series_diff2)
    '''
    
    
    
    # df = np.log(data5)
    # df.plot()
    # plt.show()
    
    # n = 100 #number of tesing data
    # msk = (df.index < len(df)-n)
    # print(df.index)
    # print(msk)
    # df_train = df[msk].copy()
    # df_test = df[~msk].copy()
    
    # print(len(df_train))
    # print(len(df_test))
    # df_test.replace([np.inf, -np.inf], 0, inplace=True)

    # acf_original = plot_acf(df_train)
    # pacf_original = plot_pacf(df_train)
    # plt.show()
    
    # from statsmodels.tsa.stattools import adfuller
    # adf_test = adfuller(df_train)
    # print(f'p-value: {adf_test[1]}')
    
    # # df_train_diff = df_train.diff().dropna()
    # # df_train_diff.plot()    
    
    # # acf_diff = plot_acf(df_train_diff)
    # # pacf_diff = plot_pacf(df_train_diff)
    # # plt.show()

    # model = ARIMA(df_train, order=(2,1,0))
    # model_fit = model.fit()
    # print(model_fit.summary())
    # import matplotlib.pyplot as plt

    # residuals = model_fit.resid[1:]
    # fig, ax = plt.subplots(1,2)
    # residuals.plot(title='Residuals', ax=ax[0])
    # residuals.plot(title='Density', kind='kde', ax=ax[1])
    # # plt.show()
    
    # acf_res = plot_acf(residuals)
    # pacf_res = plot_pacf(residuals)
    # # plt.show()
    
    # forecast_test = model_fit.forecast(len(df_test))

    # from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

    # mae = mean_absolute_error(df_test, forecast_test)
    # mape = mean_absolute_percentage_error(df_test, forecast_test)
    # rmse = np.sqrt(mean_squared_error(df_test, forecast_test))

    # print(f'mae - manual: {mae}')
    # print(f'mape - manual: {mape}')
    # print(f'rmse - manual: {rmse}')