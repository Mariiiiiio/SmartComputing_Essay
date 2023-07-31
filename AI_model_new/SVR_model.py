from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox as lb_test
from statsmodels.tsa.stattools import adfuller 
import pmdarima as pm
plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False

sys.path.append(r'C:\Users\USER\Desktop\University\Project\SmartComputing_Essay\AI_model') #for windows
sys.path.append('/Users/mariio/專題/論文專題')  #for mac
sys.path.append(r'C:\Users\USER\Desktop\University\Project\SmartComputing_Essay')

# sys.path.append('/Users/mariio/專題/論文專題')  #for mac

from AI_model.data_process import data_col
def stableCheck(origin, time_diff):
    fig = plt.figure(figsize=(12, 8))
    orig = plt.plot(origin, color='blue', label='Original')
    diff = plt.plot(time_diff, color='red', label='After Differencing')
    plt.legend(loc='best')
    plt.title('Before & After Differencing')
    plt.show()

def  draw_graph(train_data, test_data , round_num):

    plt.subplot(2, 4, round_num)
    plt.title(f'ICA = {round_num}')
    plt.plot(range(1, 50), train_data, 'co-', label = f'train data', markersize=4)
    plt.plot(range(1, 50), test_data, 'go-', label = f'test data', markersize=4)
    plt.legend()
    plt.xlabel("C number")
    plt.ylabel("Value")
    # plt.show()
def whiteNoiseCheck(data):
    
    result = lb_test(data, lags=1)
    # temp = result[1]
    print('白噪聲檢驗结果：', result)
    
    # 如果temp小於0.05，則可以以95%的概率拒絕虛無假設，認為該序列為非白噪聲序列；否則，為白噪聲序列，將會沒有分析意義
    # print(result.shape)
    
    
def ARIMA_preVal(data, ind_name):
    
    '''
    In this part, i annotate all the print line because of the finsh of testing
    '''
    
    
    # from chart_studio.plotly import plot_mpl
    # from plotly.offline import plot_mpl
    result = seasonal_decompose(data, model='multiplicative')
    # result = seasonal_decompose(data3['總指數(不含土石採取業)'], model='multiplicative')
    # result = seasonal_decompose(data, model=’multiplicative’)
    
    # result.plot()
    # plt.show()
    
    '''

    '''
      

    # print('--------First Test - Results of Dickey-Fuller Test:')
    dftest = adfuller(data, autolag='AIC')
    # 对检验结果进行语义描述
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
        
    # print('ADF檢驗結果:')
    # print(dfoutput)
    
    time_series_diff1 = data.diff(1)
    
    # print('Nan Result : ', end="")
    # print(time_series_diff1[time_series_diff1.isnull().values==True], time_series_diff1.shape)
    # print()
    
    time_series_diff1 = time_series_diff1.dropna()
    
    # second_adf_test(time_series_diff1, data)
    # print('---White Noise Test:')
    # whiteNoiseCheck(time_series_diff1)
    

    
    return time_series_diff1
def second_adf_test(data, orig):
    
    stableCheck(orig, data)
    
    print('Second Test - Results of Dickey-Fuller Test:')
    dftest = adfuller(data, autolag='AIC')
    # 对检验结果进行语义描述
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print('ADF檢驗結果:')
    print(dfoutput)



# Documents : 
def call_ARIMA_model():
    data_target, data_mine, data_ele_gas, data_water, data_tech, data_chemi, data_metal_mach= Call_329data()
    cont_nm = [data_tech, data_metal_mach, data_chemi, data_ele_gas]
    # cont_nm = [data_metal_mach, data_chemi]
    Industry_name = ['資訊電子工業',"金屬機電工業", '化學工業', '電力及燃氣供應業']
    # Industry_name = ["金屬機電工業", '化學工業']
    cnt = 0 
    prediction_value_temp = pd.DataFrame(columns=[])
    
    for i in cont_nm:
        data = 0
        data = ARIMA_preVal(i, Industry_name[cnt])

        print(data.shape)
        print()
        print(f'Model Training : {Industry_name[cnt]}')
        n = 263 #number of training data
        
        df_train = data[:][:n].copy()
        df_test = data[:][n:].copy()
        
        print(f'train : {df_train.shape}')
        print(f'test : {df_test.shape}')

        df_test.replace([np.inf, -np.inf], 0, inplace=True)
        
        
        
        # auto_arima  = pm.auto_arima(df_train, stepwise=False, seasonal=False)
        auto_arima  = pm.auto_arima(df_train, start_p=0,   # p最小值
                                start_q=0,   
                                test='adf',  #d
                                max_p=5,     
                                max_q=5,                                 
                                stepwise=False, 
                                seasonal=True,   
                                m = 12,                               
                                error_action='ignore')
        
        # model_fit = auto_arima.fit()
        auto_arima.fit(df_train)    
        
        # print(auto_arima)
        
        print(auto_arima.summary())

        import matplotlib.pyplot as plt
        
        forecast_test_auto = auto_arima.predict(n_periods=len(df_test))
        # print(forecast_test_auto)
        
        
        prediction_value_temp.insert(cnt, column=Industry_name[cnt], value=forecast_test_auto)
        
        
        print('-'*50)
        # print(forecast_test_auto)
        
        forecast_test_auto = pd.DataFrame(forecast_test_auto,index = df_test.index,columns=['Prediction'])
        
        concat1 = pd.concat([df_test, forecast_test_auto],axis=1).plot()
        plt.title(f'{Industry_name[cnt]} | 預測效果展示圖')
        # concat2 = pd.concat([target,forecast_test_auto],axis=1)
        
        from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

        mae = mean_absolute_error(df_test, forecast_test_auto)
        mape = mean_absolute_percentage_error(df_test, forecast_test_auto)
        rmse = np.sqrt(mean_squared_error(df_test, forecast_test_auto))
        r2_rec = r2_score(df_test, forecast_test_auto)


        print(f'mae - auto: {mae}')
        print(f'mape - auto: {mape}')
        print(f'rmse - auto: {rmse}')
        print(f'R2 score : {r2_rec}')
        plt.show()
        
        
        print('-'*100)
        cnt += 1
        # print(f'----------預測值：{prediction_value_temp}')
    prediction_value_temp.to_csv('Prediction_value.csv')
    return prediction_value_temp
    
def Call_Model_SVR(predict_value):
    # start = time.time()
    
    data_target, data_mine, data_ele_gas, data_water, data_tech, data_chemi, data_metal_mach= Call_329data()
    cont_nm = [data_tech, data_metal_mach, data_chemi, data_ele_gas]
    # cont_nm = [data_metal_mach, data_chemi]
    Industry_name = ['資訊電子工業',"金屬機電工業", '化學工業', '電力及燃氣供應業']

    # print(data_ele_gas.index)
    # print(f'len of data tech : {data_target.shape}')
    # print(f'len of data mine : {data_mine.shape}')
    # print(f'len of data ele gas : {data_ele_gas.shape}')
    # print(f'len of data tech : {data_tech.shape}')
    
    #-----------container init
    mse_rec = 1000000
    count = 0
    r2_rec = 0

    # -----------Training & Testing Data prepare:
    '''
    ----------DataSet split

    原本資料 : 288筆 ; 扣除經由2018-2019年的24筆資料，兩年資料將做為驗證預測的可行性

    Data : 264(含土礦) : 1996 M2 ~ 2017 M12
        >>> training set : 237
        >>> testing set : 27
        
    '''
    
    data_col = pd.DataFrame(columns=[])
    for i in range(len(Industry_name)):
        data_col.insert(i, column=Industry_name[i], value = cont_nm[i])
        
    print(data_col)
    n = 264
    data_col_svr = data_col.iloc[:][:n]
    data_target_svr = data_target.iloc[:][:n]
    print(data_col_svr)
    print(data_target_svr)
    # Number of Training Data
    x_train, x_test, y_train, y_test = train_test_split(data_col_svr, data_target_svr, test_size=0.0)
    print(x_train)
    print(x_test)
    print(f'X training data : {x_train.shape},\n x testing data : {x_test.shape}, \n y training data : {y_train.shape}, \n y testing data : {y_test.shape} ')
    a = input('press anything~~~')
    # -----------SVR_model
    print('SVR Result =====')
    test_sc = []
    train_sc = []
    test_sc_num = 0
    train_sc_num = 0
    
    for j in range(1,50):
        # print(f'C = {j} ..........')
        svr_model = SVR(C= j,kernel='rbf', degree= 3, gamma='auto', max_iter=-1)
        svr_model.fit(x_train, y_train)
        y_hat = svr_model.predict(x_test)
        #Score showing
        # print("Training  Score : ", svr_model.score(x_train,y_train))
        # print("Testing  Score : ", svr_model.score(x_test, y_test))
        # print("R^2 得分:", r2_score(y_test, y_hat))
        mse_score = mse(y_test, y_hat)
        # print("MSE_Score : ", mse_score)
        # print("RMSE_Score : ", np.sqrt(mse_score))
        if mse_score < mse_rec:
            mse_rec = mse_score
            count = j
            r2_rec = r2_score(y_test, y_hat)
            MAE_score = mae(y_test, y_hat)
            MAPE_score = mape(y_test, y_hat)
            test_sc_num = svr_model.score(x_test, y_test)
            train_sc_num = svr_model.score(x_train,y_train)
        train_sc.append(svr_model.score(x_train, y_train))
        test_sc.append(svr_model.score(x_test, y_test))
    '''
    draw_graph(train_sc, test_sc, i)
    mse_fig.append(math.sqrt(mse_rec))
    param_record[i] = {            
            'C': count,
            'best_rmse_score': math.sqrt(mse_rec), 
            'MAE Score' :  MAE_score,
            'MAPE Score' : MAPE_score,
        }
    '''

def Call_329data():
    #data_329 = pd.DataFrame(pd.read_csv('..\..\OriginalValue(329)_copy.csv',encoding='cp950', index_col=0)) #mac
    # data_329 = pd.DataFrame(pd.read_csv('..\OriginalValue(329)_copy.csv',encoding='cp950', index_col=0)) 
    data_329 = pd.DataFrame(pd.read_csv('/Users/mariio/專題/論文專題/OriginalValue(329)_copy.csv',encoding='cp950', index_col=0)) 
    # print(data_329.head())
    data_329.index = pd.to_datetime(data_329.index)
    # print(data_329.head(10))
    
    '''
    使用行業：
    >>> 資訊電子工業
    >>> 金屬機電工業
    >>> 化學工業
    >>> 電力及燃氣供應業
    '''



    data_329_target = data_329['總指數']
    data_329_mine = data_329['礦業及土石採取業']
    data_329_ele_gas = data_329['電力及燃氣供應業']
    data_329_water = data_329['用水供應業']
    data_329_tech = data_329['資訊電子工業']
    data_329_chemi = data_329['化學工業']
    data_329_metal_mach = data_329['金屬機電工業']
    return data_329_target, data_329_mine, data_329_ele_gas, data_329_water, data_329_tech, data_329_chemi, data_329_metal_mach
if __name__ == '__main__':
    # data_prediction_value = call_ARIMA_model()
    data_prediction_file = pd.read_csv('/Users/mariio/專題/論文專題/AI_model_new/Prediction_value_24m.csv', index_col=0)
    data_prediction_file.index = pd.to_datetime(data_prediction_file.index)
    print(data_prediction_file)
    # Call_Model_SVR(data_prediction_value)
    
