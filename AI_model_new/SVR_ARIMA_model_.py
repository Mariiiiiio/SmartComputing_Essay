from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, classification_report
from sklearn.metrics import mean_absolute_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
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
from sklearn import svm
from sklearn import datasets
import joblib


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

def  draw_graph(train_data, test_data , round_num = 0):

    # plt.subplot(2, 4, round_num)
    # plt.title(f'ICA = {round_num}')
    plt.plot(range(1, len(train_data)+1), train_data, 'co-', label = f'train value', markersize=4)
    plt.plot(range(1, len(test_data)+1), test_data, 'go-', label = f'test value', markersize=4)
    
    plt.legend()
    plt.xlabel("C number")
    plt.ylabel("Value")
    plt.show()
    
def  draw_graph_SVR_Score(true_val, pred_val):
    date_pre = ['2018-01-01', '2018-02-01', '2018-03-01', '2018-04-01',
               '2018-05-01', '2018-06-01', '2018-07-01', '2018-08-01',
               '2018-09-01', '2018-10-01', '2018-11-01', '2018-12-01',
               '2019-01-01', '2019-02-01', '2019-03-01', '2019-04-01',
               '2019-05-01', '2019-06-01', '2019-07-01', '2019-08-01',
               '2019-09-01', '2019-10-01', '2019-11-01', '2019-12-01']
    # plt.subplot(2, 4, round_num)
    # plt.title(f'ICA = {round_num}')
    # print(true_val)
    # print(pred_val)
    # a = input()
    
    print(true_val)
    print(pred_val)
    # a = input()
    print(true_val.shape)
    print(pred_val.shape)
    a = input()
    plt.plot(range(1, len(true_val)+1), true_val, 'co-', label = f'True', markersize=4)
    plt.plot(range(1, len(pred_val)+1), pred_val, 'go-', label = f'Predictions', markersize=4)
    plt.legend()
    # plt.xticks(range(24), date_pre, rotation=60)
    plt.xlabel("Observation")
    plt.ylabel("Value")
    plt.show()
    
    
def whiteNoiseCheck(data):
    
    result = lb_test(data, lags=1)
    # temp = result[1]
    print('白噪聲檢驗结果：', result)
    
    # 如果temp小於0.05，則可以以95%的概率拒絕虛無假設，認為該序列為非白噪聲序列；否則，為白噪聲序列，將會沒有分析意義
    # print(result.shape)
    
def inv_diff(diff_df, first_value, add_first=True):
    """
    差分序列的索引从1开始
    """
    diff_df.reset_index(drop=True, inplace=True)
    print(diff_df)
    diff_df.index = diff_df.index + 1
    print(diff_df)
    diff_df = pd.DataFrame(diff_df)
    diff_df = diff_df.cumsum()
    df = diff_df + first_value
    if add_first:
        df.loc[0] = first_value
        df.sort_index(inplace=True)
    return df


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
    data_target, data_mine, data_ele_gas, data_water, data_tech, data_chemi, data_metal_mach, data_normal= Call_329data()
    cont_nm = [data_mine, data_metal_mach, data_tech, data_chemi, data_normal, data_ele_gas, data_water]
    # cont_nm = [data_metal_mach, data_chemi]

    Industry_name = ['礦業及土石採取業', '金屬機電工業', '資訊電子工業','化學工業','民生工業','電力及燃氣供應業', '用水供應業']
    # Industry_name = ["金屬機電工業", '化學工業']
    cnt = 0 
    prediction_value_temp = pd.DataFrame(columns=[], index=['mae', 'mape', 'rmse', 'R2'])
    
    for i in cont_nm:
        data = 0
        # data = ARIMA_preVal(i, Industry_name[cnt])
        data = i
        print(data.shape)
        print()
        # print(f'Model Training : {Industry_name[cnt]}')
        n = 264 #number of training data
        
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
        contain_ind = []
        # print(forecast_test_auto)
        
        # forcasting = inv_diff(forecast_test_auto, 1)
        mae = mean_absolute_error(df_test, forecast_test_auto)
        mape = mean_absolute_percentage_error(df_test, forecast_test_auto)
        rmse = np.sqrt(mean_squared_error(df_test, forecast_test_auto))
        r2_rec = r2_score(df_test, forecast_test_auto)
        contain_ind = [mae, mape, rmse, r2_rec]
        prediction_value_temp.insert(cnt, column=Industry_name[cnt], value=contain_ind )
        # print('-'*50)
        # print(forecast_test_auto)
        # print(prediction_value_temp)              
        
        # print('-'*50)
        # print(forecast_test_auto)
        
        forecast_test_auto = pd.DataFrame(forecast_test_auto,index = df_test.index,columns=['Prediction'])
        
        concat1 = pd.concat([df_test, forecast_test_auto],axis=1).plot()
        plt.title(f'{Industry_name[cnt]} | 預測效果展示圖')
        # concat2 = pd.concat([target,forecast_test_auto],axis=1)
        
        

        


        print(f'mae - auto: {mae}')
        print(f'mape - auto: {mape}')
        print(f'rmse - auto: {rmse}')
        print(f'R2 score : {r2_rec}')
        plt.show()        
        
        print('-'*100)
        cnt += 1
        # print(f'----------預測值：{prediction_value_temp}')
    print(prediction_value_temp)
    a = input()
    prediction_value_temp.to_csv('Season_SARIMA_Indicator_value.csv')
    return prediction_value_temp
    
def SVR_prediction(pred, target, weight_name, time):
        reg1 = joblib.load(weight_name)

        print(pred)
        if time == 0:
            temp = np.array(pred)
            temp = temp.reshape(-1, 1)
            pred_result = reg1.predict(temp)
        else:
            pred_result = reg1.predict(pred[:])


        print('-'*50)

        a = input()

        mse_score = mse(target, pred_result)
        MAE_score = mae(target, pred_result)
        MAPE_score = mape(target, pred_result)
        r2_val_score = r2_score(target, pred_result)
        cont = [mse_score, MAE_score, MAPE_score, r2_val_score]
        print("MAE Score : ", MAE_score)
        print("MAPE Score : ", MAPE_score)
        print("RMSE_Score : ", np.sqrt(mse_score))
        print("R square : ", r2_val_score)
        pred_result = pred_result.reshape(-1,1)
        target = target.values.reshape(-1,1)
        draw_graph_SVR_Score(target, pred_result) 
        result_svr_pred = pd.DataFrame(columns=[f'Predictions_set{time}'], index=['MAE', 'MAPE', 'RMSE', 'R2'], data=cont)
        print(result_svr_pred)
        result_svr_pred.to_csv(f'{weight_name}_result_svr_predictions.csv')
def Call_Model_SVR(train_num):
    # start = time.time()
    
    data_target, data_mine, data_ele_gas, data_water, data_tech, data_chemi, data_metal_mach, data_normal= Call_329data()

    cont_nm1 = [data_ele_gas]
    cont_nm2 = [data_ele_gas, data_chemi]
    cont_nm3 = [data_ele_gas, data_chemi, data_metal_mach]


    # cont_nm = [data_metal_mach, data_chemi]
    Industry_name = ['電力及燃氣供應業', '化學工業', "金屬機電工業"]
    

    
    
    # print(data_ele_gas.index)
    # print(f'len of data tech : {data_target.shape}')
    # print(f'len of data mine : {data_mine.shape}')
    # print(f'len of data ele gas : {data_ele_gas.shape}')
    # print(f'len of data tech : {data_tech.shape}')
    
    #-----------container init
    mse_rec = 1000000
    count = 0
    r2_rec = 0
    param_record = {}
    mse_fig = []
    MAE_rec = 100
    MAPE_rec = 100
    # -----------Training & Testing Data prepare:
    '''
    ----------DataSet split

    原本資料 : 288筆 ; 扣除經由2018-2019年的24筆資料，兩年資料將做為驗證預測的可行性

    Data : 264(含土礦) : 1996 M2 ~ 2017 M12
        >>> training set : 237
        >>> testing set : 27
        

    '''
    # data_col = cont_nm1 #for cont_nm1
    
    data_col = pd.DataFrame(columns=[])
    for i in range(len(cont_nm3)):
        data_col.insert(i, column=Industry_name[i], value = cont_nm3[i])
    

    
    # print(data_col)
    n = train_num
    # print('-'*50)
    
    '''
    data_col_svr_ori = data_col[0][:n]
    data_col_svr_ori = np.array(data_col_svr_ori).reshape(-1, 1)
    # data_col_svr_ori = np.array(data_col_svr_ori).reshape(len(data_col_svr_ori), 1)

    data_target_svr_ori = data_target.iloc[:][:n]
    print(data_col_svr_ori)
    print(data_target_svr_ori)
    '''



    '''  for cont_nm1
    data_col_svr_ori = data_col[0][:n] # for cont_nm1 

    data_col_svr_ori = np.array(data_col_svr_ori).reshape(-1, 1)
    '''

    '''for else set'''
    data_col_svr_ori = data_col.iloc[:][:n]
    data_target_svr_ori = data_target.iloc[:][:n]


    # print(data_target_svr_ori.shape)

    # data_col_svr_pre = data_col.iloc[:][n:]
    data_target_svr_pre = data_target.iloc[:][n:]
    
    x_train, x_test, y_train, y_test = train_test_split(data_col_svr_ori, data_target_svr_ori, test_size=0.1
    , random_state=2)
    print(f'data1 : {data_col_svr_ori.shape} \n data2 : {data_target_svr_ori.shape}')
    
    

    '''
    print(f'data1 : {data_col_svr_ori.shape} \n data2 : {data_target_svr_ori.shape}')
    
    x_tr, x_te, y_tr, y_te = train_test_split(data_col_svr_ori, data_target_svr_ori, test_size=0.2, random_state=2)
    
    x_train = pd.concat([x_tr, x_te])
    y_train = pd.concat([y_tr, y_te])
    print(x_train)
    print(y_train)
    
    x_tr2, x_te2, y_tr2, y_te2 = train_test_split(predict_value, data_target_svr_pre, test_size=0.2, random_state=2)
    x_test = pd.concat([x_tr2, x_te2])
    y_test = pd.concat([y_tr2, y_te2])
    print(x_test)
    print(y_test)
    '''
    
    
    
    
    
    # print(data_col_svr)
    # print(data_target_svr)
    
    print(f'X training data : {x_train.shape},\n x testing data : {x_test.shape},\
        \n y training data : {y_train.shape}, \n y testing data : {y_test.shape} ')
    # a = input('presssssss')
    
    
    ''' -----------------SVR part
    data_col_arima_pridiction = predict_value
    data_target_arima = data_target.iloc[:][n:]
    
    
    
    # print(data_col_arima_pridiction)
    # print(data_target_arima)
    
    # Number of Training Data
    x_train_svr, x_test_svr, y_train_svr, y_test_svr = train_test_split(data_col_svr, data_target_svr, test_size=0.2)
    
    
    
    x_train = pd.concat([x_train_svr, x_test_svr])
    # x_train.plot()
    y_train = pd.concat([y_train_svr, y_test_svr])
    # y_train.plot()
    # plt.show()
    x_train_ar, x_test_ar, y_train_ar, y_test_ar = train_test_split(data_col_arima_pridiction, data_target_arima, test_size=0.2)
    print(f'X training data : {x_train_ar.shape},\n x testing data : {x_test_ar.shape}, \n y training data : {y_train_ar.shape}, \n y testing data : {y_test_ar.shape} ')
    '''
    
    '''-----------------ARIMA part
    x_test = pd.concat([x_train_ar, x_test_ar])
    # x_test.plot()
    y_test = pd.concat([y_train_ar, y_test_ar])
    # y_test.plot()
    
    # plt.show()

    print(f'X training data : {x_train.shape},\n x testing data : {x_test.shape}, \n y training data : {y_train.shape}, \n y testing data : {y_test.shape} ')

    '''

    # -----------SVR_model
    print('SVR Result =====')
    test_sc = []
    train_sc = []
    test_sc_num = 0
    train_sc_num = 0
    # num_c = 0.1
    for j in range(30,31):
        
        print(f'C = {j} ..........')
        
        svr_model = SVR(C= j,kernel='rbf', degree= 3, gamma='auto', max_iter=-1)
        svr_model.fit(x_train, y_train)
        y_hat = svr_model.predict(x_test)
        #Score showing
        # print("Training  Score : ", svr_model.score(x_train,y_train))
        # print("Testing  Score : ", svr_model.score(x_test, y_test))
        # print("R^2 得分:", r2_score(y_test, y_hat))
        mse_score = mse(y_test, y_hat)
        MAE_score = mae(y_test, y_hat)
        MAPE_score = mape(y_test, y_hat)
        # print("MSE_Score : ", mse_score)
        # print("RMSE_Score : ", np.sqrt(mse_score))
        r2_val_score = r2_score(y_test, y_hat)
        if r2_val_score > r2_rec:
            # mse_rec = mse_score
            mse_score = mse(y_test, y_hat)
            count = j
            r2_rec = r2_val_score
            MAE_rec = MAE_score
            MAPE_rec = MAPE_score
        # num_c += 0.1
        train_sc.append(svr_model.score(x_train, y_train))
        test_sc.append(svr_model.score(x_test, y_test))
        print(f'training score : {svr_model.score(x_train, y_train)}')
        print(f'testing score : {svr_model.score(x_test, y_test)}')
        print(f'R square score : {r2_val_score}')
        mse_fig.append(math.sqrt(mse_rec))
        
        print()
        true_val = y_test.values.reshape(-1,1)
        pred_val = y_hat.reshape(-1,1)
        
        draw_graph_SVR_Score(true_val, pred_val) 
        a = input()
        joblib.dump(svr_model,'svr_set3.pkl')
        

    # print(classification_report(y_test, y_hat))
    
    # draw_graph(train_sc, test_sc)
    
    
    # print(f'y_hat : \n{y_hat} \ny_ture : {y_test}')
    # for i in range(len(y_hat)):
    #     print(y_hat[i] - y_test[i])
    
    # plt.show()
    param_record = {            
            'C': count,
            'best_rmse_score': math.sqrt(mse_score), 
            'MAE Score' :  MAE_rec,
            'MAPE Score' : MAPE_rec,
            'R2_score': r2_rec, 
        }
    print(param_record)
    
    # a = input('Stopsssssss')
    # result_svr = pd.DataFrame.from_dict(param_record)
    # result_svr.to_csv('Result_SVR.csv')
    
    plt.title('RMSE Score')
    plt.plot(range(len(mse_fig)), mse_fig, 'co-', label="Train Score")
    plt.xlabel('C number')
    plt.ylabel('RMSE value')
    plt.show()
    # return data_target_svr_pre
def Call_329data():
    #data_329 = pd.DataFrame(pd.read_csv('..\..\OriginalValue(329)_copy.csv',encoding='cp950', index_col=0)) #mac
    # data_329 = pd.DataFrame(pd.read_csv('..\OriginalValue(329)_copy.csv',encoding='cp950', index_col=0)) 
    
    data_329 = pd.DataFrame(pd.read_csv('/Users/mariio/專題/論文專題/OriginalValue(329)_copy.csv',encoding='cp950', index_col=0))  #mac ver
    # data_329 = pd.DataFrame(pd.read_csv('..\OriginalValue(329)_copy.csv',encoding='cp950', index_col=0))   #windows ver
    
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
    data_329_normal = data_329['民生工業']
    return data_329_target, data_329_mine, data_329_ele_gas, data_329_water, data_329_tech, data_329_chemi, data_329_metal_mach, data_329_normal
if __name__ == '__main__':
    # data_prediction_value = call_ARIMA_model()


    ''' Prediction value part'''
    #data_prediction_file = pd.read_csv('/Users/mariio/專題/論文專題/AI_model_new/Prediction_value_undiff.csv', index_col=0) #mac ver
    
    # data_prediction_file_264 = pd.read_csv('./Prediction_value_264.csv', index_col=0) #mac ver
    data_prediction_file_264 = pd.read_csv('./Prediction_value_264copy.csv', index_col=0, encoding='cp950') #mac ver
    
    # data_prediction_file_264 = pd.read_csv('.\Prediction_value_264.csv', index_col=0) #windows ver
    data_prediction_file_264.index = pd.to_datetime(data_prediction_file_264.index)
    print(data_prediction_file_264)
    # data_prediction_file_263 = pd.read_csv('.\Prediction_value_263.csv', index_col=0) #windows ver
    # data_prediction_file_263.index = pd.to_datetime(data_prediction_file_263.index)
    a = input()
    n = 264
    data_target, data_mine, data_ele_gas, data_water, data_tech, data_chemi, data_metal_mach, data_normal= Call_329data()
    target = data_target.iloc[:][n:]
    # print(data_prediction_file_264)
    
    # print(f'-----------Start The 264-----------')
    # target = Call_Model_SVR(264)


    
    weight_cont = ['svr_set1.pkl', 'svr_set2.pkl', 'svr_set3.pkl', 'svr_set4.pkl']
    for i in range(0, len(data_prediction_file_264.columns)):
        print(f'-----------Start The prediction part-Weight : {weight_cont[i]}-----------')
        if i == 0:
            print(data_prediction_file_264.iloc[:, i])    
            data_pred = data_prediction_file_264.iloc[:, i]
            # data_pred = data_pred.values.reshape(-1, 1)
            SVR_prediction(data_pred, target, weight_cont[i], i)

        else:
            # print(data_prediction_file_264.iloc[:, 0:i+1])
            SVR_prediction(data_prediction_file_264.iloc[:, 0:i+1], target, weight_cont[i], i)


    
    '''
    # print(f'-----------Start The 263-----------')
    # Call_Model_SVR(data_prediction_file_263, 263)
    '''
