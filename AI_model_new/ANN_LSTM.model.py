import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.layers import LSTM
import sys
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False

sys.path.append(r'C:\Users\USER\Desktop\University\Project\SmartComputing_Essay\AI_model') #for windows
sys.path.append('/Users/mariio/專題/論文專題')  #for mac
sys.path.append(r'C:\Users\USER\Desktop\University\Project\SmartComputing_Essay')

def ann_model():
    data_target, data_mine, data_ele_gas, data_water, data_tech, data_chemi, data_metal_mach, data_normal= Call_329data()
    Industry_name = ['總指數', '資訊電子工業',"金屬機電工業", '化學工業', '電力及燃氣供應業']

    
    data_catagory = [data_mine, data_metal_mach, data_tech, data_chemi, data_normal, data_ele_gas, data_water]
    Industry_name = ['礦業及土石採取業', '金屬機電工業', '資訊電子工業','化學工業','民生工業','電力及燃氣供應業', '用水供應業']
    
    date_pre = ['2018-01-01', '2018-02-01', '2018-03-01', '2018-04-01',
               '2018-05-01', '2018-06-01', '2018-07-01', '2018-08-01',
               '2018-09-01', '2018-10-01', '2018-11-01', '2018-12-01',
               '2019-01-01', '2019-02-01', '2019-03-01', '2019-04-01',
               '2019-05-01', '2019-06-01', '2019-07-01', '2019-08-01',
               '2019-09-01', '2019-10-01', '2019-11-01', '2019-12-01']
    
    # split_date = pd.Timestamp()
    n = 240 #number of training data
    cnt = 0
    prediction_value_temp = pd.DataFrame(columns=[], index=['mae', 'mape', 'rmse', 'R2'])
    for data in data_catagory:
        df_train = data[:][:n].copy()
        df_test = data[:][n:].copy()
        
        # split_date = pd.Timestamp('2018-01-01')
        plt.figure(figsize=(10, 6))
        ax = df_train.plot()
        df_test.plot(ax=ax)
        plt.legend(['train', 'test'])
        plt.title(Industry_name[cnt])
        plt.show()
        
        # scaler = MinMaxScaler(feature_range=(-1, 1))
        # train_sc = scaler.fit_transform(df_train.values.reshape(-1,1))
        # test_sc = scaler.transform(df_test.values.reshape(-1,1))
        
        df_train = df_train.values.reshape(-1,1)
        df_test = df_test.values.reshape(-1,1)
        # print(f'train : {train_sc.shape}, test : {test_sc.shape}')
        predict_mth = 24
        

        X_train = df_train[:-predict_mth]
        y_train = df_train[predict_mth:]
        
        X_test = df_test[:-predict_mth]
        y_test = df_test[predict_mth:]

        # print(target[-predict_mth:].index)
        
        print(f'x_train : {X_train.shape},x_test : {X_test.shape},y_train : {y_train.shape}, y_test : {y_test.shape}')
        # a = input()
        print(f'{Industry_name[cnt]}---------')
        nn_model = Sequential()
        nn_model.add(Dense(24, input_dim=1, activation='relu'))
        # nn_model.add(Dense(12, input_dim=2, activation='PRelu'))
        nn_model.add(Dense(1))
        
        nn_model.summary()
        nn_model.compile(loss='mean_squared_error', optimizer='adam')
        early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
        history = nn_model.fit(X_train, y_train, epochs=200, batch_size=1, verbose=1, callbacks=[early_stop], shuffle=False)
        
        y_pred_test_nn = nn_model.predict(X_test)
        y_train_pred_nn = nn_model.predict(X_train)
        print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_train_pred_nn)))
        print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_pred_test_nn)))
        print(df_test)
        print(y_pred_test_nn)
        
        mae = mean_absolute_error(y_test, y_pred_test_nn)
        mape = mean_absolute_percentage_error(y_test, y_pred_test_nn)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test_nn))
        r2_rec = r2_score(y_test, y_pred_test_nn)
        contain_ind = [mae, mape, rmse, r2_rec]
        prediction_value_temp.insert(cnt, column=Industry_name[cnt], value=contain_ind )
        print(y_test.shape)
        print(y_pred_test_nn.shape)
        plt.figure(figsize=(10, 6))
        plt.plot(y_test, label='True')
        plt.plot(y_pred_test_nn, label='Preditcions')
        plt.title(f"ANN's Prediction : {Industry_name[cnt]}")
        plt.xlabel('Observation')
        plt.ylabel('Adj Close Scaled')
        plt.xticks(range(24), date_pre, rotation=60)
        plt.legend()
        plt.show()

        
        cnt += 1
    prediction_value_temp.to_csv('ANN_Indicator_value.csv')
    

def Call_329data():
    #data_329 = pd.DataFrame(pd.read_csv('..\..\OriginalValue(329)_copy.csv',encoding='cp950', index_col=0)) #mac
    # data_329 = pd.DataFrame(pd.read_csv('..\OriginalValue(329)_copy.csv',encoding='cp950', index_col=0)) 
    # data_329 = pd.DataFrame(pd.read_csv('/Users/mariio/專題/論文專題/OriginalValue(329)_copy.csv',encoding='cp950', index_col=0))  #mac ver
    data_329 = pd.DataFrame(pd.read_csv('..\OriginalValue(329)_copy.csv',encoding='cp950', index_col=0))   #windows ver
    data_329.index = pd.to_datetime(data_329.index)

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
    ann_model()