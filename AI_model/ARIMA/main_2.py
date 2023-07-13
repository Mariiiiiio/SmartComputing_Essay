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

sys.path.append('/Users/mariio/專題/論文專題/AI_model')  #for mac

# sys.path.append(r'C:\Users\USER\Desktop\University\Project\SmartComputing_Essay\AI_model') #for windows

from data_process import data_col, lessData


if __name__== "__main__":
    data1, data2, data1_1, data2_2, data3, data4, data5, data6 = data_col()
    # data1, data5 =  lessData()
    
    df = np.log(data5)
    df.plot()
    plt.show()
    
    n = 100 #number of tesing data
    msk = (df.index < len(df)-n)
    
    df_train = df[msk].copy()
    df_test = df[~msk].copy()
    
    print(len(df_train))
    print(len(df_test))
    df_test.replace([np.inf, -np.inf], 0, inplace=True)

    acf_original = plot_acf(df_train)
    pacf_original = plot_pacf(df_train)
    plt.show()
    
    from statsmodels.tsa.stattools import adfuller
    adf_test = adfuller(df_train)
    print(f'p-value: {adf_test[1]}')
    
    # df_train_diff = df_train.diff().dropna()
    # df_train_diff.plot()    
    
    # acf_diff = plot_acf(df_train_diff)
    # pacf_diff = plot_pacf(df_train_diff)
    # plt.show()

    model = ARIMA(df_train, order=(2,1,0))
    model_fit = model.fit()
    print(model_fit.summary())
    import matplotlib.pyplot as plt

    residuals = model_fit.resid[1:]
    fig, ax = plt.subplots(1,2)
    residuals.plot(title='Residuals', ax=ax[0])
    residuals.plot(title='Density', kind='kde', ax=ax[1])
    plt.show()
    
    acf_res = plot_acf(residuals)
    pacf_res = plot_pacf(residuals)
    plt.show()
    
    forecast_test = model_fit.forecast(len(df_test))

    from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

    mae = mean_absolute_error(df_test, forecast_test)
    mape = mean_absolute_percentage_error(df_test, forecast_test)
    rmse = np.sqrt(mean_squared_error(df_test, forecast_test))

    print(f'mae - manual: {mae}')
    print(f'mape - manual: {mape}')
    print(f'rmse - manual: {rmse}')