from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt
import numpy as np
import time
import sys  
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as mse

sys.path.append('/Users/mariio/專題/論文專題/AI_model')  #for mac

# sys.path.append(r'C:\Users\USER\Desktop\University\Project\SmartComputing_Essay\AI_model') #for windows

from data_process import data_col, data_col_329

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

def  draw_graph(train_data, test_data , round_num):

    plt.subplot(2, 4, round_num)
    plt.title(f'ICA = {round_num}')
    plt.plot(range(1, 50), train_data, 'co-', label = f'train data', markersize=4)
    plt.plot(range(1, 50), test_data, 'go-', label = f'test data', markersize=4)
    plt.legend()
    plt.xlabel("C number")
    plt.ylabel("Value")

def Call_Model(data, target):
    start = time.time()
    
   
    
    #StandardScaler 
    from sklearn.preprocessing import StandardScaler as ss
    
    scaled_data = ss().fit_transform(data)

    best_mse = []
    param_record = {}
    mse_fig = []
    Ica_record = {}

    # print(scaled_data.shape)
    # for i in range(1, len(data.columns)+1):
    for i in range(1, 7):
        fast_ica = FastICA(n_components=i)
        S_ = fast_ica.fit(scaled_data).fit_transform(scaled_data)
        x_train, x_test, y_train, y_test = train_test_split(S_, target, test_size=0.2, random_state=1)

        mse_rec = 1000000
        count = 0
        r2_rec = 0

        # -----------SVR_model
        print('SVR Result =====')
        test_sc = []
        train_sc = []
        test_sc_num = 0
        train_sc_num = 0
        for j in range(1,50):

            # print(f'C = {j} ..........')
            svr_model = SVR(C= j,kernel='poly', degree= i, gamma='auto', max_iter=-1)
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
                test_sc_num = svr_model.score(x_test, y_test)
                train_sc_num = svr_model.score(x_train,y_train)
            train_sc.append(svr_model.score(x_train, y_train))
            test_sc.append(svr_model.score(x_test, y_test))

        draw_graph(train_sc, test_sc, i)
        mse_fig.append(mse_rec)
        param_record[i] = {'C': j, 
                           'best_mse_score': mse_rec, 
                           'R2_score': r2_rec, 
                           'Training score' : train_sc_num, 
                           'Testing score' : test_sc_num}
        
    # plt.plot(range(len(train_sc)), train_sc, 'go-', label="Train Score")
    # plt.plot(range(len(test_sc)), test_sc, 'co-', label="Test Score")


    end = time.time()
    print("執行時間：%f 秒" % (end - start))
    
    plt.subplots_adjust(left=0.125,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.2, 
                    hspace=0.35)
    plt.show()
    #PCA result
    print(Ica_record)

    #--------------Model result

    print(param_record)
    plt.title('SVR + ICA ')
    plt.plot(range(1, len(data.columns)), mse_fig, 'co-', label="Train Score")
    plt.plot(range(1, len(data.columns)), mse_fig, 'co-', label="Train Score")
    plt.xlabel('ICA number')
    plt.ylabel('MSE value')
    plt.show()


def Call_497data():

    # Data loading
    data1, data2, data1_1, data2_2, data3, data4, data5, data6 = data_col()
    
    data_column = ['金屬機電工業', '資訊電子工業', '化學工業', '民生工業', '電力及燃氣供應業', '用水供應業']

    # 原始值-變數-轉換矩陣型態
    data1_ar = np.array(data1)
    data1_1ar = np.array(data1_1)

    # 年增率-變數-轉換矩陣型態
    #Target-setting-To array
    target_ori = np.array(data5)

    Call_Model(data1_1, target_ori)


def Call_329data():
    data, target = data_col_329()
    print(f'Data number : {data.shape}, target number : {target.shape}')

    Call_Model(data, target)


if __name__ == '__main__':
    print('-'*50+'329')
    # Call_329data()


    Call_497data()




    



        

