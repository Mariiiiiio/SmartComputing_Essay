from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as mse
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import sys

sys.path.append('/Users/mariio/專題/論文專題/AI_model')  #for mac
from data_process import data_col


# Documents : 
''' 
----------Data file 
        data1 : 原始值-變數
        data2 : 年增率-變數
        data1_1 : 原始值(不包含礦業與土石採取業)-變數
        data2_2 : 年增率(不包含礦業與土石採取業)-變數
        data3 : 原始值-目標
        data4 : 年增率-目標
        data5 : 原始值（不包含礦業與土石採取業) -目標
        data6 : 年增率（不包含礦業與土石採取業) -目標
----------PCA Result:




----------

'''

if __name__ == '__main__':
    start = time.time()
    #--------------Data Collection
    data1, data2, data1_1, data2_2, data3, data4, data5, data6 = data_col()

    # 原始值-變數-轉換矩陣型態
    data1_ar = np.array(data1)

    data1_1ar = np.array(data1_1)
    print(data1_1.columns)
    # print(data1_1ar)

    # 年增率-變數-轉換矩陣型態
    data2_ar = np.array(data2)
    data2_2ar = np.array(data2_2)
    # print(data2_2ar)

    #Target-setting-To array
    target_ori = np.array(data5)
    target_Year = np.array(data6)


    
    #--------------Model testing
    best_mse = []
    param_record = {}
    mse_fig = []


    for i in range(1, 7):
        
        #-----------container init
        mse_rec = 1000000
        count = 0
        r2_rec = 0



        print('-'*100+'Round('+str(i)+')')
        # -----------pca_model
        pca_model = PCA(n_components=i)
        pca_model.fit(data1_1)
        X_pca = pca_model.transform(data1_1)
        print('Data Shape :')
        print(f'-----Origin Data shape : {data1_1.shape}')
        print(f'-----PCA Data shape : {X_pca.shape}')


        # -----------Training & Testing Data prepare:
        x_train, x_test, y_train, y_test = train_test_split(X_pca, data5, test_size=0.2)
        
        # -----------Result
        # pca_result = pd.DataFrame(pca_model.components_,columns=data1_1.columns,index = ['PC-1','PC-2'])
        print('PCA Result =====')
        print('PCA N_components : ', end='')
        print(pca_model.n_components_)
        print('PCA Ratio : ', end='')
        print(pca_model.explained_variance_ratio_) 
        print('PCA n_feature in:', end='')
        print(pca_model.n_features_in_)
        print('PCA feature name:', end='')
        print(pca_model.feature_names_in_)
        print('PCA singular values:', end='')
        print(pca_model.singular_values_)

        # -----------SVR_model
        print('SVR Result =====')

        for j in range(1,29):
            print(f'C = {j} ..........')
            svr_model = SVR(C= 29,kernel='rbf', degree= 3, gamma='auto', max_iter=-1)
            svr_model.fit(x_train, y_train)
            y_hat = svr_model.predict(x_test)
            #Score showing
            print("Training  Score : ", svr_model.score(x_train,y_train))
            print("Testing  Score : ", svr_model.score(x_test, y_test))
            print("R^2 得分:", r2_score(y_test, y_hat))
            mse_score = mse(y_test, y_hat)
            print("MSE_Score : ", mse_score)
            print("RMSE_Score : ", np.sqrt(mse_score))
            if mse_score < mse_rec:
                mse_rec = mse_score
                count = j
                r2_rec = r2_score(y_test, y_hat)
        mse_fig.append(mse_rec)
        param_record[i] = {'C': j, 'best_mse_score': mse_rec, 'R2_score': r2_rec}



    end = time.time()
    print("執行時間：%f 秒" % (end - start))
    #--------------Model result
    print(param_record)
    plt.title('SVR + PCA ')
    plt.plot(len(mse_rec), mse_rec, 'co-', label="Train Score")
    plt.xlabel('pca number')
    plt.ylabel('MSE value')
    plt.show()


    #show plot
