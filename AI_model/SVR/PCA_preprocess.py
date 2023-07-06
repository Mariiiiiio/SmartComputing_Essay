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
def  draw_graph(train_data, test_data , round_num):

    plt.subplot(3, 2, round_num)
    plt.title(f'PCA = {round_num}')
    plt.plot(range(1, 50), train_data, 'co-', label = f'train data', markersize=4)
    plt.plot(range(1, 50), test_data, 'go-', label = f'test data', markersize=4)
    plt.legend()
    plt.xlabel("C number")
    plt.ylabel("Value")



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
    pca_record = {}

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
        x_train, x_test, y_train, y_test = train_test_split(X_pca, data5, test_size=0.2, random_state=0)
        
        # -----------Result
        if i == 1:
            pca_result = pd.DataFrame(pca_model.components_,columns=data1_1.columns,index = ['PC-1'])
        elif i == 2:
            pca_result = pd.DataFrame(pca_model.components_,columns=data1_1.columns,index = ['PC-1','PC-2'])
        elif i == 3:
            pca_result = pd.DataFrame(pca_model.components_,columns=data1_1.columns,index = ['PC-1','PC-2', 'PC-3'])
        elif i == 4:
            pca_result = pd.DataFrame(pca_model.components_,columns=data1_1.columns,index = ['PC-1','PC-2', 'PC-3', 'PC-4'])
        elif i == 5:
            pca_result = pd.DataFrame(pca_model.components_,columns=data1_1.columns,index = ['PC-1','PC-2', 'PC-3', 'PC-4', 'PC-5'])
        elif i == 6:
            pca_result = pd.DataFrame(pca_model.components_,columns=data1_1.columns,index = ['PC-1','PC-2', 'PC-3', 'PC-4', 'PC-5', 'PC-6'])
        print(pca_result)
        # print('PCA Result =====')
        # print('PCA N_components : ', end='')
        # print(pca_model.n_components_)
        # print('PCA Ratio : ', end='')
        # print(pca_model.explained_variance_ratio_) 
        # print('PCA n_feature in:', end='')
        # print(pca_model.n_features_in_)
        # print('PCA feature name:', end='')
        # print(pca_model.feature_names_in_)
        # print('PCA singular values:', end='')
        # print(pca_model.singular_values_)

        pca_record[i] = {'PCA Compnoents':pca_model.components_,
                         'PCA N_components':pca_model.n_components_, 
                         'PCA Ratio' : pca_model.explained_variance_ratio_, 
                         'PCA n_feature' : pca_model.n_features_in_, 
                         'PCA feature_names':pca_model.feature_names_in_,
                        #  'PCA Result' : pca_result
                        #  'PCA singular values':pca_model.singular_values_
                        }
        
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
    print(pca_record)

    #--------------Model result
    print(param_record)
    plt.title('SVR + PCA ')
    plt.plot(range(1, 7), mse_fig, 'co-', label="Train Score")
    plt.xlabel('pca number')
    plt.ylabel('MSE value')
    plt.show()


    #show plot
