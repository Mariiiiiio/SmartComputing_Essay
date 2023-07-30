from sklearn.decomposition import PCA
from sklearn.svm import SVR
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

plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False

sys.path.append(r'C:\Users\USER\Desktop\University\Project\SmartComputing_Essay\AI_model') #for windows
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

----------

'''
def  draw_graph(train_data, test_data , round_num):

    plt.subplot(4, 2, round_num)
    plt.title(f'PCA = {round_num}')
    plt.plot(range(1, 50), train_data, 'co-', label = f'train data', markersize=4)
    plt.plot(range(1, 50), test_data, 'go-', label = f'test data', markersize=4)
    plt.legend()
    plt.xlabel("C number")
    plt.ylabel("Testing & Training Score ")
    
def Call_PCA_graph(data, original):
    pca = PCA()
    X_pca = pca.fit_transform(data)
    PCnames = ['PC'+str(i+1) for i in range(pca.n_components_)]
    Loadings = pd.DataFrame(data = pca.components_, columns=PCnames,index=original.columns)
    print(Loadings.iloc[:,:])
    exp_var_ratio = pca.explained_variance_ratio_
    
    print(exp_var_ratio.shape)
    #Draw Graph
    plt.figure(figsize=(6, 4))
    plt.bar(range(1, len(original.columns)+1), exp_var_ratio, alpha=0.5, label='individual explained ratio')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    
    Loadings["PC1"].sort_values().plot.barh()
    plt.show()
    
    pass 

def Call_Model_PCA(data, target):
    start = time.time()
    #--------------Model testing
    best_mse = []
    param_record = {}
    mse_fig = []
    pca_record = {}
    
    #Data preprocessing
    from sklearn.preprocessing import StandardScaler as ss
    scaled_data = ss().fit_transform(data)
    Call_PCA_graph(scaled_data, data)
    
    
    for i in range(1, len(data.columns)+1 ):
        
        #-----------container init
        mse_rec = 1000000
        count = 0
        r2_rec = 0
        


        print('-'*100+'Round('+str(i)+')')
        # -----------pca_model
        
    
        
        pca_model = PCA(n_components=i)
        X_pca = pca_model.fit_transform(scaled_data)
        # print('Data Shape :')
        # print(f'-----Origin Data shape : {data.shape}')
        # print(f'-----PCA Data shape : {X_pca.shape}')
        

        # -----------Training & Testing Data prepare:
        '''
        ----------DataSet split
        Data : 497
            >>> training set : 397
                >>> Date : 1982 M1 ~ 2014 M12
            >>> testing set : 100
                >>> Date : 2015 M1 ~ 2023 M4

        Data : 329
            >>> training set : 276
                >>> Date : 1996 M1 ~ 2018 M12
            >>> testing set : 52
                >>> Date : 2019 M1 ~ 2023 M4
        '''
        n = 276 # Number of Training Data
        x_train = X_pca[:][:n].copy()
        x_test = X_pca[:][n:].copy()
    
        y_train = target[:][:n].copy()
        y_test = target[:][n:].copy()

        print(f'X training data : {x_train.shape},\n x testing data : {x_test.shape}, \n y training data : {y_train.shape}, \n y testing data : {y_test.shape} ')
        
        # -----------Result
        '''
        if i == 1:
            pca_result = pd.DataFrame(pca_model.components_,columns=data.columns,index = ['PC-1'])
        elif i == 2:
            pca_result = pd.DataFrame(pca_model.components_,columns=data.columns,index = ['PC-1','PC-2'])
        elif i == 3:
            pca_result = pd.DataFrame(pca_model.components_,columns=data.columns,index = ['PC-1','PC-2', 'PC-3'])
        elif i == 4:
            pca_result = pd.DataFrame(pca_model.components_,columns=data.columns,index = ['PC-1','PC-2', 'PC-3', 'PC-4'])
        elif i == 5:
            pca_result = pd.DataFrame(pca_model.components_,columns=data.columns,index = ['PC-1','PC-2', 'PC-3', 'PC-4', 'PC-5'])
        elif i == 6:
            pca_result = pd.DataFrame(pca_model.components_,columns=data.columns,index = ['PC-1','PC-2', 'PC-3', 'PC-4', 'PC-5', 'PC-6'])
        elif i == 7:
            pca_result = pd.DataFrame(pca_model.components_,columns=data.columns,index = ['PC-1','PC-2', 'PC-3', 'PC-4', 'PC-5', 'PC-6', 'PC-7'])
        print(pca_result)
        '''
        
        '''
        pca_record[i] = {'PCA Compnoents':pca_model.components_,
                         'PCA N_components':pca_model.n_components_, 
                         'PCA Ratio' : pca_model.explained_variance_ratio_, 
                         'PCA n_feature' : pca_model.n_features_in_, 
                        #  'PCA feature_names':pca_model.feature_names_in_,
                        #  'PCA Result' : pca_result
                        #  'PCA singular values':pca_model.singular_values_
                        }
        '''
        
        

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
        draw_graph(train_sc, test_sc, i)
        mse_fig.append(math.sqrt(mse_rec))
        # param_record[i] = {'C': count, 
        #                     'best_mse_score': mse_rec, 
        #                     'R2_score': r2_rec, 
        #                     'Training score' : train_sc_num, 
        #                     'Testing score' : test_sc_num}
        param_record[i] = {            
                'C': count,
                'best_rmse_score': math.sqrt(mse_rec), 
                'MAE Score' :  MAE_score,
                'MAPE Score' : MAPE_score,
            }
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
    # print(pca_record)

    #--------------Model result
    print(param_record)
    plt.title('SVR + PCA : 資料集A2')
    plt.plot(range(1, len(data.columns)+1), mse_fig, 'co-', label="Train Score")
    plt.xlabel('pca number')
    plt.ylabel('RMSE value')
    plt.show()


    #show plot



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
    # print(print(data1_1.head(10)))
    data1_1.drop('製造業', axis=1, inplace=True)
    # print(data1_1.columns)
    print(f'Data number : {data1_1.shape}, target number : {target_ori.shape}')
    Call_Model_PCA(data1_1, target_ori)


def Call_329data():
    data1 = pd.read_csv('..\..\OriginalValue(329).csv',encoding='cp950')
    
    

    #Split the year from the data
    data1_Orininal_year = data1.iloc[:, 0]
    data1.drop(' ', axis=1, inplace=True)
    print(data1.head(10))
    data1 = data1.astype('float64')



    #target set : 總指數 and 總指數(不含土石採取業)
    target_data1 = data1.iloc[:, 0]
    #target set : train value -> data except year and 總指數
    
    data1.drop(['總指數', '總指數(不含土石採取業)', '製造業'], axis=1, inplace=True)
    print(data1.columns)
    print(f'Data number : {data1.shape}, target number : {target_data1.shape}')
    # print(data1.head(10))
    # print(target_data1.head(10))
    Call_Model_PCA(data1, target_data1)
if __name__ == '__main__':
    print('-'*50+'329')
    Call_329data()
    
    # print('-'*50+'497')
    # Call_497data()
    
    
