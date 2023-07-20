from sklearn.svm import SVR
# from thundersvm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as mse
import matplotlib.pyplot as plt
import numpy as np
import time
import sys  


# sys.path.append('/Users/mariio/專題/論文專題/AI_model')  #for mac

sys.path.append(r'C:\Users\USER\Desktop\University\Project\SmartComputing_Essay\AI_model') #for windows

from data_process import data_col

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






if __name__ == '__main__':

    start = time.time()
    # Data loading
    data1, data2, data1_1, data2_2, data3, data4, data5, data6 = data_col()

    # data1 = pd.read_csv('../../OriginalValue(329).csv',encoding='cp950')
    
    '''
    #Split the year from the data
    data1.drop(' ', axis=1, inplace=True)
    # print(data1.head(10))
    data1 = data1.astype('float64')
    #target set : 總指數 and 總指數(不含土石採取業)
    target_data1 = data1.iloc[:, 0]
    '''
    #Data Re-Organize
    # print(data2)
    # print(data3)
    
    # 原始值-變數-轉換矩陣型態
    data1_ar = np.array(data1)
    data1_1ar = np.array(data1_1)
    # print(data1_1ar)

    # 年增率-變數-轉換矩陣型態
    data2_ar = np.array(data2)
    data2_2ar = np.array(data2_2)
    # print(data2_2ar)

    #Target-setting-To array
    target_ori = np.array(data5)
    target_Year = np.array(data6)

    # print(data1.shape) 
    # print(data1_1ar.shape) 
    # print(data2_ar.shape) 
    # print(data2_2ar.shape) 
    # print(target_ori.shape) 
    # print(target_Year.shape) 

    #Data prepare
    x_train, x_test, y_train, y_test = train_test_split(data1_1, data5, test_size=0.2)
    

    #SVR model create
    '''
    print('-'*50+'SVR Started--')
    polyModel=SVR(C=6, kernel='rbf', degree= 7, gamma='auto', max_iter=-1, verbose=0)
    polyModel.fit(x_train, y_train)
    y_hat=polyModel.predict(x_test)
    '''
    
    '''
    #for finding the svr best score
    score_con_train = []
    score_con_test = []
    best_mse = 20
    count_i = 0
    mes_val = []
    R2_score = []
    for i in range(2, 50):
        print('-'*50+'SVR'+'_'+str(i)+' '+'Started')
        # for j in range(1, 7):
        polyModel=SVR(C=i, kernel='rbf', degree= 3, gamma='auto', max_iter=-1)
        polyModel.fit(x_train, y_train)
        y_hat=polyModel.predict(x_test)      
          
        #Score showing
        print("Training  Score : ", polyModel.score(x_train,y_train))
        print("Testing  Score : ", polyModel.score(x_test, y_test))
        print("R^2 得分:", r2_score(y_test, y_hat))
        mse_score = mse(y_test, y_hat)
        print("MSE_Score : ", mse_score)
        print("RMSE_Score : ", np.sqrt(mse_score))
        if mse_score < best_mse:
            best_mse = mse_score
            count_i = i
        score_con_train.append(polyModel.score(x_train,y_train))
        score_con_test.append(polyModel.score(x_test, y_test))
        mes_val.append(mse_score)
        R2_score.append(r2_score(y_test, y_hat))
        
    print('Score :')
    print('-'*50)
    print(f'The best C and best mse value : C = {count_i}, {best_mse}')
    print(f'R2 score : {R2_score}')
    print(f'Mse score : {mse_score}')
    print('-'*50)
    #draw score
    plt.plot(range(len(score_con_test)), score_con_test, 'go-', label="Test Score")
    plt.plot(range(len(score_con_test)), score_con_train, 'co-', label="Train Score")
    plt.legend()
    end = time.time()
    print("執行時間：%f 秒" % (end - start))
    plt.show()
    
    #draw score_lossfunction & r2
    plt.plot(range(len(mes_val)), mes_val, 'go-', label="Mse Score")
    plt.plot(range(len(R2_score)), R2_score, 'co-', label="R2 Score")
    plt.legend()
    plt.show()
    
    #draw prediction figure
    r = len(x_test) + 1
    plt.plot(np.arange(1,r), y_hat, 'go-', label="predict")
    plt.plot(np.arange(1,r), y_test, 'co-', label="real")
    plt.legend()
    plt.show()
    
    '''
    
    
    #GridSearch
    param = {'kernel' : ('rbf', 'sigmoid'),'C' : range(1, 29),'degree' : [3,8],'gamma' : ('auto','scale')}

    grid_search = GridSearchCV(estimator = SVR(), param_grid = param, 
                      cv = 3, n_jobs = -1, verbose = 2 )
    grid_search.fit(x_train, y_train)
    y_hat = grid_search.predict(x_test)

    #Score showing
    print("Training  Score : ", grid_search.score(x_train,y_train))
    print("Testing  Score : ", grid_search.score(x_test, y_test))

    print("R^2 得分:", r2_score(y_test, y_hat))
    mse_score = mse(y_test, y_hat)
    print("MSE_Score : ", mse_score)
    print("RMSE_Score : ", np.sqrt(mse_score))


    print(" Results from Grid Search " )
    print("\n The best estimator across ALL searched params:\n",grid_search.best_estimator_)
    print("\n The best score across ALL searched params:\n",grid_search.best_score_)
    print("\n The best parameters across ALL searched params:\n",grid_search.best_params_)
    
    #Draw the result graph
    r = len(x_test) + 1
    plt.plot(np.arange(1,r), y_hat, 'go-', label="predict")
    plt.plot(np.arange(1,r), y_test, 'co-', label="real")
    plt.legend()
    end = time.time()
    print("執行時間：%f 秒" % (end - start))
    plt.show()
    
    




