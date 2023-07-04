from sklearn.svm import SVR
import sys  
sys.path.append('/Users/mariio/專題/論文專題/AI_model') 

from data_process import data_col


if __name__ == '__main__':

    data1, data2, data3, data4, data5, data6 = data_col()
    '''
    data1 : 原始值-變數
    data2 : 年增率-變數
    data3 : 原始值-目標
    data4 : 年增率-目標
    data5 : 原始值（不包含礦業與土石採取業) -目標
    data6 : 年增率（不包含礦業與土石採取業) -目標
    '''
    print(data3)

    # print(data1, data2)

    # polyModel=SVR(C=6, kernel='poly', degree=3, gamma='auto')




