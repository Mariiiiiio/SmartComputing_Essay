import pandas as pd 
import numpy as np

def data_col():
    #data1 = pd.read_csv('/Users/mariio/專題/論文專題/OriginalValue.csv', encoding='cp950')
    #data2 = pd.read_csv('/Users/mariio/專題/論文專題/YearsRate.csv', encoding='cp950')

    data1 = pd.read_csv('/Users/mariio/專題/論文專題/OriginalValue.csv', encoding='cp950')
    data2 = pd.read_csv('/Users/mariio/專題/論文專題/YearsRate.csv', encoding='cp950')

    col = ['礦業及土石採取業', '製造業', '金屬機電工業', '資訊電子工業',
        '化學工業', '民生工業', '電力及燃氣供應業', '用水供應業']
    target_col = ['總指數', '總指數(不含土石採取業)']

    #Split the year from the data
    data1_Orininal_year = data1.iloc[:, 0]
    data2_Orininal_year = data2.iloc[:, 0]
    data1.drop(' ', axis=1, inplace=True)
    data2.drop(' ', axis=1, inplace=True)
    figure_show_year = []
    for i in range(0, len(data1_Orininal_year), 3):
        figure_show_year.append(data1_Orininal_year[i])
    #data re-organize
    for i in range(len(col)):

        for j in range(len(data1.iloc[:, 0])):
            if data1[col[i]][j] == '-':
                data1.replace(data1[col[i]][j], 0.0, inplace=True)
            if data2[col[i]][j] == '-':
                data2.replace(data2[col[i]][j], 0.0, inplace=True)
    data1 = data1.astype('float64')
    data2 = data2.astype('float64')


    #target set : 總指數 and 總指數(不含土石採取業)
    target_data1 = data1.iloc[:, 0]
    target_data2 = data2.iloc[:, 0]
    target_data1_without_one_ele = data1['總指數(不含土石採取業)']
    target_data2_without_one_ele = data2['總指數(不含土石採取業)']
    # print(target_data1.head(10))
    # print(target_data2.head(10))
    # print(target_data1_without_one_ele.head(10))
    # print(target_data2_without_one_ele.head(10))

    #target set : train value -> data except year and 總指數
    data1.drop(['總指數', '總指數(不含土石採取業)'], axis=1, inplace=True)
    data2.drop(['總指數', '總指數(不含土石採取業)'],axis=1, inplace=True)
    data1_1 = data1.drop('礦業及土石採取業', axis=1)
    data2_2 = data2.drop('礦業及土石採取業', axis=1)
    # print(data1_1)
    # print(data1.head(10))
    # print(data2.head(10))
    return data1, data2, data1_1, data2_2, target_data1, target_data2, target_data1_without_one_ele, target_data2_without_one_ele
    '''
    data1 : 原始值-變數
    data2 : 年增率-變數
    target_data1 : 原始值-目標
    target_data2 : 年增率-目標
    target_data1_without_one_ele : 原始值（不包含礦業與土石採取業) -目標
    target_data2_without_one_ele : 年增率（不包含礦業與土石採取業) -目標
    '''


if __name__ == '__main__':
    pass