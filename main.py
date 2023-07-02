
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from matplotlib.font_manager import FontProperties
import numpy as np
#Read data
data1 = pd.read_csv('OriginalValue.csv', encoding='cp950')
data2 = pd.read_csv('YearsRate.csv', encoding='cp950')


col = ['礦業及土石採取業', '製造業', '金屬機電工業', '資訊電子工業',
       '化學工業', '民生工業', '電力及燃氣供應業', '用水供應業']
target_col = ['總指數', '總指數(不含土石採取業)']
# data1.drop(index=0, inplace=True)
# print(data1.iloc[ 0: 10 , 0])



#Split the year from the data
data1_Orininal_year = data1.iloc[:, 0]
data2_Orininal_year = data2.iloc[:, 0]
data1.drop(' ', axis=1, inplace=True)
data2.drop(' ', axis=1, inplace=True)
figure_show_year = []
for i in range(0, len(data1_Orininal_year), 12):
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
# print(data1.head(10))
# print(data2.head(10))


x = range(0, len(data1_Orininal_year), 12)
labels = figure_show_year


# Original value figure
for i in range(len(col)):
    
    #fig setting
    plt.title(f'{col[i]} 原始值', fontsize = 20)
    plt.xlabel('Years', fontsize = 15)
    plt.ylabel('數值', fontsize= 15)
    plt.xticks(ticks=x,
          labels=labels,
          color='#08395c',
          fontsize=15,
          rotation=70)

    #draw figure
    plt.plot(range(497),data1[col[i]])
    # plt.plot(x, data1[col[i]], 'r:o')
    plt.ylim(0,300, 10) 
    plt.grid(True)
    plt.show()  


# Years rate figure
for i in range(len(col)):

    #fig setting
    plt.title(f'{col[i]} 年增率(%)', fontsize = 20)
    plt.xlabel('Years', fontsize = 15)
    plt.ylabel('年增長數值', fontsize= 15)
    plt.xticks(ticks=x,
          labels=labels,
          color='#9c5307',
          fontsize=15,
          rotation=70)

    #draw figure
    plt.plot(range(497),data2[col[i]])
    plt.ylim(-60, 200, 10) 
    plt.grid(True)
    plt.show()  

