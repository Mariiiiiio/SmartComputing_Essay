
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
from matplotlib.font_manager import FontProperties


data1 = pd.read_csv('OriginalValue.csv', encoding='cp950')
data2 = pd.read_csv('YearsRate.csv', encoding='cp950')


col = ['總指數', '總指數(不含土石採取業)', '礦業及土石採取業', '製造業', '金屬機電工業', '資訊電子工業',
       '化學工業', '民生工業', '電力及燃氣供應業', '用水供應業']
print(data1.columns)
print(data2.columns)
# data1.drop(index=0, inplace=True)
# print(data1.iloc[ 0: 10 , 0])
data1_Orininal_year = data1.iloc[:, 0]
data2_Orininal_year = data2.iloc[:, 0]
# data1.drop(' ', axis=1, inplace=True)
# data2.drop(' ', axis=1, inplace=True)
print(data1)
print(data2)
# print(data1_Orininal_year)
# print(data2_Orininal_year)
plt.ylim(0,200, 10) 
# plt.xticks(data1_Orininal_year)
# sns.set_style("whitegrid",{"font.sans-serif":['Microsoft JhengHei']})
plt.rcParams['font.sans-serif']='SimHei'
for i in range(len(col)):
    plt.rcParams['font.sans-serif']='SimHei'
    plt.subplot(5, 2, i+1)
    # plt.plot(len(data1_Orininal_year),data1[col[i]])
    sns.lineplot(x=col[i], y=' ', data=data1, err_style=None)

plt.show()

# sns.pairplot(data1)
# plt.show()
