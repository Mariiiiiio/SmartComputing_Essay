import matplotlib.pyplot as plt
import numpy as np
import time
import sys  
from sklearn.decomposition import PCA
from sklearn import preprocessing
import pandas as pd

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


    #Data Re-Organize
    # print(data2)
    # print(data3)
    
    # 原始值-變數-轉換矩陣型態
    data1_ar = np.array(data1)
    
    
    data1_1ar = np.array(data1_1.drop('製造業',axis=1))
    # print(data1_1ar)

    # 年增率-變數-轉換矩陣型態
    data2_ar = np.array(data2)
    data2_2ar = np.array(data2_2)
    # print(data2_2ar)

    #Target-setting-To array
    target_ori = np.array(data5)
    target_Year = np.array(data6)
    
    #PCA
    scaled_data = preprocessing.scale(data1_1ar)
    print(scaled_data)
    
    pca = PCA()
    pca.fit(scaled_data)
    pca_data = pca.transform(scaled_data)
    
    
    per_var = np.round(pca.explained_variance_ratio_*100, decimals=1)
    labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]
    # plt.bar(x=range(1, len(per_var)+1), height=per_var, tick_label=labels)
    plt.ylabel('Percentage of Explained Variance')
    plt.xlabel('Principal Component')
    plt.title('PCA plot')
    # plt.show()
    # print(data1_1.columns)
    # pca_df = pd.DataFrame(pca_data,columns=labels)
    
    # plt.scatter(pca_df.PC1, pca_df.PC2)
    # plt.title('PCA Graph')
    # plt.xlabel('PC1 - {0}%'.format(per_var[0]))
    # plt.ylabel('PC2 - {0}%'.format(per_var[1]))
    
    # for sample in pca_df.index:
    #     plt.annotate(sample, (pca_df.loc[sample], pca_df.PC2.loc[sample]))
    # plt.show()
    print('-'*50)
    for i in range(6):
        print(pca.components_[i])
        
    pc1 = [-0.44350436, -0.40031045, -0.45146493,  0.24391929, -0.44931558, -0.42172708]
    pc1_mi = []
    pc1_avg = []
    a = min(pc1)
    b = 0 - a
    
    for i in pc1:
        pc1_mi.append(i+b)
    for i in pc1:
        mean = np.mean(pc1_mi)
        # print(mean)
        pc1_avg.append((i/mean)*100)
    print(data1_1.drop('製造業',axis=1).columns)
    labels_2 = ['Metal_industry', 'Electricity_industry', 'Chemical_industry', ' Consumer Goods Industries', 'Ele&fuels_supply', 'Water_supply']
    plt.plot(range(len(pc1_avg)), pc1_avg, 'co-')
    # plt.xticks(range(len(pc1_avg)), labels=data1_1.drop('製造業',axis=1).columns)
    plt.xticks(ticks=range(len(pc1_avg)),
          labels=labels_2,
          color='#08395c',
          fontsize=10,
          rotation=70)
    plt.show()
    # print(pca.components_[1])
    # loading_scores = pd.Series(pca.components_[0], index=)