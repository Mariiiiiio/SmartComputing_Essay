import matplotlib.pyplot as plt
import numpy as np
import time
import sys  
from sklearn.decomposition import PCA
from sklearn import preprocessing
import pandas as pd

sys.path.append('/Users/mariio/專題/論文專題/AI_model')  #for mac

# sys.path.append(r'C:\Users\USER\Desktop\University\Project\SmartComputing_Essay\AI_model') #for windows

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
    data_column = ['金屬機電工業', '資訊電子工業',
        '化學工業', '民生工業', '電力及燃氣供應業', '用水供應業']

    #Data Re-Organize
    # print(data2)
    # print(data3)
    
    # 原始值-變數-轉換矩陣型態
    data1_ar = np.array(data1)
    
    # print(data1_1.head(10))
    # data1_1ar = np.array(data1_1.drop('製造業',axis=1))
    data1_1ar = np.array(data1_1)
    # print(data1_1ar)

    # 年增率-變數-轉換矩陣型態
    data2_ar = np.array(data2)
    data2_2ar = np.array(data2_2)
    # print(data2_2ar)

    #Target-setting-To array
    target_ori = np.array(data5)
    target_Year = np.array(data6)
    
    #PCA - Ver2
    from sklearn.preprocessing import StandardScaler as ss
    
    scaled_data = ss().fit_transform(data1_1ar)
    pca = PCA()
    X_pca = pca.fit_transform(scaled_data)
    X_cov = pca.get_covariance() 
    print(X_pca.shape)
    print(pca.components_)

    exp_var_ratio = pca.explained_variance_ratio_
    print(exp_var_ratio)

    plt.figure(figsize=(6, 4))
    plt.bar(range(6), exp_var_ratio, alpha=0.5, label='individual explained ratio')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


    PCnames = ['PC'+str(i+1) for i in range(pca.n_components_)]
    Loadings = pd.DataFrame(pca.components_,columns=PCnames,index=data1_1.columns)
    
    print(Loadings.iloc[:,:])
    # Loadings["PC1"].sort_values().plot.barh()
    # plt.show()

    # PCA_coll = pd.DataFrame(data=X_pca, index=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6'])
    # print(PCA_coll)
    # print(X_cov)






    # print(scaled_data)
    # print(scaled_data.shape)
    ''' Ver1 
    pca = PCA()
    pca.fit(scaled_data)
    pca_data = pca.transform(scaled_data)
    
    
    per_var = np.round(pca.explained_variance_ratio_*100, decimals=1)
    labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]
    plt.bar(x=range(1, len(per_var)+1), height=per_var, tick_label=labels)
    plt.ylabel('Percentage of Explained Variance')
    plt.xlabel('Principal Component')
    plt.title('PCA plot')
    plt.show()
    print(data1_1.columns)
    # pca_df = pd.DataFrame(pca_data,index=data1_1.columns,columns=labels)
    
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
    #PC1 EigenValue
    pc1 = [-0.44350436, -0.40031045, -0.45146493,  0.24391929, -0.44931558, -0.42172708]

    pc1_mi = []
    pc1_avg = []
    a = min(pc1)
    b = 0 - a + 0.1
    for i in pc1:
        pc1_mi.append(i+b)

    mean = np.mean(pc1_mi)
    for i in pc1_mi:
        # print(mean)
        pc1_avg.append((i/mean)*100)
        # print(pc1_avg)
    # labels_2 = ['Metal_industry', 'Electricity_industry', 'Chemical_industry', ' Consumer Goods Industries', 'Ele&fuels_supply', 'Water_supply']
    labels_2 = data_column
    print(pc1_avg)
    plt.plot(range(len(pc1_avg)), pc1_avg, 'co-')
    plt.title('EigenValue Ratio')
    # plt.xticks(range(len(pc1_avg)), labels=data1_1.drop('製造業',axis=1).columns)
    plt.ylabel('Ratio')
    plt.xlabel('Indicators')
    plt.xticks(ticks=range(len(pc1_avg)),
          labels=labels_2,
          color='#08395c',
          fontsize=10,
          rotation=70)

    plt.show()
    # print(pca.components_[1])
    # loading_scores = pd.Series(pca.components_[0], index=)
    '''