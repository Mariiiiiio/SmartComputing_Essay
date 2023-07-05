from sklearn.svm import SVR
# from thundersvm import SVR

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as mse

import matplotlib.pyplot as plt
import numpy as np
import time
from pyearth import Earth
# sys.path.append('/Users/mariio/專題/論文專題/AI_model')  #for mac

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




start = time.time()
# Data loading
data1, data2, data1_1, data2_2, data3, data4, data5, data6 = data_col()
print(data1_1.head(10))

#Data Re-Organize

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


    
    
import numpy
# from pyearth import Earth
from matplotlib import pyplot

#Fit an Earth model
model = Earth(feature_importance_type='rss')
model.fit(data1_1,data5)


#Print the model
print(model.trace())
print(model.summary())
print(f'Equation : {model.coef_[0][1]} * x0 + {model.coef_[0][2]} * x1 + {model.coef_[0][3]}')
# print(model.summary_feature_importances(sort_by='rss'))



#Plot the model
y_hat = model.predict(data1_1)
print(y_hat)
# pyplot.figure()
# pyplot.plot(data1_1,data5,'r.')
# pyplot.plot(data1_1,y_hat,'b.')
# pyplot.xlabel('x_6')
# pyplot.ylabel('y')
# pyplot.title('Simple Earth Example')
# pyplot.show()