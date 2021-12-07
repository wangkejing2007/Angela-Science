import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

#### Load Data ####
data = pd.read_csv('D:/Angela-Science/owid-covid-data-4-27.csv', sep = ',')
data = data[['id','new_cases']]

#### Prepare Data ####
x = np.array(data['id']).reshape(-1, 1)
y = np.array(data['new_cases']).reshape(-1, 1)
plt.plot(y, '-m')
polyFeat = PolynomialFeatures(degree=6, interaction_only=False) #設定degree
x = polyFeat.fit_transform(x)

#### 訓練資料(線性迴歸) ####
print('-'*45)
model = linear_model.LinearRegression()
model.fit(x,y)
accuracy = model.score(x,y)
print(f'準確率:{round(accuracy*100,2)} %')
y0 = model.predict(x)

#### 感染人數預測 ####
days = 10
print('-'*45)
print(f'感染人數預測 - Cases after {days} days :', end='')
print(round(int(model.predict(polyFeat.fit_transform([[224+days]])))),'人')
print('-'*45)

x1 = np.array(list(range(1,224+days))).reshape(-1,1)
y1 = model.predict(polyFeat.fit_transform(x1))
plt.plot(y1, '--r')
plt.plot(y0, '--b')

plt.xlabel('疫情天數', fontproperties = 'SimHei', fontsize=14)
plt.ylabel('感染人數', fontproperties = 'SimHei', fontsize=14)

plt.xticks((np.arange(0, 260, 10)),fontsize=10)
plt.yticks((np.arange(0, 18000, 1000)),fontsize=12)

plt.show()