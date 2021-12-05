import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model


#### Load Data ####
data = pd.read_csv('D:/Angela-Science/owid-covid-data.csv', sep = ',')
data = data[['id','new_cases']]
print('-'*30);print('HEAD');print('-'*30)
print(data.head())


#### Prepare Data ####
print('-'*30);print('Prepare Data');print('-'*30)
x = np.array(data['id']).reshape(-1, 1)
y = np.array(data['new_cases']).reshape(-1, 1)
plt.plot(y, '-m')
polyFeat = PolynomialFeatures(degree=3,interaction_only=True)
x = polyFeat.fit_transform(x)
#print(x)

#### 訓練資料 ####
print('-'*30);print('訓練資料');print('-'*30)
model = linear_model.LinearRegression()
model.fit(x,y)
accuracy = model.score(x,y)
print(f'準確率:{round(accuracy*100,2)} %')
y0 = model.predict(x)



#### 感染人數預測 ####
days = 1
print('-'*30);print('PREDICTION');print('-'*30)
print(f'感染人數預測 - Cases after {days} days :', end='')
print(round(int(model.predict(polyFeat.fit_transform([[286+days]])))/1,0),'人')

x1 = np.array(list(range(1,286+days))).reshape(-1,1)
y1 = model.predict(polyFeat.fit_transform(x1))
plt.plot(y1, '--r')
plt.plot(y0, '--b')
plt.show()