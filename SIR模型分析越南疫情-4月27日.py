# 使用SIR模型对2019新型冠状病毒的疫情发展进行分析
# https://cloud.tencent.com/developer/article/1618838?from=14588

import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np

plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体

def linear_regression(x, y):
    N = len(x)
    sumx = sum(x)
    sumy = sum(y)
    sumx2 = sum(x ** 2)
    sumxy = sum(x * y)
    A = np.mat([[N, sumx], [sumx, sumx2]])
    b = np.array([sumy, sumxy])
    return np.linalg.solve(A, b)

data = pd.read_csv('D:/Angela-Science/owid-covid-data-4-27.csv', encoding='utf-8')
I = list(data['new_cases'])
N =98540000
Day = []
logI = []
for i in range(len(I)):
    Day.append(i+1)
    logI.append(math.log(I[i]))


X1=np.array(Day)
Y1=np.array(logI)
a0, a1 = linear_regression(X1, Y1)
_Y1 = [a0 + a1 * x for x in Day]

ax1 = plt.subplot(1,2,1)
ax2 = plt.subplot(1,2,2)
ax1.set_xlabel('模擬週期', fontsize=14)
ax1.set_ylabel('感染人數', fontsize=16)
ax2.set_xlabel('模擬週期', fontsize=14)
ax2.set_ylabel('感染人數', fontsize=16)

plt.sca(ax1)
plt.scatter(Day,I, marker = 'x', s = 10,color='blue', label='實際確診')
plt.yticks((np.arange(0, 20000, 1000)),fontsize=10)

plt.sca(ax2)
plt.scatter(Day,logI, marker = '+', s = 10,color='gray', label='logarithm/對數值')

plt.yticks([])

plt.plot(Day, _Y1,color='red', label='simulation/擬合走勢')

plt.legend()
plt.show()