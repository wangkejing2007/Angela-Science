import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import pandas as pd 

# population
N = 8837544
# simuation Time / 模擬週期
T = 60
# susceptiable ratio / 易染者比率
s = np.zeros([T])
# infective ratio / 染病者比率
i = np.zeros([T])
# remove ratio / 移出者比率
r = np.zeros([T])

# contact rate / 接觸率
lamda = 1.20
# recover rate / (治癒+死亡)率
gamma = 0.608

# initial infective people / 期初染病人數
s[0] = (8837544-31) / N
i[0] = 31 / N
#r[0] = 1 / N

for t in range(T-1):
    i[t + 1] = i[t] + i[t] * lamda * s[t] - gamma*i[t]
    s[t + 1] = s[t] - lamda * s[t] * i[t]
    r[t + 1] = r[t] + gamma*i[t]

fig, ax = plt.subplots(figsize=(10,10))
ax.plot(s, c='b', lw=2, label='S')
ax.plot(i, c='r', lw=2, label='I')
ax.plot(r, c='g', lw=2, label='R')
ax.set_xlabel('Day/模擬週期', fontproperties = 'SimHei', fontsize=16)
ax.set_ylabel('Infective Ratio/感染率', fontproperties = 'SimHei', fontsize=16)
ax.grid(1)
plt.xticks((np.arange(0, 61, 5)),fontsize=20)
plt.yticks((np.arange(0, 1.1, 0.1)),fontsize=20)
plt.legend()

plt.show()