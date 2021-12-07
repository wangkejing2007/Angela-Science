import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import pandas as pd 

# population (胡志明市人口數)
N = 8837544

# simuation Time / Day (模擬疫情發展週期)
T = 176 # Till 2021/10/16

# susceptiable ratio (易受感染人數比例)
s = np.zeros([T])

# exposed ratio
e = np.zeros([T])

# infective ratio (已受感染人數比例)
i = np.zeros([T])

# remove ratio (已復原移出人數比例)
r = np.zeros([T])

# contact rate (患者接觸率)
lamda = 0.3 #(每個患者每天的接觸感染)

# recover rate (患者復原率)
gamma = 0.0821

# exposed period (病毒潛伏期)
sigma = 1 / 5

# initial infective people (初始人數)
i[0] = 1.0 / N
s[0] = 8837544 / N
e[0] = 1.0 / N
for t in range(T-1):
    s[t + 1] = s[t] - lamda * s[t] * i[t]
    e[t + 1] = e[t] + lamda * s[t] * i[t] - sigma * e[t]
    i[t + 1] = i[t] + sigma * e[t] - gamma * i[t]
    r[t + 1] = r[t] + gamma * i[t]

fig, ax = plt.subplots(figsize=(10,6))
ax.plot(s, c='b', lw=2, label='S')
ax.plot(e, c='orange', lw=2, label='E')
ax.plot(i, c='r', lw=2, label='I')
ax.plot(r, c='g', lw=2, label='R')
ax.set_xlabel('Day',fontsize=20)
ax.set_ylabel('Infective Ratio', fontsize=20)
ax.grid(1)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend();

plt.show()