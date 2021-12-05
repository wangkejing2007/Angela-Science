import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import pandas as pd 

# population人口
N = 9.816e7 + 2 + 1
# simuation Time / Day
T = 460 # Till 2021/4/26

# susceptiable ratio
s = np.zeros([T])

# exposed ratio
e = np.zeros([T])

# infective ratio
i = np.zeros([T])

# remove ratio
r = np.zeros([T])

# contact rate
#lamda =  0.016397 # Till 2021/4/26
lamda =  0.452 # Till 2021/10/16

# recover rate
gamma = 0.00821

# exposed period
sigma = 1 / 14

# initial infective people
i[0] = 2 / N
s[0] = 9.816e7 / N
e[0] = 40.0 / N
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