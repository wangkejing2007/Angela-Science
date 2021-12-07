#   我們定義函數的名稱為SIR
def SIR(y,t,beta,gamma):
    S,I,R = y
    dSdt = -S*(I/(S+I+R))*beta
    dIdt = beta*S*I/(S+I+R)-gamma*I
    dRdt = gamma*I
    return [dSdt,dIdt,dRdt]

# 設置人群總人數為N
N = 98540000
# 設置初始時的感染人數I0為239
I0 = 239
# 設置初始時的恢復人數R0為31
R0 = 31
# 所以，初始易感者人群人數 = 總人數 - 初始感染人數 - 初始治癒人數
S0 = N - I0 - R0
# 設置初始值
y0 = [S0, I0, R0]

import numpy as np

# 設置估計疫情的時間跨度為60天
t = np.linspace(1,60,60)

# 設置beta值等於0.125
beta = 0.125

# 設置gamma的值等於0.05
gamma = 0.05

from scipy.integrate import odeint

# 求解
solution = odeint(SIR, y0, t, args = (beta, gamma))

# 要求Python的所有輸出不用科學計數法表示
np.set_printoptions(suppress=True)

# 輸出結果的前四行進行查看
solution[0:4,0:3]

import matplotlib.pyplot as plt
# 繪圖展示
#%matplotlib inline

fig, ax = plt.subplots(facecolor='w', dpi=100)

for data, color, label_name in zip([solution[:,1], solution[:,2]], ['r', 'g'], ['infectious', 'recovered']):
    ax.plot(t, data, color, alpha=0.5, lw=2, label=label_name)

ax.set_xlabel('Time/days')
ax.set_ylabel('Number')
ax.legend()
ax.grid(axis='y')
plt.box(False)

plt.show()

# 觀察的時間週期擴充為360天
t = np.linspace(1,360,360)

solution = odeint(SIR, y0, t, args = (beta, gamma))
fig, ax = plt.subplots(facecolor='w', dpi=100)

for index, color, label_name in zip(range(3), ['b','r','g'], ['susceptible', 'infectious', 'recovered']):
    ax.plot(t, solution[:, index], color, alpha=0.5, lw=2, label=label_name)

ax.set_xlabel('Time/days')
ax.set_ylabel('Number')
ax.legend()
ax.grid(axis='y')
plt.box(False)

plt.show()