import pandas as pd
import numpy as np

# 導入函數
from scipy.optimize import minimize
from scipy.integrate import odeint

# 我們定義SIR模型函數的名稱為SIR
# S：易感者人數，I：感染者人數，R：恢復者人數
# 我們用I/N表示易感者與感染者接觸的概率，β(beta)表示易感者與感染者接觸後被感染的概率，γ(gamma)表示感染者恢復的速率。

def SIR(y,t,beta,gamma):
    S,I,R = y
    dSdt = -S*(I/(S+I+R))*beta
    dIdt = beta*S*I/(S+I+R)-gamma*I
    dRdt = gamma*I
    return [dSdt,dIdt,dRdt]

def loss(parameters, infectious, recovered, y0):
    # 確定訓練模型的天數
    size = len(infectious)
    # 設置時間跨度
    t = np.linspace(1,size,size)
    beta, gamma = parameters
    # 計算預測值
    solution = odeint(SIR, y0, t, args=(beta, gamma))
    # 計算每日的感染者人數的預測值和真實值的均方誤差
    l1 = np.mean((solution[:,1] - infectious)**2)
    # 計算每日的治癒者人數的預測值和真實值之間的均方誤差
    l2 = np.mean((solution[:,2] - recovered)**2)
    # 返回SIR模型的損失值
    return l1+l2

# 讀取所有國家的疫情數據
data = pd.read_csv('D:/Angela-Science/vietnam-covid-data.csv')
# 挑選出其中關於越南的疫情數據
italy = data[data['location']=='Vietnam']
italy.head()
print(italy.head())

# 截取1月31日至3月15日之間的意大利疫情數據
italy_train = italy.set_index('date').loc['2020-01-23':'2021-08-31']
# 確定訓練集每天的感染者人數
infectious_train = italy_train['total_cases'] - italy_train['total_deaths']
# 與建立SIR模型時相類似，這裡我們也選取每天的康復者和死亡者作為SIR模型的恢復者
recovered_train = italy_train['total_deaths']

# 設置總人口N = 98000000
N =98000000
# 確定訓練集每天的易感者人數
susceptible_train = N - recovered_train - infectious_train

# 截取3月16日至4月3日之間的意大利疫情數據
italy_valid = italy.set_index('date').loc['2021-09-01':'2021-10-10']
# 確定驗證集的每天的感染者人數
infectious_valid = italy_valid['total_cases'] - italy_valid['total_deaths']
# 確定驗證集的每天的治癒者人數
recovered_valid = italy_valid['total_deaths']
# 因為我們的損失函數中只包含I(t)和R(t),所以在驗證集中，我們不再計算易感者人數

# 模型初始值
I0 = 2
R0 = 0
S0 = N - I0 - R0
y0 = [S0,I0,R0]

# 訓練模型
optimal = minimize(loss,[0.01,0.01], args=(infectious_train,recovered_train,y0), method='L-BFGS-B', bounds=[(0.000001, 1), (0.000001, 1)])
beta,gamma = optimal.x

# 輸出beta、gamma值
print([beta,gamma])

