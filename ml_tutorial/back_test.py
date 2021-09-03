import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp
import math
# 含み損によるゼロカットをシミュレーションしていないので注意


def backtest(cl=None, hi=None, lo=None, pips=None,
             buy_entry=None, sell_entry=None,
             buy_cost=None, sell_cost=None
             ):
    n = cl.size
    y = cl.copy() * 0.0
    poss = cl.copy() * 0.0
    ret = 0.0
    pos = 0.0
    for i in range(n):
        prev_pos = pos

        # exit
        if buy_cost[i]:
            vol = np.maximum(0, -prev_pos)
            ret -= buy_cost[i] * vol
            pos += vol

        if sell_cost[i]:
            vol = np.maximum(0, prev_pos)
            ret -= sell_cost[i] * vol
            pos -= vol

        # entry
        if buy_entry[i] and buy_cost[i]:
            vol = np.minimum(1.0, 1 - prev_pos) * buy_entry[i]
            ret -= buy_cost[i] * vol
            pos += vol

        if sell_entry[i] and sell_cost[i]:
            vol = np.minimum(1.0, prev_pos + 1) * sell_entry[i]
            ret -= sell_cost[i] * vol
            pos -= vol

        if i + 1 < n:
            ret += pos * (cl[i + 1] / cl[i] - 1)

        y[i] = ret
        poss[i] = pos
    return y, poss


df = pd.read_pickle('df_fit.pkl')

df['cum_ret'], df['poss'] = backtest(
    cl=df['cl'].values,
    buy_entry=df['y_pred_buy'].values > 0,
    sell_entry=df['y_pred_sell'].values > 0,
    buy_cost=df['buy_cost'].values,
    sell_cost=df['sell_cost'].values,
)

df['cum_ret'].plot()
plt.title('cum_ret')
plt.show()

df['poss'].plot()
plt.title('position')
plt.show()

df['poss'].diff(1).abs().dropna().cumsum().plot()
plt.title('trade count')
plt.show()

print('t検定')

x = df['cum_ret'].diff(1).dropna()
print(ttest_1samp(x, 0))

print('p平均法 https://note.com/btcml/n/n0d9575882640')


def calc_p_mean(x, n):
    ps = []
    for i in range(n):
        x2 = x[i * x.size // n:(i + 1) * x.size // n]
        if np.std(x2) == 0:
            ps.append(1)
        else:
            t, p = ttest_1samp(x2, 0)
            if t > 0:
                ps.append(p)
            else:
                ps.append(1)
    return np.mean(ps)


def calc_p_mean_type1_error_rate(p_mean, n):
    return (p_mean * n) ** n / math.factorial(n)


x = df['cum_ret'].diff(1).dropna()
p_mean_n = 5
p_mean = calc_p_mean(x, p_mean_n)
print('p mean {}'.format(p_mean))
print('error rate {}'.format(calc_p_mean_type1_error_rate(p_mean, p_mean_n)))
print(
    'error rateが十分小さくないと有意ではない。\
    何度も試行錯誤することを考えると1e-5以下くらい'
    )