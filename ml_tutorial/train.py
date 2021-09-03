import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import KFold
from config import cnf_dict


def calc_force_entry_price(entry_price=None, lo=None, pips=None):
    y = entry_price.copy()
    y[:] = np.nan
    for i in range(entry_price.size):
        for j in range(i + 1, entry_price.size):
            if round(lo[j] / pips) < round(entry_price[j - 1] / pips):
                y[i] = entry_price[j - 1]
                break
    return y


def add_target(df):
    # FTX BTC-PERPだとpipsが時期で変化するので(0.25〜1くらい)、
    # 約定シミュレーションは小さい単位でやって、指値計算は大きい単位でやる
    min_pips = 0.001
    max_pips = 1

    # limit_price_dist = max_pips
    limit_price_dist = df['ATR'] * 0.5
    limit_price_dist = np.maximum(
        1, (limit_price_dist / max_pips).round().fillna(1)) * max_pips

    df['buy_price'] = df['cl'] - limit_price_dist
    df['sell_price'] = df['cl'] + limit_price_dist

    df['buy_fep'] = calc_force_entry_price(
        entry_price=df['buy_price'].values,
        lo=df['lo'].values,
        pips=min_pips,
    )

    # calc_force_entry_priceは入力をマイナスにすれば売りに使える
    df['sell_fep'] = -calc_force_entry_price(
        entry_price=-df['sell_price'].values,
        lo=-df['hi'].values,  # 売りのときは高値
        pips=min_pips,
    )

    horizon = 1
    fee = 0.0

    df['y_buy_profit_rate'] = np.where(
        (df['buy_price'] / min_pips).round() > (
            df['lo'].shift(-1) / min_pips).round(),
        df['sell_fep'].shift(-horizon)/df['buy_price']-1-2*fee,
        0
    )
    df['y_sell_profit_rate'] = np.where(
        (df['sell_price'] / min_pips).round() < (
            df['hi'].shift(-1) / min_pips).round(),
        -(df['buy_fep'].shift(-horizon)/df['sell_price']-1)-2*fee,
        0
    )

    # バックテストで利用
    df['buy_cost'] = np.where(
        (df['buy_price'] / min_pips).round() > (
            df['lo'].shift(-1) / min_pips).round(),
        df['buy_price'] / df['cl'] - 1 + fee,
        0
    )
    df['sell_cost'] = np.where(
        (df['sell_price'] / min_pips).round() < (
            df['hi'].shift(-1) / min_pips).round(),
        -(df['sell_price'] / df['cl'] - 1) + fee,
        0
    )
    return df


df = pd.read_pickle('df_features.pkl')
df = add_target(df)

# df['y_buy_profit_rate'].cumsum().plot()
# df['y_sell_profit_rate'].cumsum().plot()

df.to_pickle('df_y.pkl')


################################################################################################
# 学習 + CV

df = pd.read_pickle('df_y.pkl')
df = df.dropna()


model = lgb.LGBMRegressor(n_jobs=-1, random_state=1)


# 厳密にはどちらもリーク(yに未来の区間のデータが含まれている)しているので注意

cv_indicies = KFold().split(df)
# ウォークフォワード法
# cv_indicies = TimeSeriesSplit().split(df)

for train_idx2, val_idx2 in cv_indicies:
    train_idx = df.index[train_idx2]
    val_idx = df.index[val_idx2]

    model.fit(
        df.loc[train_idx, cnf_dict['features']],
        df.loc[train_idx, 'y_buy_profit_rate']
    )
    df.loc[val_idx, 'y_pred_buy_profit_rate'] = model.predict(
        df.loc[val_idx, cnf_dict['features']])

    model.fit(
        df.loc[train_idx, cnf_dict['features']],
        df.loc[train_idx, 'y_sell_profit_rate']
    )
    df.loc[val_idx, 'y_pred_sell_profit_rate'] = model.predict(
        df.loc[val_idx, cnf_dict['features']])

df = df.dropna()

df[df['y_pred_buy_profit_rate'] > 0]['y_buy_profit_rate'].cumsum().plot()
df[df['y_pred_sell_profit_rate'] > 0]['y_sell_profit_rate'].cumsum().plot()
plt.show()

df.to_pickle('df_fit.pkl')
