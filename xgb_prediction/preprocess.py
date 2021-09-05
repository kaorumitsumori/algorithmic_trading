from pandas.core.algorithms import value_counts
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ex = 'bitflyer'
pair = 'btcjpy'
periods_hrs = 1
periods = int(3600 * periods_hrs)
after = 1514764800
URL = f'https://api.cryptowat.ch/markets/{ex}/{pair}/ohlc?periods={periods}&after={after}'


cols = [
    'CloseTime',
    'OpenPrice',
    'HighPrice',
    'LowPrice',
    'ClosePrice',
    'Volume',
    'QuoteVolume',
]
edge = 0.02

def generate_dataset():
    data = requests.get(URL).json()
    past_data = data['result'][f'{periods}']
    past_data_df = pd.DataFrame(past_data, columns=cols)
    past_data_df['CloseTime'] = pd.to_datetime(
        past_data_df['CloseTime'], unit='s')
    past_data_df.set_index('CloseTime', inplace=True)
    return past_data_df


def add_past_cl_pct_chg(df):
    for dist in range(1, 11):
        df[f'cl_pct_chg_past_{int(60*periods_hrs*dist)}_mins'] = \
            df['ClosePrice'].pct_change(dist)
    return df


def add_target(df):
    df[f'cl_pct_chg_coming_{int(60*periods_hrs)}_mins'] = \
        df[f'cl_pct_chg_past_{int(60*periods_hrs)}_mins'].shift(-1)
    df['alpha_occurrence'] = \
        df[f'cl_pct_chg_coming_{int(60*periods_hrs)}_mins'].apply(
            lambda x: 1 if np.abs(x) > edge else 0)
    return df


raw_past_data_df = generate_dataset()
past_data_df = add_past_cl_pct_chg(raw_past_data_df)
past_data_df = add_target(past_data_df)

# past_data_df.head(20).to_csv('past_data_df.csv')
past_data_df.to_csv('dataset.csv')


# 最大幅と最小幅
# for dist in range(1, 11):
#     print(
#         f'{int(60*periods_hrs*dist)} mins:',
#         np.amin(past_data_df[
#             f'close_price_pct_chg_{int(60*periods_hrs*dist)}_mins']),
#         np.max(past_data_df[
#             f'close_price_pct_chg_{int(60*periods_hrs*dist)}_mins'])
#     )

# edgeが出現する頻度
# edge = 0.02
# for dist in range(1, 11):
#     print(
#         f'{int(60*periods_hrs*dist)} mins:',
#         (np.abs(past_data_df[
#             f'close_price_pct_chg_{int(60*periods_hrs*dist)}_mins'
#         ]) > edge).sum() / len(past_data_df[
#             f'close_price_pct_chg_{int(60*periods_hrs*dist)}_mins'
#         ])
#     )
