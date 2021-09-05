import requests
import json
from datetime import datetime
import time
import pandas as pd
import os
import pybitflyer

api = pybitflyer.API()
base_url = "https://api.bitflyer.jp"
endpoint = "/v1/board?product_code="
pair = "BTC_JPY"

data = requests.get(base_url + endpoint + pair, timeout=5)


# スリープ秒数
SLEEP_T = 1

# データ保存の時間[s]
DATA_T = 1800

# boardデータ取得関数

base_url = "https://api.bitflyer.jp"
endpoint = "/v1/board?product_code="

# 対象
pair = "BTC_JPY"


def get_board():
    # board = api.board(product_code=source)
    data = requests.get(base_url + endpoint + pair, timeout=5)
    board = json.loads(data.text)
    dict_data = {}
    dict_data['time'] = datetime.now()
    mid_price = {'mid_price': board['mid_price']}
    dict_data.update(mid_price)
    asks_price = {'asks_price_{}'.format(
        i): board["asks"][i]["price"] for i in range(10)}
    dict_data.update(asks_price)
    asks_size = {'asks_size_{}'.format(
        i): board["asks"][i]["size"] for i in range(10)}
    dict_data.update(asks_size)
    bids_price = {'bids_price_{}'.format(
        i): board["bids"][i]["price"] for i in range(10)}
    dict_data.update(bids_price)
    bids_size = {'bids_size_{}'.format(
        i): board["bids"][i]["size"] for i in range(10)}
    dict_data.update(bids_size)

    return pd.Series(dict_data)


def get_btc_board():
    # １秒待ってのループ処理
    print("start")
    init_time = datetime.now()
    end_time = datetime.now()
    main_list = []
    # 1hで1つのdfを作成
    while (end_time - init_time).seconds < DATA_T:
        try:
            dict_data = get_board()
            main_list.append(dict_data)
        except Exception as e:
            print("exception: ", e.args)
        # sleep
        time.sleep(SLEEP_T)
        end_time = datetime.now()

    df_data = pd.concat(main_list, axis=1).T

    # 何時から何時までの板情報かを記載した
    df_data.to_csv(
        './data/hour_bitcoin_day_{}_init_{}_{}_end_{}_{}.csv'.format(
            init_time.day,
            init_time.hour,
            init_time.minute,
            end_time.hour,
            init_time.minute
        )
    )
    print("end")


# 指値買い注文
def buy_btc_lmt(amt, prc):
    amt = int(amt*100000000)/100000000
    buy = api.sendchildorder(
        product_code="BTC_JPY",
        child_order_type="LIMIT",
        price=prc, side="BUY",
        size=amt,
        minute_to_expire=10,
        time_in_force="GTC"
    )
    print("BUY ", amt, "BTC")
    print(buy)


# 指値売り注文
def sell_btc_lmt(amt, prc):
    amt = int(amt*100000000)/100000000
    sell = api.sendchildorder(
        product_code="BTC_JPY",
        child_order_type="LIMIT",
        price=prc,
        side="SELL",
        size=amt,
        minute_to_expire=10,
        time_in_force="GTC"
    )
    print("SELL ", amt, "BTC")
    print(sell)


# の時はロング
def long():
    if r > 0:
        # JPY資産のうちどれだけBTCに変えるかを計算
        amt_jpy = compute(r, th1, th2)*jpy
        amt_btc = amt_jpy/ltps[itr-1]
        # 購入量が最小取引額を超えていれば指値買い
        if amt_btc > min_btc:
            buy_btc_lmt(amt_btc, ltps[itr-1])


if __name__ == '__main__':
    # カレントディレクトリー以下にdataを作成
    if not os.path.exists('./data'):
        os.mkdir('./data')
    # ループ処理
    while True:
        try:
            get_btc_board()
        except Exception as e:
            print("exception: ", e.args)