import os
import numpy as np
import pandas as pd
import ccxt
import time
from datetime import datetime
from config import cnf_dict
from utils import CHECKPOINT_PATH, BACKUP_PATH


def activate_exchange():
    '''
    取引所を指定し、APIを利用可能にする
    '''
    ex = eval('ccxt.'+cnf_dict['ex_name']+'()')
    ex.apiKey = cnf_dict['apikey']
    ex.secret = cnf_dict['secret']
    return ex


def check_balance(ex):
    '''
    BTCとJPYの利用可能量を取得する
    '''
    try:
        balance = ex.fetch_balance()
    except Exception as e:
        print(datetime.now(), e)
    # else:
    available_btc = float(balance['BTC']['total'])
    available_jpy = float(balance['JPY']['total'])
    print('available_btc: ', available_btc)
    print('available_jpy: ', available_jpy)
    return available_btc, available_jpy


def calc_order_amt(ex):
    '''
    BTCの最低ask価格を取得し、注文量を計算する
    規定の最低注文量に満たない場合、最低注文量を採用する
    '''
    # BTC価格を調べる。tickerよりorder_bookが無難
    order_book = ex.fetch_order_book(cnf_dict['pair'], 5)
    print('min_ask_btc: ', order_book['asks'][0][0])
    daily_amt_btc = cnf_dict['daily_amt_jpy'] / order_book['asks'][0][0]
    print('daily_amt_btc: ', daily_amt_btc)
    order_amt_btc = max(
        (daily_amt_btc * cnf_dict['lmt_dict']['dgt'] // 1) / cnf_dict[
            'lmt_dict']['dgt'],
        cnf_dict['lmt_dict']['min']
    )
    print('order_amt_btc: ', order_amt_btc)
    return order_book, order_amt_btc


def bid_mkt_order(ex, order_amt_btc):
    '''
    BTCを成行でbidする
    '''
    time.sleep(cnf_dict['diff_sec'] * np.random.rand())
    bid_mkt_order = ex.create_market_buy_order(cnf_dict['pair'], order_amt_btc)
    print('bid_mkt_order: ', bid_mkt_order)


def monitor_order(ex, available_btc, available_jpy):
    '''
    注文が約定したかを毎秒チェックし、約定したらBTCとJPYの約定量を取得する
    '''
    while True:
        time.sleep(cnf_dict['waiting_sec'])
        latest_available_btc = float(ex.fetch_balance()['BTC']['total'])
        latest_available_jpy = float(ex.fetch_balance()['JPY']['total'])
        if latest_available_btc > available_btc and \
                latest_available_jpy < available_jpy:
            break
    print('latest_available_btc: ', latest_available_btc)
    print('latest_available_jpy: ', latest_available_jpy)
    agree_amt_jpy = available_jpy - latest_available_jpy
    agree_amt_btc = latest_available_btc - available_btc
    return agree_amt_jpy, agree_amt_btc


def get_past_data():
    '''
    前回までの積み立て額を取得する
    '''
    accumulation_hist_df = pd.read_csv(
        os.path.join(CHECKPOINT_PATH, cnf_dict['fname']),
        index_col=cnf_dict['index_col'],
        encoding='cp932'
    )
    if accumulation_hist_df.empty:
        accumulated_btc, invested_jpy = 0, 0
    else:
        accumulated_btc = accumulation_hist_df.iat[
            -1, cnf_dict['cols_lst'].index('accumulated_btc')-1]
        invested_jpy = accumulation_hist_df.iat[
            -1, cnf_dict['cols_lst'].index('invested_jpy')-1]
    return accumulation_hist_df, accumulated_btc, invested_jpy


def record(order_book, agree_amt_jpy, agree_amt_btc):
    '''
    前回までの積み立て額を取得し、今回の積み立て額を加算して追記する
    '''
    accumulation_hist_df, accumulated_btc, invested_jpy = get_past_data()
    accumulation_hist_df.loc[datetime.now().strftime('%Y-%m-%d-%H-%M')] = [
        float(order_book['bids'][0][0]),
        float(agree_amt_btc),
        float(agree_amt_jpy),
        float(accumulated_btc + agree_amt_btc),
        float(invested_jpy + agree_amt_jpy),
        float(order_book['bids'][0][0] * (accumulated_btc + agree_amt_btc)),
        float((order_book['bids'][0][0] * (
            accumulated_btc + agree_amt_btc)) - (
                invested_jpy + agree_amt_jpy)),
    ]
    accumulation_hist_df.to_csv(
        os.path.join(CHECKPOINT_PATH, cnf_dict['fname']), encoding='cp932')
    accumulation_hist_df.to_csv(
        os.path.join(
            BACKUP_PATH, datetime.now().strftime('%Y-%m-%d-%H-%M')+'.csv'),
        encoding='cp932'
    )


def main():
    print(datetime.now(), '-'*30)
    ex = activate_exchange()
    available_btc, available_jpy = check_balance(ex)
    order_book, order_amt_btc = calc_order_amt(ex)
    try:
        bid_mkt_order(ex, order_amt_btc)
    except Exception as e:
        print(datetime.now(), e)
    else:
        agree_amt_jpy, agree_amt_btc = monitor_order(
            ex, available_btc, available_jpy)
        record(order_book, agree_amt_jpy, agree_amt_btc)
    print(datetime.now(), '-'*30)


if __name__ == '__main__':
    main()