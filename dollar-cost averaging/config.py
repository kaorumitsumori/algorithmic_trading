exchange = ''  # 利用する取引所。liquid, coincheck, bitflyer, bitbank
apikey = ''  # API Key
secret = ''  # API Secret
daily_amt_jpy = 3288  # 日時でBTCを購入する金額（円）


lmts = {
    'liquid': {'min': 0.001, 'dgt': 10000000},
    'coincheck': {'min': 0.001, 'dgt': 10000},
    'bitflyer': {'min': 0.001, 'dgt': 10000},
    'bitbank': {'min': 0.0001, 'dgt': 10000},
}


cols_lst = [
    'datetime',
    'max_bid_btc',
    'agree_amt_btc',
    'agree_amt_jpy',
    'accumulated_btc',
    'invested_jpy',
    'total_cap',
    'total_revenue',
]


CNF_DICT = {
    'master': {
        'ex_name': exchange,
        'lmt_dict': lmts[exchange],
        'apikey': apikey,
        'secret': secret,
        'daily_amt_jpy': daily_amt_jpy,
        'diff_sec': 3600*8,  # 購入時刻をランダムでずらす範囲
        'fname': 'accumulation_hist.csv',
        'pair': 'BTC/JPY',
        'waiting_sec': 1,
        'cols_lst': cols_lst,
        'index_col': 'datetime',
    },
}


cnf_name = 'master'
cnf_dict = CNF_DICT[cnf_name]