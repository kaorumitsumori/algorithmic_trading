import requests
from datetime import datetime


# CryptowatchのAPIを使用する関数
def get_price(min, before=0, after=0):
    price = []
    params = {'periods': min}
    if before != 0:
        params['before'] = before
    if after != 0:
        params['after'] = after

    response = requests.get(
        'https://api.cryptowat.ch/markets/bitflyer/btcfxjpy/ohlc',
        params
    )
    data = response.json()

    if data['result'][str(min)] is not None:
        for i in data['result'][str(min)]:
            price.append({'close_time': i[0],
                          'close_time_dt': datetime.fromtimestamp(
                              i[0]).strftime('%Y/%m/%d %H:%M'),
                          'open_price': i[1],
                          'high_price': i[2],
                          'low_price': i[3],
                          'close_price': i[4]})
        return price
    else:
        print('No data exist')
        return None


# 時間と始値・終値を表示する関数
def print_price( data ):
	print( "時間： " + datetime.fromtimestamp(data["close_time"]).strftime('%Y/%m/%d %H:%M') + " 高値： " + str(data["high_price"]) + " 安値： " + str(data["low_price"]) )


# ここからメイン
hrs = 60 / 3600
mins = hrs * 3600
price = get_price(60)


if price is not None:
    print('--------------------------')
    print('先頭データ : ' + price[0]['close_time_dt'] +
          '  UNIX時間 : ' + str(price[0]['close_time']))
    print('末尾データ : ' + price[-1]['close_time_dt'] +
          '  UNIX時間 : ' + str(price[-1]['close_time']))
    print('合計 ： ' + str(len(price)) + '件のローソク足データを取得')
    print({price[i]['close_time_dt']: price[i]['close_price'] for i in range(50)})
    print('--------------------------')
