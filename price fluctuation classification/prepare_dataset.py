import pandas as pd
import ccxt



def get_ohlcv_histry():
    exchange = ccxt.binance()
    symbol = 'BTC/USDT'
    timeframe = '1m'
    print("\n" + exchange.name + ' ' + symbol + ' ' + timeframe + ' chart')

    # get a list of ohlcv candles
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe)

    # each ohlcv candle is a list of [timestamp, open, high, low, close, volume]
    df_ohlcv = pd.DataFrame(
        ohlcv, columns=[
            'timestamp','op','hi','lo','cl','vl']).set_index('timestamp')


    index = 4  # use close price from each ohlcv candle
    # get the ohlCv (closing price, index == 4)
    series = [x[index] for x in ohlcv]

    height = 15
    length = 80
    # # print the chart
    # print("\n" + asciichart.plot(series[-length:], {'height': height}))  # print the chart

    last = ohlcv[len(ohlcv) - 1][index]  # last closing price

    # print last closing price
    print("\n" + exchange.name + " ₿ = $" + str(last) + "\n")
    return df_ohlcv


def main():
    # ftx = ccxt.ftx()
    # fetcher = FtxFetcher(ccxt_client=ftx)
    # df = fetcher.fetch_ohlcv(market='BTC-PERP', interval_sec=5*60)
    # df = df.dropna()
    # df = df.reset_index()
    # df = df[ # テスト期間を残せるように少し前で設定
    #     df['timestamp'] < pd.to_datetime('2021-01-01 00:00:00Z')]

    ohlcv = get_ohlcv_histry()
    ohlcv.dropna(inplace=True)
    # df.to_pickle('df_ohlcv.pkl')
    # df = df[
    #     df['timestamp'] < pd.to_datetime('2021-01-01 00:00:00')]  # テスト期間を残せるように少し前で設定
    ohlcv.to_pickle('dataset.pkl')


if __name__ == '__main__':
    main()
