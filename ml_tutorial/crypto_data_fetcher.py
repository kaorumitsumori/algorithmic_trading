import time
import pandas as pd
from logging import getLogger


def smart_append(df, other):
    if other is None or other.shape[0] == 0:
        return df.copy()
    if df is None:
        df = other.copy()
    else:
        df = df.append(other)
    df.sort_index(inplace=True, kind='mergesort')
    # https://stackoverflow.com/questions/13035764/remove-rows-with-duplicate-indices-pandas-dataframe-and-timeseries
    return df[~df.index.duplicated(keep='last')]


def create_null_logger():
    return getLogger(__name__ + 'null_logger')


def normalize_to_unix(tm):
    if hasattr(tm, 'timestamp'):
        return tm.timestamp()
    else:
        return tm


class FtxFetcher:
    def __init__(self, logger=create_null_logger(), ccxt_client=None):
        self.logger = logger
        self.ccxt_client = ccxt_client

    def fetch_ohlcv(
            self, df=None, start_time=None, interval_sec=None,
            market=None, price_type=None):
        limit = 5000

        if start_time:
            from_time = normalize_to_unix(start_time)
        else:
            from_time = self._find_start_time(market=market)

        if df is not None and df.shape[0]:
            from_time = int(df.index.max().timestamp()) + interval_sec

        # ftxは最近のデータが返ってくるから、最初から取得するには、end_timeをstart_time + interval * limitに設定する必要がある
        # そうすると、データが足りないことを終了判定に使えないから、データが足りないときはfrom_timeを進める
        # from_timeが現在を超えたら終了
        # end_timeが現在時刻を超えたら何も返らないので注意 (expired futureは現在時刻の代わりにfuture期限)

        dfs = []

        total_end_time = self._find_total_end_time(market=market)

        while from_time < total_end_time:
            end_time = from_time + interval_sec * limit
            end_time = min([end_time, total_end_time])  # 未来時刻だと何も返らないので
            self.logger.debug('{} {} {}'.format(market, from_time, end_time))

            data = self.ccxt_client.publicGetMarketsMarketNameCandles({
                'market_name': market,
                'start_time': from_time,
                'end_time': end_time - 1,  # キャッシュを無効にするために必要。境界値を含む仕様っぽいので含まないように調整
                'resolution': interval_sec,
                'limit': limit
            })['result']

            from_time = end_time

            if len(data) <= 0:
                self.logger.debug('len(data) <= 1')
                continue

            # self.logger.debug(data)

            df2 = pd.DataFrame(data)
            columns = ['timestamp', 'op', 'hi', 'lo', 'cl']
            if price_type is None:
                columns.append('volume')
            df2 = df2.rename(columns={
                'open': 'op',
                'high': 'hi',
                'low': 'lo',
                'close': 'cl',
                'startTime': 'timestamp',
            })[columns]
            df2['timestamp'] = pd.to_datetime(df2['timestamp'], utc=True)

            for col in ['op', 'hi', 'lo', 'cl', 'volume']:
                if col in columns:
                    df2[col] = df2[col].astype('float64')

            dfs.append(df2)

        if len(dfs) == 0:
            return None if df is None else df.copy()
        else:
            df = smart_append(df, pd.concat(dfs).set_index('timestamp'))
            # 最後は未確定足なので削除
            df = df[df.index != df.index.max()]
            return df

    def fetch_fr(self, df=None, start_time=None, market=None):
        limit = 1000  # undocumented
        interval_sec = 60 * 60

        if start_time:
            from_time = normalize_to_unix(start_time)
        else:
            from_time = self._find_start_time(market=market)

        if df is not None and df.shape[0]:
            from_time = int(df.index.max().timestamp()) + interval_sec

        # ftxは最近のデータが返ってくるから、最初から取得するには、end_timeをstart_time + interval * limitに設定する必要がある
        # そうすると、データが足りないことを終了判定に使えないから、データが足りないときはfrom_timeを進める
        # from_timeが現在を超えたら終了

        dfs = []

        total_end_time = self._find_total_end_time(market=market)

        while from_time < total_end_time:
            end_time = from_time + interval_sec * limit
            end_time = min([end_time, total_end_time])  # 未来時刻だと何も返らないので
            self.logger.debug('{} {} {}'.format(market, from_time, end_time))

            data = self.ccxt_client.publicGetFundingRates({
                'future': market,
                'start_time': from_time,
                'end_time': end_time - 1,  # キャッシュを無効にするために必要。境界値を含む仕様っぽいので含まないように調整
                'limit': limit
            })['result']

            from_time = end_time

            if len(data) <= 0:
                self.logger.debug('len(data) <= 1')
                continue

            # self.logger.debug(data)

            df2 = pd.DataFrame(data)
            df2 = df2.rename(columns={
                'rate': 'fr',
                'time': 'timestamp',
            })[['fr', 'timestamp']]
            df2['timestamp'] = pd.to_datetime(df2['timestamp'], utc=True)

            for col in ['fr']:
                df2[col] = df2[col].astype('float64')

            dfs.append(df2)

        if len(dfs) == 0:
            return None if df is None else df.copy()
        else:
            return smart_append(df, pd.concat(dfs).set_index('timestamp'))

    def _find_start_time(self, market=None):
        limit = 5000

        data = self.ccxt_client.publicGetMarketsMarketNameCandles({
            'market_name': market,
            'resolution': 24 * 60 * 60,
            'limit': limit
        })['result']

        if len(data) == 0:
            return int(time.time())

        df2 = pd.DataFrame(data)
        columns = ['timestamp']
        df2 = df2.rename(columns={
            'startTime': 'timestamp',
        })[columns]
        df2['timestamp'] = pd.to_datetime(df2['timestamp'], utc=True)

        return df2['timestamp'].min().timestamp()

    def _find_total_end_time(self, market=None):
        try:
            future = self.ccxt_client.publicGetFuturesFutureName({
                'future_name': market,
            })['result']
        except Exception as e:
            return time.time() - 1
        if future is not None and future['expiry'] is not None:
            return min([pd.to_datetime(future['expiry'], utc=True).timestamp(), time.time() - 1])
        else:
            return time.time() - 1
