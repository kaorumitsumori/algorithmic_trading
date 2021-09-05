ex = 'bitflyer'
pair = 'btcjpy'
periods_hrs = 1 / 4
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


edge = 0.0095
