from datetime import datetime
import pandera as pa
import numpy as np




CNF_DICT = {
    'master': {
        'ex':  'bitflyer',
        'pair':  'btcjpy',
        'periods_hrs':  1 / 4,
        # 'periods':  int(3600 * periods_hrs),
        'after':  1514764800,
        'raw_cols':  [
            'CloseTime',
            'OpenPrice',
            'HighPrice',
            'LowPrice',
            'ClosePrice',
            'Volume',
            'QuoteVolume',
        ],
        'alpha':  0.0095,
        'MODEL_TYPES_LIST': [
            'young',
            'old',
        ],
        'wala_split_point': 15,
        'limited_term': datetime(2016, 12, 31),
        'limited_amt': 5e8,
        'CROSS_VALIDATION_TYPE': 'time_series',
    }
}


cnf_name = 'master'
cnf_dict = CNF_DICT[cnf_name]

# URL = f'https://api.cryptowat.ch/markets/{cnf_dict['ex']}/{cnf_dict['pair']}/ohlc?periods={cnf_dict['periods']}&after={cnf_dict['after']}'
# cnf_dict['URL'] = URL