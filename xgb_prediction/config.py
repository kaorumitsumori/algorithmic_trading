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
        'limited_term': datetime(2019, 12, 31),
        'test_size': 0.2,
        'valid_length': 3,
        'optuna_params': {
            'max_depth': list(range(3, 13)),
            'min_child_weight': list(range(1, 7)),
            'gamma': list(np.linspace(0, 1, 5)),
            'subsample': list(np.linspace(0.5, 1, 5)),
            'colsample_bytree': list(np.linspace(0.5, 1, 5)),
            'lambda': [i**2 / 100.0 for i in np.arange(0, 11)],
            'eta': 0.01,
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
        },
        'num_boost_round': 500,
        'early_stopping_rounds': 15,
        'n_trials': 20,#100
        'n_splits': 3,
        'CROSS_VALIDATION_TYPE': 'time_series',
    }
}


cnf_name = 'master'
cnf_dict = CNF_DICT[cnf_name]

# URL = f'https://api.cryptowat.ch/markets/{cnf_dict['ex']}/{cnf_dict['pair']}/ohlc?periods={cnf_dict['periods']}&after={cnf_dict['after']}'
# cnf_dict['URL'] = URL
