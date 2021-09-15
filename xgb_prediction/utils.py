import requests
import json
from datetime import datetime
import time
import pandas as pd
import os
import pybitflyer


import os
import re
import pathlib
import logging
import numpy as np
import pandas as pd
from datetime import datetime
import pandera as pa
from config import cnf_dict


def get_ABSOLUTE_PATH():
    DIR_ROOT_PATH = os.getcwd()
    STRATEGY_PATH = os.path.join(DIR_ROOT_PATH, 'xgb_prediction')
    RAW_DATA_PATH = os.path.join(STRATEGY_PATH, 'Data', 'raw')
    os.makedirs(RAW_DATA_PATH, exist_ok=True)
    PREPROCESS_DATA_PATH = os.path.join(STRATEGY_PATH, 'Data', 'preprocess')
    os.makedirs(PREPROCESS_DATA_PATH, exist_ok=True)
    CHECKPOINT_PATH = os.path.join(STRATEGY_PATH, 'checkpoint')
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    return RAW_DATA_PATH, PREPROCESS_DATA_PATH, CHECKPOINT_PATH


def make_CURRENT_TIME_PATH():
    current_time = datetime.now().strftime('%Y-%m-%d-%H-%M')
    CURRENT_TIME_PATH = pathlib.Path(CHECKPOINT_PATH, current_time)
    os.makedirs(CURRENT_TIME_PATH, exist_ok=True)
    return CURRENT_TIME_PATH


def get_CURRENT_TIME_PATH():
    CHECKPOINT_LIST = []
    for checkpoint in os.listdir(CHECKPOINT_PATH):
        if not re.fullmatch(r'[0-9]+-[0-9]+-[0-9]+-[0-9]+-[0-9]+', checkpoint):
            continue
        CHECKPOINT_LIST.append(datetime.strptime(checkpoint, '%Y-%m-%d-%H-%M'))
    CURRENT_CHECKPOINT_PATH = os.path.join(
        CHECKPOINT_PATH, max(CHECKPOINT_LIST).strftime('%Y-%m-%d-%H-%M'))
    return CURRENT_CHECKPOINT_PATH


def read_dataset(file_name):
    df = pd.read_csv(
        make_file_path(PREPROCESS_DATA_PATH, file_name),
        encoding='cp932',
    )
    return df


def read_train_data(file_name):
    df = pd.read_csv(
        make_file_path(PREPROCESS_DATA_PATH, file_name),
        encoding='cp932',
    )
    df['CloseTime'] = pd.to_datetime(df['CloseTime'])
    df.set_index('CloseTime', inplace=True)
    return df


def make_file_path(DIR_PATH, file_name):
    return os.path.join(DIR_PATH, file_name)


def export_dataframe(df, DIR_PATH, file_name):
    file_path = make_file_path(DIR_PATH, file_name)
    df.to_csv(file_path, encoding='cp932', index=False)


def export_dataframe_with_index(df, DIR_PATH, file_name):
    file_path = make_file_path(DIR_PATH, file_name)
    df.to_csv(file_path, encoding='cp932')


def set_X_y(train, test):
    tr_X = train.drop(['alpha_occurrence', 'cl_pct_chg_coming_60_mins'], axis=1)
    ts_X = test.drop(['alpha_occurrence', 'cl_pct_chg_coming_60_mins'], axis=1)
    tr_y = train['cl_pct_chg_coming_60_mins']
    ts_y = test['cl_pct_chg_coming_60_mins']
    return tr_X, ts_X, tr_y, ts_y


def get_diff_target(df, mth):
    df_target = pd.DataFrame()
    cols = [col for col in df.columns if col.endswith('_')]
    df_target = df[cols].apply(lambda x: x.diff(-1*(mth)), axis=0)
    return df_target


def get_log_target(df, mth):
    df_target = pd.DataFrame()
    cols = [col for col in df.columns if col.endswith('_')]
    df_target = df[cols].apply(lambda x: np.log1p(x).diff(-1*(mth)), axis=0)
    return df_target


def check_df():
    columns_dict = cnf_dict['columns_dict']
    tr_checker = pa.DataFrameSchema(columns=columns_dict)
    _ = columns_dict.pop('alpha_occurrence')
    prd_checker = pa.DataFrameSchema(columns=columns_dict)
    return tr_checker, prd_checker


def setup_logger(file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    logger = logging.getLogger(__name__)
    log_format = logging.Formatter('%(asctime)s : %(message)s')
    handler = logging.FileHandler(file_path)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(log_format)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    return logger


RAW_DATA_PATH, PREPROCESS_DATA_PATH, CHECKPOINT_PATH = get_ABSOLUTE_PATH()
CURRENT_TIME_PATH = make_CURRENT_TIME_PATH()