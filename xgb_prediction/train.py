import os
import json
import pickle
import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from datetime import datetime
from dateutil.relativedelta import relativedelta
import optuna
import functools
from utils import get_CURRENT_TIME_PATH
from utils import read_train_data, set_X_y
from config import cnf_dict


def split_test_over_time(df):
    yrs_lst = list(
        range(cnf_dict['limited_term'].year, datetime.now().year + 1))
    mths_lst = list(range(1, 13))
    tr = pd.DataFrame()
    ts = pd.DataFrame()
    for yr in yrs_lst:
        for mth in mths_lst:
            start = datetime(yr, mth, 1)
            end = start + relativedelta(months=1)
            df_temp = df[(df['日付'] >= start) & (df['日付'] < end)]
            if len(df_temp) < 1 / cnf_dict['test_size']:
                continue
            tr_temp, ts_temp = train_test_split(
                df_temp, test_size=cnf_dict['test_size'], random_state=9407)
            tr = pd.concat([tr, tr_temp])
            ts = pd.concat([ts, ts_temp])
    return tr, ts


def split_time_series_data(tr_vd, tr_start, vd_start):
    tr_end = vd_start
    vd_start = tr_end
    vd_end = vd_start + relativedelta(months=cnf_dict['valid_length'])
    tr = tr_vd[(tr_vd['日付'] >= tr_start) & (tr_vd['日付'] < tr_end)]
    vd = tr_vd[(tr_vd['日付'] >= vd_start) & (tr_vd['日付'] < vd_end)]
    return tr, vd


def train_valid_xgb(tr_X, tr_y, vd_X, vd_y, params):
    dtrain = xgb.DMatrix(tr_X, label=tr_y)
    dtest = xgb.DMatrix(vd_X, label=vd_y)
    watchlist = [(dtrain, 'train'), (dtest, 'test')]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=cnf_dict['num_boost_round'],
        early_stopping_rounds=cnf_dict['early_stopping_rounds'],
        verbose_eval=True,
        evals=watchlist,
    )
    prd_y = model.predict(dtest, ntree_limit=model.best_ntree_limit)
    RMSE_loss = np.sqrt(((prd_y - np.array(vd_y)) ** 2).mean())
    return RMSE_loss


def optimize_hps(tr_X, vd_X, tr_y, vd_y, trial):
    params = {
        'max_depth': trial.suggest_categorical(
            'max_depth', cnf_dict['optuna_params']['max_depth']),
        'min_child_weight': trial.suggest_categorical(
            'min_child_weight', cnf_dict['optuna_params']['min_child_weight']),
        'gamma': trial.suggest_categorical(
            'gamma', cnf_dict['optuna_params']['gamma']),
        'subsample': trial.suggest_categorical(
            'subsample', cnf_dict['optuna_params']['subsample']),
        'colsample_bytree': trial.suggest_categorical(
            'colsample_bytree', cnf_dict['optuna_params']['colsample_bytree']),
        'lambda': trial.suggest_categorical(
            'lambda', cnf_dict['optuna_params']['lambda']),
        'eta': cnf_dict['optuna_params']['eta'],
        'objective': cnf_dict['optuna_params']['objective'],
        'random_state': 2396,  # any random number
        'eval_metric': cnf_dict['optuna_params']['eval_metric'],
    }
    RMSE_loss = train_valid_xgb(tr_X, tr_y, vd_X, vd_y, params)
    return RMSE_loss


def optimize_hps_on_KFold(tr_vd_X, tr_vd_y, trial):
    params = {
        'max_depth': trial.suggest_categorical(
            'max_depth', cnf_dict['optuna_params']['max_depth']),
        'min_child_weight': trial.suggest_categorical(
            'min_child_weight', cnf_dict['optuna_params']['min_child_weight']),
        'gamma': trial.suggest_categorical(
            'gamma', cnf_dict['optuna_params']['gamma']),
        'subsample': trial.suggest_categorical(
            'subsample', cnf_dict['optuna_params']['subsample']),
        'colsample_bytree': trial.suggest_categorical(
            'colsample_bytree', cnf_dict['optuna_params']['colsample_bytree']),
        'lambda': trial.suggest_categorical(
            'lambda', cnf_dict['optuna_params']['lambda']),
        'eta': cnf_dict['optuna_params']['eta'],
        'objective': cnf_dict['optuna_params']['objective'],
        'random_state': 9762,  # any random number
        'eval_metric': cnf_dict['optuna_params']['eval_metric'],
    }
    RMSE_lst = []
    kf = KFold(
        n_splits=cnf_dict['n_splits'],
        random_state=1928,  # any random number
        shuffle=True
    )
    for i, (tr_id, vd_id) in enumerate(kf.split(tr_vd_y)):
        tr_X, vd_X = tr_vd_X.iloc[tr_id], tr_vd_X.iloc[vd_id]
        tr_y, vd_y = tr_vd_y.iloc[tr_id], tr_vd_y.iloc[vd_id]
        RMSE_loss = train_valid_xgb(tr_X, tr_y, vd_X, vd_y, params)
        RMSE_lst.append(RMSE_loss)
    avg_RMSE = np.mean(RMSE_lst)
    return avg_RMSE


def calc_cpr_RMSE(prd_cpr, real_cpr):
    RMSE_loss = np.sqrt(((prd_cpr - real_cpr) ** 2).mean()[0])
    return RMSE_loss


def test_bst_hps(tr_X, ts_X, tr_y, ts_y, ts_X_copy, params):
    dtrain = xgb.DMatrix(tr_X, label=tr_y)
    dtest = xgb.DMatrix(ts_X, label=ts_y)
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=cnf_dict['num_boost_round'],
    )
    prd_y = model.predict(dtest, ntree_limit=model.best_ntree_limit)
    prd_cpr, real_cpr = revert_target_to_cpr(
        cnf_dict['TARGET_TRANSFORM_TYPE'], ts_X_copy, prd_y, ts_y)
    RMSE_loss = calc_cpr_RMSE(prd_cpr, real_cpr)
    return RMSE_loss


def execute_time_cross_valid(tr_vd, ts, mth, model_type):
    tr_start = cnf_dict['limited_term']
    latest_vd_start = datetime(
        datetime.now().year, datetime.now().month, 1
    ) - relativedelta(months=cnf_dict['valid_length'])
    reslt_dict = {}
    for fold in range(cnf_dict['n_splits'], 0, -1):
        vd_start = latest_vd_start - relativedelta(months=fold)
        print('Now validating at ', vd_start)
        tr, vd = split_time_series_data(tr_vd, tr_start, vd_start)
        if vd.empty:
            print('Pass, empty valid data.\n')
            continue
        tr_X, vd_X, tr_y, vd_y = set_X_y(tr, vd)
        tr_X, vd_X, _ = clean_data(tr_X, vd_X)
        study = optuna.create_study()
        study.optimize(functools.partial(
            optimize_hps, tr_X, vd_X, tr_y, vd_y
        ), n_trials=cnf_dict['n_trials'])
        best_hps_dict = study.best_params
        tr_X, ts_X, tr_y, ts_y = set_X_y(tr, ts)
        tr_X, ts_X, ts_X_copy = clean_data(tr_X, ts_X)
        RMSE_loss = test_bst_hps(
            tr_X, ts_X, tr_y, ts_y, ts_X_copy, best_hps_dict)
        reslt_dict[str(vd_start)] = {
            'Train term': str(tr_start) + ' <= t < ' + str(vd_start),
            'RMSE Loss': RMSE_loss,
            'Hyper parameters': best_hps_dict,
        }
    with open(os.path.join(
            CURRENT_CHECKPOINT_PATH, f'{mth}MF_train_info_{model_type}.json'
    ), 'w') as f:
        json.dump(reslt_dict, f, indent=4, ensure_ascii=False)
    bst_hps_dict = list(reslt_dict.items())[-1][1]['Hyper parameters']
    return bst_hps_dict


def execute_cross_valid(tr_vd, ts, mth, model_type):
    tr_vd_X, ts_X, tr_vd_y, ts_y = set_X_y(tr_vd, ts)
    tr_vd_X, ts_X, ts_X_copy = clean_data(tr_vd_X, ts_X)
    study = optuna.create_study()
    study.optimize(functools.partial(
        optimize_hps_on_KFold, tr_vd_X, tr_vd_y
    ), n_trials=cnf_dict['n_trials'])
    bst_hps_dict = study.best_params
    RMSE_loss = test_bst_hps(
        tr_vd_X, ts_X, tr_vd_y, ts_y, ts_X_copy, bst_hps_dict)
    reslt_dict = {
        'RMSE Loss': RMSE_loss,
        'Hyper parameters': bst_hps_dict,
    }
    with open(os.path.join(
            CURRENT_CHECKPOINT_PATH, f'{mth}MF_train_info_{model_type}.json'
    ), 'w') as f:
        json.dump(reslt_dict, f, indent=4, ensure_ascii=False)
    return bst_hps_dict


def plot_xgb_importance(xgb_model, mth, model_type):
    df_features = pd.DataFrame(
        xgb_model.get_fscore().items(), columns=['feature', 'importance']
    ).sort_values('importance', ascending=False)
    plt.figure(figsize=(18, 10))
    if len(df_features) < 15:
        sns.barplot(x="feature", y="importance", data=df_features)
    else:
        sns.barplot(x="feature", y="importance", data=df_features[:15])
    plt.xticks(fontsize=14, rotation=70)
    plt.tight_layout()
    plt.savefig(os.path.join(
        CURRENT_CHECKPOINT_PATH,
        f'{mth}MF_feature_importance_{model_type}.png'
    ))
    plt.clf()


def build_model(df, params, mth, model_type):
    X = df.drop(columns=cnf_dict['drop_lst'], axis=1)
    X = X.drop(columns=['Target'], axis=1)
    y = df['Target']
    dtrain = xgb.DMatrix(X, label=y)
    bst = xgb.train(params, dtrain)
    importance_dict = OrderedDict(sorted(
        OrderedDict(bst.get_fscore()).items(),
        key=lambda k_v: k_v[1],
        reverse=True
    ))
    model_info_dict = {
        'Hyper parameters': params,
        'Feature　importance': importance_dict,
    }
    with open(os.path.join(
            CURRENT_CHECKPOINT_PATH, f'{mth}MF_model_info_{model_type}.json'
    ), 'w') as f:
        json.dump(model_info_dict, f, indent=4, ensure_ascii=False)
    with open(os.path.join(
            CURRENT_CHECKPOINT_PATH, f'{mth}MF_xgb_model_{model_type}.pkl'
    ), mode='wb') as f:
        pickle.dump(bst, f)
    plot_xgb_importance(bst, mth, model_type)


def main():
    df = read_train_data(mth, model_type)
    tr_vd, ts = split_test_over_time(df)
    if cnf_dict['CROSS_VALIDATION_TYPE'] == 'time_series':
        bst_hps = execute_time_cross_valid(
            tr_vd, ts, mth, model_type)
    if cnf_dict['CROSS_VALIDATION_TYPE'] == 'kfold':
        bst_hps = execute_cross_valid(
            tr_vd, ts, mth, model_type)
    build_model(df, bst_hps, mth, model_type)


if __name__ == '__main__':
    CURRENT_CHECKPOINT_PATH, _ = get_CURRENT_TIME_PATH()
    main()