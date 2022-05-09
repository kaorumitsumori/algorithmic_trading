import os
import pandas as pd
from config import cnf_dict


def get_ABSOLUTE_PATH():
    DIR_ROOT_PATH = os.getcwd()
    DOLLAR_COST_AVG_DIR = os.path.join(DIR_ROOT_PATH, 'dollar_cost_avg')
    CHECKPOINT_PATH = os.path.join(DOLLAR_COST_AVG_DIR, 'checkpoint')
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    BACKUP_PATH = os.path.join(CHECKPOINT_PATH, 'backup')
    os.makedirs(BACKUP_PATH, exist_ok=True)
    return CHECKPOINT_PATH, BACKUP_PATH


def make_hist_csv():
    '''
    積み立て履歴を保存するcsvを作成する
    '''
    if not os.path.exists(os.path.join(CHECKPOINT_PATH, cnf_dict['fname'])):
        accumulation_hist_df = pd.DataFrame(columns=cnf_dict['cols_lst'])
        accumulation_hist_df.to_csv(
            os.path.join(CHECKPOINT_PATH, cnf_dict['fname']),
            index=False,
            encoding='cp932'
        )
    else:
        pass


CHECKPOINT_PATH, BACKUP_PATH = get_ABSOLUTE_PATH()
make_hist_csv()