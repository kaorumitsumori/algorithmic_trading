import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


from config import cnf_dict


from sklearn.model_selection import cross_val_score, KFold




def check_importance():
    print('featuresは使う特徴量カラム名配列')
    print('重要度表示。重要度が高いものは汎化性能に悪影響を与える可能性がある')
    df = pd.read_pickle('ml_tutorial/df_features.pkl')

    # 設定など。重要ではない
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    pd.options.mode.chained_assignment = None

    model = lgb.LGBMRegressor(n_jobs=-1, random_state=1)
    model.fit(df[cnf_dict['features']], np.arange(df.shape[0]))

    feature_imp = pd.DataFrame(
        sorted(zip(
            model.feature_importances_,
            cnf_dict['features']
        )),
        columns=['Value', 'Feature']
    )
    plt.figure(figsize=(20, 40))
    sns.barplot(
        x="Value",
        y="Feature",
        data=feature_imp.sort_values(by="Value", ascending=False)
    )
    plt.title('LightGBM Features adv val (avg over folds)')
    plt.tight_layout()
    plt.show()

    print('スコア計算。スコアが高いと汎化性能が悪い可能性ある (目安は0.3以下)')
    cv = KFold(n_splits=2, shuffle=True, random_state=0)
    scores = cross_val_score(
        model,
        df[cnf_dict['features']],
        np.arange(df.shape[0]), scoring='r2', cv=cv
    )
    print('score mean, std', np.mean(scores), np.std(scores))


def main():
    check_importance()


if __name__ == '__main__':
    main()