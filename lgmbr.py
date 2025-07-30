import pandas as pd
import numpy as np
import lightgbm as lgbm
from sklearn.metrics import ndcg_score,make_scorer
from scipy.stats import uniform, randint
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb

df = pd.read_csv('D:/ds/pets/mslr/data/Fold1/mslrTrain.csv')

ytrain = df['target']
group_sizes = df.groupby('qid').size().tolist()

X = df.drop(columns=['target','qid'])


lgbmranker = lgbm.LGBMRanker(num_leaves=50, n_estimators=300,
                             max_depth=8, learning_rate=0.05)

xgbranker = xgb.XGBRanker(num_leaves=50, n_estimators=300,
                             max_depth=8, learning_rate=0.05)




lgbmranker.fit(X,ytrain,group=group_sizes)
xgbranker.fit(X,ytrain,group=group_sizes)

df_test = pd.read_csv('D:/ds/pets/mslr/data/Fold1/mslrTest.csv')

ytest = df_test['target']
group_sizes_test = df_test.groupby('qid').size().tolist()
Xtest = df_test.drop(columns=['target','qid'])

y_pred = lgbmranker.predict(Xtest)

y_predxgb = xgbranker.predict(Xtest)


df_testxgb = df_test.copy()
df_test['pred'] = y_pred
df_testxgb['pred'] = y_predxgb

ranked = df_test.sort_values(['qid','pred'], ascending = [True,False])
print(ranked.head())


def ndcg_scorer(df_test):
    valid_qids = df_test['qid'].value_counts()
    valid_qids = valid_qids[valid_qids > 1].index

    ndcg_list = []

    for qid in valid_qids:
        group = df_test[df_test['qid'] == qid]
        
        y_true = [group['target'].values.tolist()]  # 2D
        y_pred = [group['pred'].values.tolist()]    # 2D

        if len(y_true[0]) < 2:
            continue  # на всякий случай

        ndcg = ndcg_score(y_true, y_pred, k=5)
        ndcg_list.append(ndcg)
    print(f"np.mean {np.mean(group)}")

    print(f"Mean NDCG@5: {np.mean(ndcg_list):.4f}")
    
ndcg_scorer(df_test)
ndcg_scorer(df_testxgb)
print(f"disbalance {df_test['target'].value_counts(normalize=True)}")

"""
ndcg = ndcg_score(ranked['target'].values.tolist(),
                  ranked['pred'].values.tolist(),k=5)
"""

"""
grouped = ranked.groupby('qid')

counts = ranked['qid'].value_counts()
valid_qids = counts[counts>1].index
grouped = grouped[grouped['qid'].isin(valid_qids)]

scores = []

for qid, group in grouped:
    y_true = np.array([group['target'].values])
    y_pred = np.array([group['pred'].values])
    score = ndcg_score(y_true, y_pred, k=5)
    scores.append(score)


print(f"{np.mean(scores)}")
"""
