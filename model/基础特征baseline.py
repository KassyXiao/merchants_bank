# -*- coding: utf-8 -*-
"""
基本特征包括：
个人信息： 30列 不处理
日志信息： 1. 用户点击次数 1列
          2. 一个月中每个id在每一天的点击次数 31列
          3. 点击模块（如饭票-代金券-门店详情）前两列ont-hot
          4. 浏览类型 2列
          5. 每个用户每次点击间隔 最小值-最大值-均值-方差 4列
          6. 最后7天累计
@author: 肖鹏程
@QQ: 609659119
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# =============================================================================
# #加载基础特征数据
# =============================================================================
data = pd.read_csv('base_line_data.csv',sep='\t')
train = data[data['FLAG']!=-1]
test = data[data['FLAG']==-1]

train_userid = train.pop('USRID')
y = train.pop('FLAG')
col = train.columns
X = train[col].values

test_userid = test.pop('USRID')
test_y = test.pop('FLAG')
test = test[col].values

# =============================================================================
# 训练 
# =============================================================================
N = 5
skf = StratifiedKFold(n_splits=N,shuffle=False,random_state=42)

lgb_cv = []
lgb_pre = []
xgb_cv = []
xgb_pre = []
mean_cv = []
mean_pre = []
for train_in,test_in in skf.split(X,y):
    X_train,X_test,y_train,y_test = X[train_in],X[test_in],y[train_in],y[test_in]   
    ########################## xgboost ###############################
    xgboost_params = {'booster': 'gbtree',
                      'objective':'binary:logistic',
                      'eta': 0.01,
                      'max_depth': 5, 
                      'colsample_bytree': 0.7,
                      'subsample': 0.7,
                      'min_child_weight': 9, 
                      'silent':1,
                      'eval_metric':'auc',
                      'lambda' : 20,
                      }
    print('XGBoost Start training...')
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    watchlist = [(dtrain,'train'),(dtest,'val')] 
    xgb_model = xgb.train(xgboost_params, dtrain,4000,evals=watchlist,verbose_eval=500,early_stopping_rounds=300)
    print('XGBoost Start predicting...')
    dvali = xgb.DMatrix(X_test)
    xgb_val_pred = xgb_model.predict(dvali)
    xgb_cv.append(roc_auc_score(y_test,xgb_val_pred))
    dfinal = xgb.DMatrix(test)
    xgb_pred = xgb_model.predict(dfinal)
    xgb_pre.append(xgb_pred)

s_xgb = 0
for i in xgb_pre:
    s_xgb = s_xgb + i
s_xgb = s_xgb /N
res_xgb = pd.DataFrame()
res_xgb['USRID'] = list(test_userid.values)
res_xgb['RST'] = list(s_xgb)
print('xgboost_cv',np.mean(xgb_cv))

####保存
res_xgb.to_csv('base_line_feature.csv',index=False,sep='\t')










