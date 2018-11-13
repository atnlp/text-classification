# encoding: utf-8
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from xgboost.sklearn import XGBClassifier
from lightgbm.sklearn import LGBMClassifier


def clf_lg():
    lg = LogisticRegression(C=10, dual=True)
    return lg


def clf_svm():
    lin_clf = svm.LinearSVC()
    return lin_clf


def clf_xgb():
    '''
    xgb = XGBClassifier(learning_rate=0.01,
                        n_estimators=3,
                        max_depth=19,
                        min_child_weight=2,
                        gamma=0.2,
                        subsample=0.9,
                        colsample_bytree=0.6,
                        reg_alpha=0.001,
                        objective='binary:logistic',
                        #  scale_pos_weight=2,
                        seed=25)
    '''
    xgb = XGBClassifier()
    return xgb


def clf_lgb():
    '''
     lgb = LGBMClassifier(boosting_type='gbdt',
                         objective='binary',
                         learning_rate=0.01,
                         num_iterations=350,
                         min_child_weight=2,
                         bagging_fraction=0.6,
                         feature_fraction=0.6,
                         # scale_pos_weight=3,
                         num_leaves=400,
                         max_depth=19,
                         seed=27,
                         reg_alpha=0.001)

    '''
    lgb = LGBMClassifier()
    return lgb



