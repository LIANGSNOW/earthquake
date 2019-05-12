import lightgbm as lgb
from sklearn.model_selection import KFold,train_test_split


def train_model(X,y,X_test,n_fold,params=None, model_type='lgb',
                plot_feature_importance='shap', topn=10, **paths):

    folds = KFold(n_splits=n_fold,shuffle=True,random_state=10)



    for fold_n,(train_index,valid_index) in enumerate(folds):

        X_trian,X_valid =  X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

        X_main_train, X_early_train, y_main_train, y_early_train = \
            train_test_split(X_trian, y_train, test_size=0.1, random_state=42)

        if model_type=='lgb':
            lgb_train = lgb.Dataset(X_main_train,y_main_train)
            lgb_valid = lgb.Dataset(X_early_train,y_early_train)
            gbm = lgb.train(params,lgb_train,30000,valid_sets=[lgb_train, lgb_valid],
                            early_stopping_rounds=200, verbose_eval=200)
            y_pred_valid = gbm.predict(X_valid)
            y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)
