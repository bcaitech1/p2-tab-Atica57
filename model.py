# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import os, sys, gc, random
import datetime
import dateutil.relativedelta
import argparse

# Machine learning
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool

# Custom library
from utils_1 import *
from features_1 import *
from model import *

#Wandb
import wandb
import optuna

TOTAL_THRES = 300 # 구매액 임계값
SEED = 42 # 랜덤 시드
seed_everything(SEED) # 시드 고정


data_dir = '../input' # os.environ['SM_CHANNEL_TRAIN']
model_dir = '../model' # os.environ['SM_MODEL_DIR']
output_dir = '../output' # os.environ['SM_OUTPUT_DATA_DIR']


'''optuna objective'''

#LightBGM optuna objective
def lgb_objective(trial) :
    # 데이터 파일 읽기
    data = pd.read_csv(data_dir + '/train.csv', parse_dates=['order_date'])
    year_month_for_val = '2011-11'
    label = generate_label2(data, year_month_for_val)['label']

    model_params = {
        'objective': 'binary', # 이진 분류
        'boosting_type': 'gbdt',
        'metric': 'auc', # 평가 지표 설정
        'num_leaves': trial.suggest_int('num_leaves', 2, 256), # num_leaves 값을 2-256까지 정수값 중에 사용
        'max_bin': trial.suggest_int('max_bin', 128, 256), # max_bin 값을 128-256까지 정수값 중에 사용
        # min_data_in_leaf 값을 10-40까지 정수값 중에 사용
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 40),
        # 피처 샘플링 비율을 0.4-1.0까지 중에 uniform 분포로 사용
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.2, 1.0),
        # 데이터 샘플링 비율을 0.4-1.0까지 중에 uniform 분포로 사용
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.2, 1.0),
        # 데이터 샘플링 횟수를 1-7까지 정수값 중에 사용
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'n_estimators': 10000, # 트리 개수
        'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 10, 500),
        # L1 값을 1e-8-10.0까지 로그 uniform 분포로 사용
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        # L2 값을 1e-8-10.0까지 로그 uniform 분포로 사용
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        'seed': SEED,
        'verbose': -1,
        'n_jobs': -1,   
    }
    
    #feature engineering
    train, test, y, features = feature_engineering2(data, year_month_for_val)
    # oof prediction 함수 호출해서 out of fold validation 예측값을 얻어옴
    y_oof, test_preds, fi = make_lgb_oof_prediction(train, y, test, features, model_params=model_params, is_optuna=True)
    
    # Validation 스코어 계산
    val_auc = roc_auc_score(label, test_preds)
    
    return val_auc

#XGBoost optuna objective
def xgb_objective(trial) :
    # 데이터 파일 읽기
    data = pd.read_csv(data_dir + '/train.csv', parse_dates=['order_date'])
    year_month_for_val = '2011-11'
    label = generate_label2(data, year_month_for_val)['label']

    model_params = {
        'objective': 'binary:logistic', # 이진 분류
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.1), # 학습률
        'max_depth': trial.suggest_int('max_depth', 2, 20), # 트리 최고 깊이
        'colsample_bytree': trial.suggest_loguniform('colsample_bytree', 0.1, 1), # 피처 샘플링 비율
        'subsample': trial.suggest_loguniform('subsample', 0.1, 1), # 데이터 샘플링 비율
        'eval_metric': 'auc', # 평가 지표 설정
        'seed': SEED,
    }
    
    #feature engineering
    train, test, y, features = feature_engineering2(data, year_month_for_val)
    # oof prediction 함수 호출해서 out of fold validation 예측값을 얻어옴
    y_oof, test_preds, fi = make_xgb_oof_prediction(train, y, test, features, model_params=model_params, is_optuna=True)
    
    # Validation 스코어 계산
    val_auc = roc_auc_score(label, test_preds)
    
    return val_auc

#CatBoost optuna objective
def cat_objective(trial) :
    # 데이터 파일 읽기
    data = pd.read_csv(data_dir + '/train.csv', parse_dates=['order_date'])
    year_month_for_val = '2011-11'
    label = generate_label2(data, year_month_for_val)['label']

    model_params = {
        'n_estimators': 10000, # 트리 개수
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.1), #학습률
        'eval_metric': 'AUC', # 평가 지표 설정
        'loss_function': 'Logloss', # 손실 함수 설정
        'random_seed': SEED,
        'metric_period': trial.suggest_int('metric_period', 10, 50),
        'od_wait': trial.suggest_int('od_wait', 10, 200), # early stopping round
        'depth': trial.suggest_int('depth', 2, 8), # 트리 최고 깊이
        'rsm': trial.suggest_uniform('rsm', 0.2, 1.0), # 피처 샘플링 비율  
    }
    
    #feature engineering
    train, test, y, features = feature_engineering2(data, year_month_for_val)
    # oof prediction 함수 호출해서 out of fold validation 예측값을 얻어옴
    y_oof, test_preds, fi = make_cat_oof_prediction(train, y, test, features, model_params=model_params, is_optuna=True)
    
    # Validation 스코어 계산
    val_auc = roc_auc_score(label, test_preds)
    
    return val_auc



'''
    머신러닝 모델 없이 입력인자으로 받는 year_month의 이전 달 총 구매액을 구매 확률로 예측하는 베이스라인 모델
'''
def baseline_no_ml(df, year_month, total_thres=TOTAL_THRES):
    # year_month에 해당하는 label 데이터 생성
    month = generate_label(df, year_month)
    
    # year_month 이전 월 계산
    d = datetime.datetime.strptime(year_month, "%Y-%m")
    prev_d = d - dateutil.relativedelta.relativedelta(months=1)
    prev_d = prev_d.strftime('%Y-%m')
    
    # 이전 월에 해당하는 label 데이터 생성
    previous_month = generate_label(df, prev_d)
    
    # merge하기 위해 컬럼명 변경
    previous_month = previous_month.rename(columns = {'total': 'previous_total'})

    month = month.merge(previous_month[['customer_id', 'previous_total']], on = 'customer_id', how = 'left')
    
    # 거래내역이 없는 고객의 구매액을 0으로 채움
    month['previous_total'] = month['previous_total'].fillna(0)
    # 이전 월의 총 구매액을 구매액 임계값으로 나눠서 예측 확률로 계산
    month['probability'] = month['previous_total'] / total_thres
    
    # 이전 월 총 구매액이 구매액 임계값을 넘어서 1보다 클 경우 예측 확률을 1로 변경
    month.loc[month['probability'] > 1, 'probability'] = 1
    
    # 이전 월 총 구매액이 마이너스(주문 환불)일 경우 예측 확률을 0으로 변경
    month.loc[month['probability'] < 0, 'probability'] = 0
    
    return month['probability']


def make_lgb_prediction(train, y, test, features, categorical_features='auto', model_params=None):
    x_train = train[features]
    x_test = test[features]
    
    print(x_train.shape, x_test.shape)

    # 피처 중요도를 저장할 데이터 프레임 선언
    fi = pd.DataFrame()
    fi['feature'] = features
    
    # LightGBM 데이터셋 선언
    dtrain = lgb.Dataset(x_train, label=y)

    # LightGBM 모델 훈련
    clf = lgb.train(
        model_params,
        dtrain,
        categorical_feature=categorical_features,
        verbose_eval=200
    )
    
    # 테스트 데이터 예측
    test_preds = clf.predict(x_test)

    # 피처 중요도 저장
    fi['importance'] = clf.feature_importance()
    
    return test_preds, fi


def make_lgb_oof_prediction(train, y, test, features, categorical_features='auto', model_params=None, folds=10, is_optuna=False):
    x_train = train[features]
    x_test = test[features]
    
    # 테스트 데이터 예측값을 저장할 변수
    test_preds = np.zeros(x_test.shape[0])
    
    # Out Of Fold Validation 예측 데이터를 저장할 변수
    y_oof = np.zeros(x_train.shape[0])
    
    # 폴드별 평균 Validation 스코어를 저장할 변수
    score = 0
    
    # 피처 중요도를 저장할 데이터 프레임 선언
    fi = pd.DataFrame()
    fi['feature'] = features
    
    # Stratified K Fold 선언
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(x_train, y)):
        # train index, validation index로 train 데이터를 나눔
        x_tr, x_val = x_train.loc[tr_idx, features], x_train.loc[val_idx, features]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        print(f'fold: {fold+1}, x_tr.shape: {x_tr.shape}, x_val.shape: {x_val.shape}')

        # LightGBM 데이터셋 선언
        dtrain = lgb.Dataset(x_tr, label=y_tr)
        dvalid = lgb.Dataset(x_val, label=y_val)
        
        # LightGBM 모델 훈련
        clf = lgb.train(
            model_params,
            dtrain,
            valid_sets=[dtrain, dvalid], # Validation 성능을 측정할 수 있도록 설정
            categorical_feature=categorical_features,
            verbose_eval=200
        )

        # Validation 데이터 예측
        val_preds = clf.predict(x_val)
        
        # Validation index에 예측값 저장 
        y_oof[val_idx] = val_preds
        
        # 폴드별 Validation 스코어 측정
        print(f"Fold {fold + 1} | AUC: {roc_auc_score(y_val, val_preds)}")
        print('-'*80)

        # score 변수에 폴드별 평균 Validation 스코어 저장
        score += roc_auc_score(y_val, val_preds) / folds
        
        # 테스트 데이터 예측하고 평균해서 저장
        test_preds += clf.predict(x_test) / folds
        
        # 폴드별 피처 중요도 저장
        fi[f'fold_{fold+1}'] = clf.feature_importance()

        del x_tr, x_val, y_tr, y_val
        gc.collect()
        
    print(f"\nMean AUC = {score}") # 폴드별 Validation 스코어 출력
    print(f"OOF AUC = {roc_auc_score(y, y_oof)}") # Out Of Fold Validation 스코어 출력

    #wandb
    if is_optuna == False : 
        wandb.log({'[lgbm]prediction Mean AUC' : score})
        wandb.log({'[lgbm]prediction OOF AUC' : roc_auc_score(y, y_oof)})

    # 폴드별 피처 중요도 평균값 계산해서 저장 
    fi_cols = [col for col in fi.columns if 'fold_' in col]
    fi['importance'] = fi[fi_cols].mean(axis=1)
    
    return y_oof, test_preds, fi

def make_xgb_oof_prediction(train, y, test, features, model_params=None, folds=10,  is_optuna=False):
    x_train = train[features]
    x_test = test[features]
    
    # 테스트 데이터 예측값을 저장할 변수
    test_preds = np.zeros(x_test.shape[0])
    
    # Out Of Fold Validation 예측 데이터를 저장할 변수
    y_oof = np.zeros(x_train.shape[0])
    
    # 폴드별 평균 Validation 스코어를 저장할 변수
    score = 0
    
    # 피처 중요도를 저장할 데이터 프레임 선언
    fi = pd.DataFrame()
    fi['feature'] = features
    
    # Stratified K Fold 선언
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(x_train, y)):
        # train index, validation index로 train 데이터를 나눔
        x_tr, x_val = x_train.loc[tr_idx, features], x_train.loc[val_idx, features]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        print(f'fold: {fold+1}, x_tr.shape: {x_tr.shape}, x_val.shape: {x_val.shape}')
        
        # XGBoost 데이터셋 선언
        dtrain = xgb.DMatrix(x_tr, label=y_tr)
        dvalid = xgb.DMatrix(x_val, label=y_val)
        
        # XGBoost 모델 훈련
        clf = xgb.train(
            model_params,
            dtrain,
            num_boost_round=10000, # 트리 개수
            evals=[(dtrain, 'train'), (dvalid, 'valid')],  # Validation 성능을 측정할 수 있도록 설정
            verbose_eval=200,
            early_stopping_rounds=100
        )
        
        # Validation 데이터 예측
        val_preds = clf.predict(dvalid)
        
        # Validation index에 예측값 저장 
        y_oof[val_idx] = val_preds
        
        # 폴드별 Validation 스코어 출력
        print(f"Fold {fold + 1} | AUC: {roc_auc_score(y_val, val_preds)}")
        print('-'*80)

        # score 변수에 폴드별 평균 Validation 스코어 저장
        score += roc_auc_score(y_val, val_preds) / folds
        
        # 테스트 데이터 예측하고 평균해서 저장
        test_preds += clf.predict(xgb.DMatrix(x_test)) / folds

        # 폴드별 피처 중요도 저장
        fi_tmp = pd.DataFrame.from_records([clf.get_score()]).T.reset_index()
        fi_tmp.columns = ['feature',f'fold_{fold+1}']
        fi = pd.merge(fi, fi_tmp, on='feature')

        del x_tr, x_val, y_tr, y_val
        gc.collect()
        
    print(f"\nMean AUC = {score}") # 폴드별 평균 Validation 스코어 출력
    print(f"OOF AUC = {roc_auc_score(y, y_oof)}") # Out Of Fold Validation 스코어 출력

    #wandb
    if is_optuna == False :
        wandb.log({'[xgb]perdictiona Mean AUC' : score})
        wandb.log({'[xgb]perdictiona OOF AUC' : roc_auc_score(y, y_oof)})
        
    # 폴드별 피처 중요도 평균값 계산해서 저장
    fi_cols = [col for col in fi.columns if 'fold_' in col]
    fi['importance'] = fi[fi_cols].mean(axis=1)
    
    return y_oof, test_preds, fi

def make_cat_oof_prediction(train, y, test, features, categorical_features=None, model_params=None, folds=10,  is_optuna=False):
    x_train = train[features]
    x_test = test[features]
    
    # 테스트 데이터 예측값을 저장할 변수
    test_preds = np.zeros(x_test.shape[0])
    
    # Out Of Fold Validation 예측 데이터를 저장할 변수
    y_oof = np.zeros(x_train.shape[0])
    
    # 폴드별 평균 Validation 스코어를 저장할 변수
    score = 0
    
    # 피처 중요도를 저장할 데이터 프레임 선언
    fi = pd.DataFrame()
    fi['feature'] = features
    
    # Stratified K Fold 선언
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)

    for fold, (tr_idx, val_idx) in enumerate(skf.split(x_train, y)):
        # train index, validation index로 train 데이터를 나눔
        x_tr, x_val = x_train.loc[tr_idx, features], x_train.loc[val_idx, features]
        y_tr, y_val = y[tr_idx], y[val_idx]
        
        print(f'fold: {fold+1}, x_tr.shape: {x_tr.shape}, x_val.shape: {x_val.shape}')
        
        # CatBoost 모델 훈련
        clf = CatBoostClassifier(**model_params)
        clf.fit(x_tr, y_tr,
                eval_set=(x_val, y_val), # Validation 성능을 측정할 수 있도록 설정
                cat_features=categorical_features,
                use_best_model=True,
                verbose=True)
        
        # Validation 데이터 예측
        val_preds = clf.predict_proba(x_val)[:,1]
        
        # Validation index에 예측값 저장 
        y_oof[val_idx] = val_preds
        
        # 폴드별 Validation 스코어 출력
        print(f"Fold {fold + 1} | AUC: {roc_auc_score(y_val, val_preds)}")
        print('-'*80)

        # score 변수에 폴드별 평균 Validation 스코어 저장
        score += roc_auc_score(y_val, val_preds) / folds
        
        # 테스트 데이터 예측하고 평균해서 저장
        test_preds += clf.predict_proba(x_test)[:,1] / folds

        # 폴드별 피처 중요도 저장
        fi[f'fold_{fold+1}'] = clf.feature_importances_
        
        del x_tr, x_val, y_tr, y_val
        gc.collect()
        
    print(f"\nMean AUC = {score}") # 폴드별 평균 Validation 스코어 출력
    print(f"OOF AUC = {roc_auc_score(y, y_oof)}") # Out Of Fold Validation 스코어 출력

    #wandb
    if is_optuna == False : 
        wandb.log({'[cat]perdictiona Mean AUC' : score})
        wandb.log({'[cat]perdictiona OOF AUC' : roc_auc_score(y, y_oof)})

    # 폴드별 피처 중요도 평균값 계산해서 저장
    fi_cols = [col for col in fi.columns if 'fold_' in col]
    fi['importance'] = fi[fi_cols].mean(axis=1)
    
    return y_oof, test_preds, fi
