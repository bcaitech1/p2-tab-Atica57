# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import os, sys, gc, random
import datetime
import dateutil.relativedelta
import argparse

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

'''
    머신러닝 모델 없이 입력인자으로 받는 year_month의 이전 달 총 구매액을 구매 확률로 예측하는 베이스라인 모델
'''

if __name__ == '__main__':

    # 인자 파서 선언
    parser = argparse.ArgumentParser()
    
    # baseline 모델 이름 인자로 받아서 model 변수에 저장
    parser.add_argument('model', type=str, default='baseline1', help="set baseline model name among baselin1,basline2,baseline3, lgbt, xgb, cat")
    args = parser.parse_args()
    model = args.model
    print('baseline model:', model)
    
    # 데이터 파일 읽기
    data = pd.read_csv(data_dir + '/train.csv', parse_dates=['order_date'])

    # 예측할 연월 설정
    year_month_for_val = '2011-11' #성능측정용 
    label_for_val = generate_label2(data, year_month_for_val)['label']
    year_month = '2011-12'
    time_term = 1 # 이전 월 계산에 이용(train, val set 구분 기준)

    #wandb 설정
    proj_specific = "_baseline3_based"
    lgbm_project_name = 'lgbm_test' + proj_specific
    xgb_project_name = 'xgb_test' + proj_specific
    cat_project_name = 'cat_test' + proj_specific
    baseline3_project_name = "baseline3_test" + proj_specific

    optuna_specific = "_optuna"
    lgbm_optuna_proj_name = 'lgbm_test' + optuna_specific
    xgb_optuna_proj_name = 'xgb_test' + optuna_specific
    cat_optuna_proj_name = 'cat_test' + optuna_specific

    ensemble_project_spec = "_lgbm_xgb_cat"
    ensemble_project_name = "ensemble_test" + ensemble_project_spec


    if model == 'baseline1': # baseline 모델 1
        test_preds = baseline_no_ml(data, year_month)
    
    #LightGBM, feature engineering2 이용
    elif model == 'lgbm': # lgbm
        model_params = {
            'objective': 'binary', # 이진 분류
            'boosting_type': 'gbdt',
            'metric': 'auc', # 평가 지표 설정
            'feature_fraction': 0.8, # 피처 샘플링 비율
            'bagging_fraction': 0.8, # 데이터 샘플링 비율
            'bagging_freq': 1,
            'n_estimators': 10000, # 트리 개수
            'early_stopping_rounds': 100,
            'seed': SEED,
            'verbose': -1,
            'n_jobs': -1,    
        }

        #optuna로 얻은 parameter 값 설정
        model_params_optuna = {
            'objective': 'binary', # 이진 분류
            'boosting_type': 'gbdt',
            'metric': 'auc', # 평가 지표 설정
            'num_leaves': 51, # num_leaves 값을 2-256까지 정수값 중에 사용
            'max_bin': 217, # max_bin 값을 128-256까지 정수값 중에 사용
            # min_data_in_leaf 값을 10-40까지 정수값 중에 사용
            'min_data_in_leaf': 25,
            # 피처 샘플링 비율을 0.4-1.0까지 중에 uniform 분포로 사용
            'feature_fraction': 0.6305316867933471,
            # 데이터 샘플링 비율을 0.4-1.0까지 중에 uniform 분포로 사용
            'bagging_fraction': 0.2813784915565985,
            # 데이터 샘플링 횟수를 1-7까지 정수값 중에 사용
            'bagging_freq': 4,
            'n_estimators': 10000, # 트리 개수
            'early_stopping_rounds': 159,
            # L1 값을 1e-8-10.0까지 로그 uniform 분포로 사용
            'lambda_l1': 5.7868521928893335,
            # L2 값을 1e-8-10.0까지 로그 uniform 분포로 사용
            'lambda_l2': 1.3182546836846302e-07,
            'seed': SEED,
            'verbose': -1,
            'n_jobs': -1,     
        }

        # wandb
        wandb.init(project=lgbm_project_name, config=model_params_optuna)
        
        # validation 성능 측정
        print("<************Validation***********>\n")
        train_for_val, test_for_val, \
            y_for_val, features_for_val = feature_engineering2(data, year_month_for_val)
        y_oof_val, test_preds_val, fi_val = \
            make_lgb_oof_prediction(train_for_val, y_for_val, \
                test_for_val, features_for_val, model_params=model_params_optuna)
        #성능 결과 출력
        print_score(label_for_val, test_preds_val); 
    
        print("\n<****************TEST****************>\n")
        # 피처 엔지니어링 실행
        train, test, y, features = feature_engineering2(data, year_month)
        print("number of feature =", features.shape)
        #wandb.run.summary.update({'features num' : features.shape[0]})
        wandb.log({'features num' : features.shape[0]})

        # Cross Validation Out Of Fold로 LightGBM 모델 훈련 및 예측
        y_oof, test_preds, fi = make_lgb_oof_prediction(train, y, test, features, model_params=model_params_optuna)
    

    # lightBGM optuna 실행 및 출력
    elif model == 'lgbm_optuna' : #optuna
        study = optuna.create_study(direction='maximize')
        study.optimize(lgb_objective, n_trials=20) # 7회 동안 하이퍼 파라미터 탐색

        print("best_params\n", study.best_params, "\n")
        print("best_value : ", study.best_value)
        

    # XGBOOST
    elif model == 'xgb': # XGBOOST
        model_params = {
            'objective': 'binary:logistic', # 이진 분류
            'learning_rate': 0.1, # 학습률
            'max_depth': 6, # 트리 최고 깊이
            'colsample_bytree': 0.8, # 피처 샘플링 비율
            'subsample': 0.8, # 데이터 샘플링 비율
            'eval_metric': 'auc', # 평가 지표 설정
            'seed': SEED,
        } 

        #optuna로 얻은 parameter 값 설정
        model_params_optuna = {
            'objective': 'binary:logistic', # 이진 분류
            'learning_rate': 0.025936970485555613, # 학습률
            'max_depth': 3, # 트리 최고 깊이
            'colsample_bytree': 0.1590915485786206, # 피처 샘플링 비율
            'subsample': 0.14915913020164723, # 데이터 샘플링 비율
            'eval_metric': 'auc', # 평가 지표 설정
            'seed': SEED,
        } 

        # wandb
        wandb.init(project=xgb_project_name, config=model_params_optuna)
        
        # validation 성능 측정
        print("<************Validation***********>\n")
        train_for_val, test_for_val, \
            y_for_val, features_for_val = feature_engineering2(data, year_month_for_val)
        y_oof_val, test_preds_val, fi_val = \
            make_xgb_oof_prediction(train_for_val, y_for_val, \
                test_for_val, features_for_val, model_params=model_params_optuna)
        #성능 결과 출력
        print_score(label_for_val, test_preds_val)

        print("\n<****************TEST****************>\n")
        #피처 엔지니어링 실행
        train, test, y, features = feature_engineering2(data, year_month)
        print("number of feature =", features.shape)
        wandb.log({'feature num' : features.shape[0]})

        # Cross Validation Out Of Fold로 XGBOOST 모델 훈련 및 예측
        y_oof, test_preds, fi_xgb = make_xgb_oof_prediction(train, y, test, features, model_params=model_params_optuna)

    #xgboost_optuna 실행 및 출력
    elif model == 'xgb_optuna' : #optuna
        study = optuna.create_study(direction='maximize')
        study.optimize(xgb_objective, n_trials=5) # 5회 동안 하이퍼 파라미터 탐색

        print("best_params\n", study.best_params, "\n")
        print("best_value : ", study.best_value)
        
    #Catboost
    elif model == 'cat': # catboost
        model_params = {
            'n_estimators': 10000, # 트리 개수
            'learning_rate': 0.07, # 학습률 #0.07 ->0.01
            'eval_metric': 'AUC', # 평가 지표 설정
            'loss_function': 'Logloss', # 손실 함수 설정
            'random_seed': SEED,
            'metric_period': 100,
            'od_wait': 100, # early stopping round
            'depth': 6, # 트리 최고 깊이
            'rsm': 0.8, # 피처 샘플링 비율
        }

        #optuna로 얻은 parameter 값 설정
        model_params_optuna = {
            'n_estimators': 10000, # 트리 개수
            'learning_rate': 0.009264948818291454, # 학습률 #0.07 ->0.01
            'eval_metric': 'AUC', # 평가 지표 설정
            'loss_function': 'Logloss', # 손실 함수 설정
            'random_seed': SEED,
            'metric_period': 39,
            'od_wait': 128, # early stopping round
            'depth': 5, # 트리 최고 깊이
            'rsm': 0.6398179008529061, # 피처 샘플링 비율
        }
        # wandb
        wandb.init(project=cat_project_name, config=model_params_optuna)
        
        # validation 성능 측정
        print("<************Validation***********>\n")
        train_for_val, test_for_val, \
            y_for_val, features_for_val = feature_engineering2(data, year_month_for_val)
        y_oof_val, test_preds_val, fi_val = \
            make_cat_oof_prediction(train_for_val, y_for_val, \
                test_for_val, features_for_val, model_params=model_params_optuna)
        #성능 결과 출력
        print_score(label_for_val, test_preds_val)

        print("\n<****************TEST****************>\n")
        #피처 엔지니어링 실행
        train, test, y, features = feature_engineering2(data, year_month)
        print("number of feature =", features.shape)
        wandb.log({'feature num' : features.shape[0]})

        # Cross Validation Out Of Fold로 CatBoost 모델 훈련 및 예측
        y_oof, test_preds, fi_cat = make_cat_oof_prediction(train, y, test, features, model_params=model_params_optuna)

    #CatBoost optuna 실행 및 출력
    elif model == 'cat_optuna' : # optuna
        study = optuna.create_study(direction='maximize')
        study.optimize(cat_objective, n_trials=15) # 7회 동안 하이퍼 파라미터 탐색

        print("\nbest_params\n", study.best_params, "\n")
        print("best_value : ", study.best_value)

    else:
        test_preds = baseline_no_ml(data, year_month)
    
    # 테스트 결과 제출 파일 읽기
    sub = pd.read_csv(data_dir + '/sample_submission.csv')
    
    # 테스트 예측 결과 저장
    sub['probability'] = test_preds
    
    
    os.makedirs(output_dir, exist_ok=True)
    # 제출 파일 쓰기
    sub.to_csv(os.path.join(output_dir , 'output.csv'), index=False)