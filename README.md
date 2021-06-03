# BoostCamp P stage 2
## -tabular data(shopping total cost prediction) 

# 프로젝트 개요

* 온라인 거래 고객 log 데이터를 이용하여 고객들의 미래 소비를 예측 분석프로젝트
* 12월을 대상으로 고객들에게 프로모션을 통해 성공적인 마케팅을 하기 위해 모델 제작
* 2011년 12월의 고객 구매액 300초과 여부를 예측
* log 데이터 범위 : 2009년 12월 ~ 2011년 11월
* ![img](https://s3-ap-northeast-2.amazonaws.com/aistages-public-junyeop/app/Users/00000025/files/56bd7d05-4eb8-4e3e-884d-18bd74dc4864..png)



# 코드 설명

### 코드 실행 방법

> 1. terminal 실행
> 2. inference_1.py가 존재하는 디렉토리로 먼저 이동
> 3. python3 inference_1.py "모델 이름/성능측정 이름" 입력
>    * 가능한 "모델 이름" : baseline1, lgbm, xgb, cat
>    * 가능한 "성능측정 이름" : lgbm_optuna, xgb_optuna, cat_optuna



### 코드 목록

> * [features_1.py](#features_1.py)
>   * data에 대한 분류 label 설정 함수, 데이터 전처리 진행 함수, feature engineering 진행 함수 정의
> * [model.py](#model.py)
>   * optuna를 위한 objective 정의, 모델에 따른 학습 진행하는 함수 정의
> * [util_1.py](#util_1.py)
>   * 시드(seed) 고정 함수, 예측 결과에 관한 평가 지표 출력 함수 정의
> * [inference_1.py](#inference_1.py)
>   * 모델 이름을 인자로 받아 각 모델에 맞는 학습을 진행, 예측 결과 저장



### 코드 주요 기능 설명



#### features_1.py

* 함수 목록
  1. generate_label2(df, year_month, total_thres=TOTAL_THRES, print_log=False)
     * 예측 연월에 대응하는 참값(label) 계산
     * parameter
       * df : dataframe, 대상 데이터
       * year_month : 예측 연월(ex. 2011년 12월)
       * total_thres : total 판단 기준값(300)
       * print_log : log를 남기는 지 판단하는 flag
     * return
       * label : 예측 연월에 대한 판별값
         * total이 300을 넘으면 1, 그 반대는 0
  2. feature_preprocessing2(train, test, features, do_imputing=True)
     * 데이터 전처리 진행 함수
     * parameter
       * train : train 데이터
       * test : test 데이터
       * features : feature 목록
       * do_imputing : 결측치 존재하는 지 판단하는 flag
         * 존재할 시 결측치는 중위값으로 채움
     * return 
       * x_tr : train data를 전처리한 결과
       * x_te : test data를 전처리한 결과
  3. feature_engineering2(df, year_month, time_term=1)
     * feature 생성 함수
     * parameter
       * df : dataframe, 대상 데이터
       * year_month : 예측 연월(ex. 2011년 12월)
       * time_term : train data의 예측 범위 지정
     * return 
       * x_tr : train data를 전처리한 결과
       * x_te : test data를 전처리한 결과
       * all_train_data['label'] : train data에 대한 판단값
         * total이 300을 넘으면 1, 그 반대는 0
       * features : 생성한 feature data



#### model.py

* 함수 목록

  1. lgb_objective(trial)

     * optuna를 이용한 hyper parameter 측정에 사용하는 objective 정의
     * 사용 모델 : LightBGM
     * parameter
       * trial :  hyper parameter 측정 진행 횟수
     * return
       * val_auc : validataion score 계산값
         * score 종류 : ROC-AUC

  2. xgb_objective(trial)

     * optuna를 이용한 hyper parameter 측정에 사용하는 objective 정의
     * 사용 모델 : XGBoost
     * parameter
       * trial :  hyper parameter 측정 진행 횟수
     * return
       * val_auc : validataion score 계산값
         * score 종류 : ROC-AUC

  3. cat_objective(trial)

     * optuna를 이용한 hyper parameter 측정에 사용하는 objective 정의
     * 사용 모델 : CatBoost
     * parameter
       * trial :  hyper parameter 측정 진행 횟수
     * return
       * val_auc : validataion score 계산값
         * score 종류 : ROC-AUC

  4. baseline_no_ml(df, year_month, total_thres=TOTAL_THRES)

     * 머신러닝 모델 없이 입력인자으로 받는 year_month의 이전 달 총 구매액을 구매 확률로 예측하는 베이스라인 모델
     * parameter
       * df : dataframe, 대상 데이터
       * year_month : 예측 연월(ex. 2011년 12월)
       * total_thres : total 판단 기준값(TOTAL_THRES = 300)
     * return
       * month['probability'] : 예측 결과값

  5. make_lgb_prediction(train, y, test, features, categorical_features='auto', model_params=None)

     * LightBGM 모델을 이용해 학습 진행하고 결과를 예측하는 함수

     * parameter
       * train : train data
       * y : train data 에 대한 label 값
       * test : 예측대상
       * features : feature 목록
       * categorical_features : 카테고리 타입인 feature 목록
       * model_params : 모델 파라미터
     * return
       * test_preds : 예측 결과값
       * fi : feature 중요도 저장 dataframe

  6. make_lgb_oof_prediction(train, y, test, features, categorical_features='auto', model_params=None, folds=10, is_optuna=False)

     * LightBGM 모델을 이용하고 stratified k-fold cross validation으로 결과 예측하는 함수

     * parameter
       * train : train data
       * y : train data 에 대한 label 값
       * test : 예측대상
       * features : feature 목록
       * categorical_features : 카테고리 타입인 feature 목록
       * model_params : 모델 파라미터
       * folds : stratified k-fold cross validation의 fold 수
       * is_optuna : optuna 사용 중 인지 판별하는 flag
     * return
       * y_oof : Cross Validation Out Of Fold 실행 결과
       * test_preds : 예측 결과값
       * fi : feature 중요도 저장 dataframe

  7. make_xgb_oof_prediction(train, y, test, features, model_params=None, folds=10, is_optuna=False)

     * XGBoost 모델을 이용하고 stratified k-fold cross validation으로 결과 예측하는 함수

     * parameter
       * train : train data
       * y : train data 에 대한 label 값
       * test : 예측대상
       * features : feature 목록
       * categorical_features : 카테고리 타입인 feature 목록
       * model_params : 모델 파라미터
       * folds : stratified k-fold cross validation의 fold 수
       * is_optuna : optuna 사용 중 인지 판별하는 flag
     * return
       * y_oof : Cross Validation Out Of Fold 실행 결과
       * test_preds : 예측 결과값
       * fi : feature 중요도 저장 dataframe

  8. make_cat_oof_prediction(train, y, test, features, categorical_features=None, model_params=None, folds=10, is_optuna=False)

     * CatBoost모델을 이용하고 stratified k-fold cross validation으로 결과 예측하는 함수

     * parameter
       * train : train data
       * y : train data 에 대한 label 값
       * test : 예측대상
       * features : feature 목록
       * categorical_features : 카테고리 타입인 feature 목록
       * model_params : 모델 파라미터
       * folds : stratified k-fold cross validation의 fold 수
       * is_optuna : optuna 사용 중 인지 판별하는 flag
     * return
       * y_oof : Cross Validation Out Of Fold 실행 결과
       * test_preds : 예측 결과값
       * fi : feature 중요도 저장 dataframe



#### util_1.py

* 함수 목록

  1. seed_everything(seed=0)

     * 매번 동일한 환경에서 모델 학습을 진행할 수 있도록 seed를 고정
     * 별도의 지정이 없으면 seed는 0으로 설정

  2. print_score(label, pred, prob_thres=0.5)

     * 예측 결과에 대한 평가 지표 출력

     * parameter

       * label : 참값
       * pred : 예측값
       * prob_thres : 확률 예측 기준값

       

#### inference_1.py

* 진행 과정

  1. 모델 이름/성능측정 이름 인자를 받아서 model 변수에 저장
  2. 데이터 파일(csv file) 읽기
  3. 필요 변수 설정
     * 예측 연월, wandb project name 설정
  4. model 변수에 따라 학습 진행하고 2011년 12월 예측 결과를 저장
     * 사용 가능한 모델
       * baseline1 : 머신러닝을 이용하지 않는 모델
       * lgbm : LightBGM
       * xgb : XGBoost
       * cat : CatBoost
     * lgbm, xgb, cat의 경우 
       * 2011년 11월에 대한 예측 실행, 결과에 대한 성능 예측 확인
       * 그후 2011년 12월에 대한 예측 실행
       * 순서
         1. parameter
            * 경우에 따라 optuna를 이용해서 얻은 parameter를 이용할 수 있음
         2. feature engineering 실행해서 train, test, y, feature 을 설정
         3. 학습을 진행해서 결과 예측값(test_pred) 계산
     * 성능측정 시 사용 가능한 모델
       * lgbm_optuna : LightBGM
       * xgb_optuna : XGBoost
       * cat_optuna : CatBoost
  5. 예측 결과를 csv file 로 생성하고 저장

  
