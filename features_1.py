import pandas as pd
import numpy as np
import os, sys, gc, random
import datetime
import dateutil.relativedelta

# Machine learning
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Custom library
from utils import seed_everything, print_score


TOTAL_THRES = 300 # 구매액 임계값
SEED = 42 # 랜덤 시드
seed_everything(SEED) # 시드 고정

data_dir = '../input/train.csv' # os.environ['SM_CHANNEL_TRAIN']
model_dir = '../model' # os.environ['SM_MODEL_DIR']


'''
    입력인자로 받는 year_month에 대해 고객 ID별로 총 구매액이
    구매액 임계값을 넘는지 여부의 binary label을 생성하는 함수
'''

#generate_label
def generate_label2(df, year_month, total_thres=TOTAL_THRES, print_log=False):
    df = df.copy()
    
    # year_month에 해당하는 label 데이터 생성
    df['year_month'] = df['order_date'].dt.strftime('%Y-%m')
    df.reset_index(drop=True, inplace=True)

    # year_month 이전 월의 고객 ID 추출
    cust = df[df['year_month']<=year_month]['customer_id'].unique()
    # year_month에 해당하는 데이터 선택
    df = df[df['year_month']==year_month]
    
    # label 데이터프레임 생성
    label = pd.DataFrame({'customer_id':cust})
    label['year_month'] = year_month
    
    # year_month에 해당하는 고객 ID의 구매액의 합 계산
    grped = df.groupby(['customer_id','year_month'], as_index=False)[['total']].sum()
    
    # label 데이터프레임과 merge하고 구매액 임계값을 넘었는지 여부로 label 생성
    label = label.merge(grped, on=['customer_id','year_month'], how='left')
    label['total'].fillna(0.0, inplace=True)
    label['label'] = (label['total'] > total_thres).astype(int)

    # 고객 ID로 정렬
    label = label.sort_values('customer_id').reset_index(drop=True)
    if print_log: print(f'{year_month} - final label shape: {label.shape}')
    
    return label



# feature_preprocessing 
def feature_preprocessing2(train, test, features, do_imputing=True):
    x_tr = train.copy()
    x_te = test.copy()
    
    # 범주형 피처 이름을 저장할 변수
    cate_cols = []

    # 레이블 인코딩
    for f in features:
        if x_tr[f].dtype.name == 'object': # 데이터 타입이 object(str)이면 레이블 인코딩
            cate_cols.append(f)
            le = LabelEncoder()
            # train + test 데이터를 합쳐서 레이블 인코딩 함수에 fit
            le.fit(list(x_tr[f].values) + list(x_te[f].values))
            
            # train 데이터 레이블 인코딩 변환 수행
            x_tr[f] = le.transform(list(x_tr[f].values))
            
            # test 데이터 레이블 인코딩 변환 수행
            x_te[f] = le.transform(list(x_te[f].values))

    print('categorical feature:', cate_cols)

    if do_imputing:
        # 중위값으로 결측치 채우기
        imputer = SimpleImputer(strategy='median')

        x_tr[features] = imputer.fit_transform(x_tr[features])
        x_te[features] = imputer.transform(x_te[features])
    
    return x_tr, x_te


def feature_engineering2(df, year_month, time_term=1):
    '''
        year_month = 예측 연월(2011-12)
    '''
    df = df.copy()
    #
    df['order_date_year_month'] = df['order_date'].dt.strftime('%Y-%m')
    df['order_date_year'] = df['order_date'].dt.strftime('%Y')
    df['order_date_hour'] = df['order_date'].dt.strftime('%H')
    #분기별(4달씩 3구역 : 8~11/ 4~7/ 12~3)
    #per_3_month = ['12-3','8-11','4-7']
    
    # customer_id 기준으로 pandas group by 후 total, quantity, price 누적합 계산
    df['cumsum_total_by_cust_id'] = df.groupby(['customer_id'])['total'].cumsum()
    df['cumsum_quantity_by_cust_id'] = df.groupby(['customer_id'])['quantity'].cumsum()
    df['cumsum_price_by_cust_id'] = df.groupby(['customer_id'])['price'].cumsum()

    # product_id 기준으로 pandas group by 후 total, quantity, price 누적합 계산
    df['cumsum_total_by_prod_id'] = df.groupby(['product_id'])['total'].cumsum()
    df['cumsum_quantity_by_prod_id'] = df.groupby(['product_id'])['quantity'].cumsum()
    df['cumsum_price_by_prod_id'] = df.groupby(['product_id'])['price'].cumsum()
    
    # order_id 기준으로 pandas group by 후 total, quantity, price 누적합 계산
    df['cumsum_total_by_order_id'] = df.groupby(['order_id'])['total'].cumsum()
    df['cumsum_quantity_by_order_id'] = df.groupby(['order_id'])['quantity'].cumsum()
    df['cumsum_price_by_order_id'] = df.groupby(['order_id'])['price'].cumsum()    

    #######추가 feature - 1차 추가
    #country 기준으로 pandas group by 후 total, quantity, price 누적합 계산
    df['cumsum_total_by_country'] = df.groupby(['country'])['total'].cumsum()
    df['cumsum_quantity_by_country'] = df.groupby(['country'])['quantity'].cumsum()
    df['cumsum_price_by_country'] = df.groupby(['country'])['price'].cumsum()

    #order_date 기준 - year
    df['cumsum_total_by_year'] = df.groupby(['order_date_year'])['total'].cumsum()
    df['cumsum_quantity_by_year'] = df.groupby(['order_date_year'])['quantity'].cumsum()
    df['cumsum_price_by_year'] = df.groupby(['order_date_year'])['price'].cumsum()
    
    #order_date 기준 - year_month
    df['cumsum_total_by_year_month'] = df.groupby(['order_date_year_month'])['total'].cumsum()
    df['cumsum_quantity_by_year_month'] = df.groupby(['order_date_year_month'])['quantity'].cumsum()
    df['cumsum_price_by_year_month'] = df.groupby(['order_date_year_month'])['price'].cumsum()
    
    #######추가 feature - 2차 추가
    # hour 기준으로 pandas group by 후 total, quantity, price 누적합 계산
    df['cumsum_total_by_hour'] = df.groupby(['order_date_hour'])['total'].cumsum()
    df['cumsum_quantity_by_hour'] = df.groupby(['order_date_hour'])['quantity'].cumsum()
    df['cumsum_price_by_hour'] = df.groupby(['order_date_hour'])['price'].cumsum()
    
   
    # year_month 이전 월 계산
    d = datetime.datetime.strptime(year_month, "%Y-%m")
    prev_ym = d - dateutil.relativedelta.relativedelta(months=time_term)
    prev_ym = prev_ym.strftime('%Y-%m')
    '''
    train_ym = d - dateutil.relativedelta.relativedelta(months=time_term*2)
    train_ym = train_ym.strftime('%Y-%m')
    test_ym = d - dateutil.relativedelta.relativedelta(months=time_term)
    test_ym = test_ym.strftime('%Y-%m')
    '''

    # train, test 데이터 선택
    train = df[df['order_date'] < prev_ym]
    test = df[df['order_date'] < year_month]
    '''
    train = df[df['order_date'] <= train_ym]
    test = df[df['order_date'] <= test_ym] 
    '''

    # train, test 레이블 데이터 생성
    train_label = generate_label2(df, prev_ym)[['customer_id','year_month','label']]
    test_label = generate_label2(df, year_month)[['customer_id','year_month','label']]
    
    # group by aggregation 함수 선언
    agg_func = ['mean','max','min','sum','count','std','skew']
    agg_dict = {
        'quantity': agg_func,
        'price': agg_func,
        'total': agg_func,
        'cumsum_total_by_cust_id': agg_func,
        'cumsum_quantity_by_cust_id': agg_func,
        'cumsum_price_by_cust_id': agg_func,
        'cumsum_total_by_prod_id': agg_func,
        'cumsum_quantity_by_prod_id': agg_func,
        'cumsum_price_by_prod_id': agg_func,
        'cumsum_total_by_order_id': agg_func,
        'cumsum_quantity_by_order_id': agg_func,
        'cumsum_price_by_order_id': agg_func,
        'order_id': ['nunique'],
        'product_id': ['nunique'],
        # 1차 추가
        'cumsum_total_by_country' : agg_func,
        'cumsum_quantity_by_country' : agg_func,
        'cumsum_price_by_country' : agg_func,
        'cumsum_total_by_year' : agg_func,
        'cumsum_quantity_by_year' : agg_func,
        'cumsum_price_by_year' : agg_func,
        'cumsum_total_by_year_month' : agg_func,
        'cumsum_quantity_by_year_month' : agg_func,
        'cumsum_price_by_year_month' : agg_func,
        'country' : ['nunique'],
        #2차 추가
        'cumsum_total_by_hour' : agg_func,
        'cumsum_quantity_by_hour' : agg_func,
        'cumsum_price_by_hour' : agg_func,
        #3차 추가
        'order_date_year_month' : ['nunique'], 
        'order_date_year' : ['nunique'], 
        'order_date_hour' : ['nunique']
    } 

    all_train_data = pd.DataFrame()
    
    for i, tr_ym in enumerate(train_label['year_month'].unique()):
        # group by aggretation 함수로 train 데이터 피처 생성
        train_agg = train.loc[train['order_date'] < tr_ym].groupby(['customer_id']).agg(agg_dict)

        new_cols = []
        for col in agg_dict.keys():
            for stat in agg_dict[col]:
                if type(stat) is str:
                    new_cols.append(f'{col}-{stat}')
                else:
                    new_cols.append(f'{col}-mode')

        train_agg.columns = new_cols
        train_agg.reset_index(inplace = True)
        
        train_agg['year_month'] = tr_ym
        
        all_train_data = all_train_data.append(train_agg)
    
    all_train_data = train_label.merge(all_train_data, on=['customer_id', 'year_month'], how='left')
    features = all_train_data.drop(columns=['customer_id', 'label', 'year_month']).columns
    
    # group by aggretation 함수로 test 데이터 피처 생성
    test_agg = test.groupby(['customer_id']).agg(agg_dict)
    test_agg.columns = new_cols
    
    test_data = test_label.merge(test_agg, on=['customer_id'], how='left')

    # train, test 데이터 전처리
    x_tr, x_te = feature_preprocessing2(all_train_data, test_data, features)
    
    print('x_tr.shape', x_tr.shape, ', x_te.shape', x_te.shape)
    
    return x_tr, x_te, all_train_data['label'], features


if __name__ == '__main__':
    
    print('data_dir', data_dir)
