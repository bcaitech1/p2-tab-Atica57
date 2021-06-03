import os
import random
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import wandb


# 시드 고정 함수
def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)


# 평가 지표 출력 함수 - wandb 적용
def print_score(label, pred, prob_thres=0.5):
    print('[Val]Precision: {:.5f}'.format(precision_score(label, pred>prob_thres)))
    #wandb - precision
    wandb.log({'[Val]Precision': precision_score(label, pred>prob_thres)})         
    print('[Val]Recall: {:.5f}'.format(recall_score(label, pred>prob_thres)))
    #wandb - Recall
    wandb.log({'[Val]Recall': recall_score(label, pred>prob_thres)})
    print('[Val]F1 Score: {:.5f}'.format(f1_score(label, pred>prob_thres)))
    #wandb - F1 Score
    wandb.log({'[Val]F1 Score': f1_score(label, pred>prob_thres)})
    print('[Val]ROC AUC Score: {:.5f}'.format(roc_auc_score(label, pred)))
    #wandb - ROC AUC Score
    wandb.log({'[Val]ROC AUC Score': roc_auc_score(label, pred)})
