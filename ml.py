import numpy as np
import os
from utils import get_data,sp
from config import cfg
from sklearn.svm import SVC,LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier,StackingClassifier,VotingClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score,roc_auc_score,recall_score,f1_score,precision_score,make_scorer
from sklearn.model_selection import StratifiedKFold,cross_validate 
from PIL import Image
import argparse

def get_m(clf,x,y,name=None):
    y_pred = clf.predict(x)
    acc,se,spe,pr,f1 = accuracy_score(y,y_pred),recall_score(y,y_pred),sp(y,y_pred),precision_score(y,y_pred),f1_score(y,y_pred)
    if name =='SVM':
        auc = roc_auc_score(y,clf.decision_function(x))
    else:
        auc = roc_auc_score(y,clf.predict_proba(x)[:,1])
    print('{} test on datasetb: acc:{}, auc:{}, se:{}, sp:{}, pr:{}, f1:{}'.format(name,acc,auc,se,spe,pr,f1))

def cross_val(clf,data,name = ''):
  x,y = get_data(cfg[data].label)
  score = {'acc':'accuracy','auc':'roc_auc','se':'recall','sp':make_scorer(sp),'pr':'precision','f1':'f1'}
  skf = StratifiedKFold(n_splits=5)
  rs = cross_validate(clf,x,y,scoring = score,cv=skf)
  print('{} cross results on {}: acc:{}, auc:{}, se:{}, sp:{}, pr:{}, f1:{}'.format(name,data,rs['test_acc'].mean(),rs['test_auc'].mean(),rs['test_se'].mean(),rs['test_sp'].mean(),rs['test_pr'].mean(),rs['test_f1'].mean()))

def transfer(clf,name = ''):
  x_train,y_train = get_data(cfg['busi'].label)
  x_test,y_test = get_data(cfg['db'].label)
  clf.fit(x_train,y_train)
  get_m(clf,x_test,y_test,name)

parser = argparse.ArgumentParser(description='feature_select')
parser.add_argument('--mode', '-m', default='cross', type=str, choices = ['corss','transfer'], help='cross validation or test transfer performance')
parser.add_argument('--data', '-d', default='busi', type=str, choices = ['busi','db'], help='choose datasets')
args = parser.parse_args()

if __name__ == '__main__':

  rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced',max_depth=5,random_state=1)
  svm = make_pipeline(StandardScaler(),SVC())

  if args.mode == 'cross':
    cross_val(rf,args.data,'Random Forest')
    cross_val(svm,args.data,'SVM')
  else:
    transfer(rf,'Random Forest')
    transfer(svm,'SVM')

