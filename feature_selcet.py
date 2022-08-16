# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 19:59:48 2021

@author: zs
"""

import csv
from sklearn.feature_selection import SelectKBest,SelectPercentile,chi2,f_classif,mutual_info_classif
from sklearn.feature_selection import RFE,RFECV
from sklearn.feature_selection import SelectFromModel,SequentialFeatureSelector
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LassoCV
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
from sklearn.svm import SVC,LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from time import time
from utils import get_radiomics
import os
from config import cfg
import numpy as np
from PIL import Image
from collections import Counter
from boruta import BorutaPy
import argparse

parser = argparse.ArgumentParser(description='feature_select')
parser.add_argument('--data', '-d', nargs='+', default=['busi'], type=str, choices = ['busi','db'], help='choose datasets')
args = parser.parse_args()

def get_data(fl,pl):
    for fs,p in zip(fl,pl):
      data=[]
      target = []
      names=''
      with open(fs,'r') as f:
        r = csv.DictReader(f)
        data = []
        target = []
        for line in r:
            img_p = os.path.join(p,line['Image'])
            mask_p = os.path.join(p,line['Image'][:-5]+'t.png')
            img = np.array(Image.open(img_p).convert('L'))
            mask = np.array(Image.open(mask_p).convert('L'))
            names,fts = get_radiomics(img,mask,'all')
            data.append([ft+0 for ft in fts])
            target.append(eval(line['Label']))
    return {'data':data,'target':target,'names':names}


train_path = []
train_label = []
for d in args.data:
    train_path.append(cfg[d].data_path)
    train_label.append(cfg[d].train_label)

data = get_data(train_label,train_path)
#print(data['names'],data['data'])
names = np.array(data['names'])
x = np.array(data['data'])
y = np.array(data['target'])

#scaler = MinMaxScaler()
scaler = StandardScaler()
x = scaler.fit_transform(x)
k = 10
with open('features.csv','w') as f:
    pass

ft = Counter()

print('Univariate feature selection')
start_time = time()
selector = SelectKBest(f_classif,k = k)
selector.fit(x,y)
s_index = np.argsort(selector.scores_)[::-1]
#print('F_measure:',names[selector.get_support()])
print('F_measure:',names[s_index[:k]])
print(selector.scores_[s_index[:k]])
print('Done in {:.2f}s'.format(time()-start_time))
with open('features.csv','a',newline='') as f:
    w = csv.writer(f)
    w.writerow(['F_measure']+list(names[s_index[:k]]))
for i in range(k):
    ft.update(names[s_index[i:i+1]])

start_time = time()
selector = SelectKBest(mutual_info_classif,k = 10)
selector.fit(x,y)
s_index = np.argsort(selector.scores_)[::-1]
#print('mutual_info:',names[selector.get_support()])
print('mutual_info:',names[s_index[:k]])
print(selector.scores_[s_index[:k]])
print('Done in {:.2f}s'.format(time()-start_time))
with open('features.csv','a',newline='') as f:
    w = csv.writer(f)
    w.writerow(['mutual_info']+list(names[s_index[:k]]))
for i in range(k):
    ft.update(names[s_index[i:i+1]])

print()
print('SelectFromModel')
start_time = time()
svc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(x,y)
selector = SelectFromModel(svc,prefit=True,max_features=k)
s_index = np.argsort(np.abs(svc.coef_)).squeeze()[::-1]
print('SelectFromModel with SVMl1:',names[s_index[:8]])
print(np.abs(svc.coef_))
print('Done in {:.2f}s'.format(time()-start_time))
with open('features.csv','a',newline='') as f:
    w = csv.writer(f)
    w.writerow(['svml1']+list(names[s_index[:8]]))
for i in range(8):
    ft.update(names[s_index[i:i+1]])


start_time = time()
clf = ExtraTreesClassifier(random_state=1).fit(x,y)
s_index = np.argsort(clf.feature_importances_)[::-1]
print('SelectFromModel with randomforest:',names[s_index[:k]])
print(clf.feature_importances_[s_index[:k]])
selector = SelectFromModel(clf,prefit=True,max_features=k)
print('Done in {:.2f}s'.format(time()-start_time))
with open('features.csv','a',newline='') as f:
    w = csv.writer(f)
    w.writerow(['randomforest']+list(names[s_index[:k]]))
for i in range(k):
    ft.update(names[s_index[i:i+1]])
    

'''start_time = time()
rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)
selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)
selector.fit(x,y)
print('Boruta--Randomforest:',names[selector.support_])
print('Done in {:.2f}s'.format(time()-start_time))
with open('features.csv','a',newline='') as f:
    w = csv.writer(f)
    w.writerow(['Boruta']+list(names[selector.support_]))'''
#for i in list(names[selector.support_]):
 #   ft.update([i])


print()
print('Recursive feature elimination')
start_time = time()
#svc = SVC(kernel='linear', C=1)
svc = LinearSVC(C=1, penalty="l2", dual=False)
rfe = RFE(estimator=svc, n_features_to_select=k, step=1)
rfe.fit(x,y)
print('REF--linearSVC:',names[rfe.support_])
print('Done in {:.2f}s'.format(time()-start_time))
with open('features.csv','a',newline='') as f:
    w = csv.writer(f)
    w.writerow(['RFE-svm']+list(names[rfe.support_]))
for i in list(names[rfe.support_]):
    ft.update([i])
    
'''start_time = time()
rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(2),scoring='accuracy',min_features_to_select=1)
rfecv.fit(x,y)
print('RFECV--linearSVC:',names[rfecv.support_])
print('Done in {:.2f}s'.format(time()-start_time))
with open('features.csv','a',newline='') as f:
    w = csv.writer(f)
    w.writerow(['RFEcv-svm']+list(names[rfecv.support_]))
for i in list(names[rfecv.support_]):
    ft.update([i])'''

print()
print('Sequential feature selector')
start_time = time()
#lasso = LassoCV().fit(x,y)
sfsfor = SequentialFeatureSelector(svc,n_features_to_select=k).fit(x,y)
print('sequential forward:',names[sfsfor.get_support()])
print('Done in {:.2f}s'.format(time()-start_time))
with open('features.csv','a',newline='') as f:
    w = csv.writer(f)
    w.writerow(['SFS-forward']+list(names[sfsfor.get_support()]))
for i in list(names[sfsfor.get_support()]):
    ft.update([i])

'''start_time = time()
sfsback = SequentialFeatureSelector(svc,n_features_to_select=k,direction='backward').fit(x,y)
print('sequential backward:',names[sfsback.get_support()])
print('Done in {:.2f}s'.format(time()-start_time))
with open('features.csv','a',newline='') as f:
    w = csv.writer(f)
    w.writerow(['SFS-backward']+list(names[sfsback.get_support()]))
for i in list(names[sfsback.get_support()]):
    ft.update([i])'''

with open('ft_{}.csv'.format(''.join(args.data)),'w') as f:
    w=csv.writer(f)
    for f,c in ft.items():
        if c>=3: w.writerow([f,c])
