#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 11:29:40 2018

@author: hidemiasakura

machine learning signal processor
inputs futures
output signals
"""

import sys
import time
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
import calendar
from datetime import datetime as dt
from multiprocessing import Process, Queue

#from nimbus.process.mean_reversion import MeanReversion
from nimbus.process.zigzag import zigzag
from nimbus.process.transform import to_signals, DPO, atr_df
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.pipeline import Pipeline
#sklearn grid_search, cross_validation -->model search
#from sklearn.grid_search import ParameterGrid
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2, f_classif, RFECV
#classification
#from sklearn.cross_validation import StratifiedShuffleSplit
#from sklearn.metrics import confusion_matrix
from sklearn.linear_model import Perceptron, PassiveAggressiveClassifier, LogisticRegression,\
                                LogisticRegressionCV, RidgeClassifier, RidgeClassifierCV,\
                                SGDClassifier, BayesianRidge
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,\
                        BaggingClassifier, ExtraTreesClassifier, VotingClassifier, ExtraTreesRegressor
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import LinearSVC, SVC, NuSVC, SVR
from sklearn.neighbors import RadiusNeighborsClassifier, KNeighborsClassifier,\
                                NearestCentroid
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.semi_supervised import LabelPropagation
from sklearn.semi_supervised import LabelSpreading
from sklearn.gaussian_process import GaussianProcessClassifier
import warnings
warnings.simplefilter('error')



class ML(object):
    
    def __init__(self, futures):
        self.futures=futures
        self.last_date=futures.last_date
        self.FEATURE_COMPRESSION=10
        self.MIN_DATAPOINTS=3
        self.signals_dict={}
        self.training_dict={}
        self.strategies={}
        self.score=pd.DataFrame()
        self.strategy_types=[]
        
        #ML PARAMS
        dt_stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
        self.rfe_estimator = [ 
                #("None","None"),
                #("GradientBoostingRegressor",GradientBoostingRegressor()),
                ("DecisionTreeRegressor",DecisionTreeRegressor()),
                #("ExtraTreeRegressor",ExtraTreesRegressor()),
                #("BayesianRidge", BayesianRidge()), #no change to x_new
                #('LogisticRegression',LogisticRegression()),
                #('SVR',SVR()),
                 ]
        
        self.fselect_models = [
                 ('PCA'+str(self.FEATURE_COMPRESSION),PCA(n_components=self.FEATURE_COMPRESSION)),\
                 ('SelectKBest'+str(self.FEATURE_COMPRESSION),SelectKBest(f_classif, k=self.FEATURE_COMPRESSION))\
                 ]
        
        self.all_models = [
                 #("LR", LogisticRegression(class_weight={1:1})), \
                 #("PRCEPT", Perceptron(class_weight={1:1})), \
                 #("PAC", PassiveAggressiveClassifier(class_weight={1:1})), \
                 #("LSVC", LinearSVC()), \
                 #('BernoulliNB', BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None)),
                 #("GNBayes",GaussianNB()),\
                 #("LDA", LinearDiscriminantAnalysis()), \
                 #("QDA", QuadraticDiscriminantAnalysis()), \
                 #("MLPC", MLPClassifier([Layer("Sigmoid", units=150), Layer("Softmax")],learning_rate=0.001, n_iter=25, verbose=True)),
                 #("rbf1SVM", SVC(C=1, gamma=.01, cache_size=200, class_weight={1:1}, kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)), \
                 #("rbf10SVM", SVC(C=10, gamma=.01, cache_size=200, class_weight={-1:500}, kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)), \
                 #("polySVM", SVC(C=1, gamma=.01, cache_size=200, class_weight={1:500}, coef0=0.0, degree=3, kernel='poly', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)), \
                 #("sigSVM", SVC(C=1, gamma=.01, cache_size=200, class_weight={1:500}, coef0=0.0, degree=3, kernel='sigmoid', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)), \
                 #("NuSVM", NuSVC(nu=0.9, kernel='rbf', degree=3, gamma=.100, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, verbose=False, max_iter=-1, random_state=None)),\
                 #("ada_discrete", AdaBoostClassifier(base_estimator=dt_stump, learning_rate=1, n_estimators=400, algorithm="SAMME")),\
                 #("ada_real", AdaBoostClassifier(base_estimator=dt_stump,learning_rate=1,n_estimators=180,algorithm="SAMME.R")),\
                 #("GBC", GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')),\
                 #("Bagging",BaggingClassifier(base_estimator=dt_stump, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0)),\
                 #("ETC", ExtraTreesClassifier(class_weight={1:1}, n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=False, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False)),\
                 #("RF", RandomForestClassifier(class_weight={1:1}, n_estimators=10, criterion='gini',max_depth=3, min_samples_split=2, min_samples_leaf=1, max_features='auto', bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0))\
                 #("kNeighbors-uniform", KNeighborsClassifier(n_neighbors=5, weights='uniform')),\
                 #("kNeighbors-distance", KNeighborsClassifier(n_neighbors=15, weights='distance')),\
                 #("rNeighbors-uniform", RadiusNeighborsClassifier(radius=8, weights='uniform')),\
                 #("rNeighbors-distance", RadiusNeighborsClassifier(radius=10, weights='distance')),\
                 #("VotingHard", VotingClassifier(estimators=[\
                     #("ada_discrete", AdaBoostClassifier(base_estimator=dt_stump, learning_rate=1, n_estimators=400, algorithm="SAMME")),\
                     #("ada_real", AdaBoostClassifier(base_estimator=dt_stump,learning_rate=1,n_estimators=180,algorithm="SAMME.R")),\
                     #("GBC", GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')),\
                     #("QDA", QuadraticDiscriminantAnalysis()),\
                     #("GNBayes",GaussianNB()),\
                     #("LDA", LinearDiscriminantAnalysis()), \
                     #("kNeighbors-uniform", KNeighborsClassifier(n_neighbors=5, weights='uniform')),\
                     #("MLPC", Classifier([Layer("Sigmoid", units=150), Layer("Softmax")],learning_rate=0.001, n_iter=25, verbose=True)),\
                     #("rbfSVM", SVC(C=1, gamma=.01, cache_size=200, class_weight={1:500}, kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)), \
                     #("kNeighbors-distance", KNeighborsClassifier(n_neighbors=5, weights='distance')),\
                     #("Bagging",BaggingClassifier(base_estimator=dt_stump, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0)),\
                     #("ETC", ExtraTreesClassifier(class_weight={1:1}, n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=False, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False)),\
                        #], voting='hard', weights=None)),
                 #("VotingSoft", VotingClassifier(estimators=[\
                     #("ada_discrete", AdaBoostClassifier(base_estimator=dt_stump, learning_rate=1, n_estimators=400, algorithm="SAMME")),\
                     #("ada_real", AdaBoostClassifier(base_estimator=dt_stump,learning_rate=1,n_estimators=180,algorithm="SAMME.R")),\
                     #("GBC", GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')),\
                     #("QDA", QuadraticDiscriminantAnalysis()),\
                     #("GNBayes",GaussianNB()),\
                     #("LDA", LinearDiscriminantAnalysis()), \
                     #("kNeighbors-uniform", KNeighborsClassifier(n_neighbors=self.MIN_DATAPOINTS, weights='uniform')),\
                     #("MLPC", Classifier([Layer("Sigmoid", units=150), Layer("Softmax")],learning_rate=0.001, n_iter=25, verbose=True)),\
                     #("rbfSVM", SVC(C=1, gamma=.01, cache_size=200, class_weight={1:1}, kernel='rbf', max_iter=-1, probability=True, random_state=None, shrinking=True, tol=0.001, verbose=False)), \
                     #("kNeighbors-distance", KNeighborsClassifier(n_neighbors=8, weights='distance')),\
                     #("Bagging",BaggingClassifier(base_estimator=dt_stump, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0)),\
                     #("ETC", ExtraTreesClassifier(class_weight={1:1}, n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=False, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False)),\
                     #    ], voting='soft', weights=None)),
                 ]
        
        self.non_models = [
                 ("PRCEPT", Perceptron()),
                 ("PAC", PassiveAggressiveClassifier()),
                 ("LDA", LinearDiscriminantAnalysis()),
                 ("QDA", QuadraticDiscriminantAnalysis()),
                 ("MLPC", MLPClassifier()),
                 ("NuSVM", NuSVC(nu=0.5, kernel='rbf', degree=3, gamma=.100, coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, verbose=False, max_iter=-1, random_state=None)),\
                 ("rNeighbors-uniform", RadiusNeighborsClassifier(radius=8, weights='uniform')),\
                 ("rNeighbors-distance", RadiusNeighborsClassifier(radius=10, weights='distance')),\
                 ('NearestCentroid', NearestCentroid()),
                 ('SGDClassifier',SGDClassifier()),
                 #("VotingHard", VotingClassifier(estimators=[\
                     #("ada_discrete", AdaBoostClassifier(base_estimator=dt_stump, learning_rate=1, n_estimators=400, algorithm="SAMME")),\
                     #("ada_real", AdaBoostClassifier(base_estimator=dt_stump,learning_rate=1,n_estimators=180,algorithm="SAMME.R")),\
                     #("GBC", GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')),\
                     #("QDA", QuadraticDiscriminantAnalysis()),\
                     #("GNBayes",GaussianNB()),\
                     #("LDA", LinearDiscriminantAnalysis()), \
                     #("kNeighbors-uniform", KNeighborsClassifier(n_neighbors=5, weights='uniform')),\
                     #("MLPC", Classifier([Layer("Sigmoid", units=150), Layer("Softmax")],learning_rate=0.001, n_iter=25, verbose=True)),\
                     #("rbfSVM", SVC(C=1, gamma=.01, cache_size=200, kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)), \
                     #("kNeighbors-distance", KNeighborsClassifier(n_neighbors=5, weights='distance')),\
                     #("Bagging",BaggingClassifier(base_estimator=dt_stump, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0)),\
                     #("ETC", ExtraTreesClassifier(n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=False, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False)),\
                        #], voting='hard', weights=None)),
                ]
        self.models = [
                 #("LR", LogisticRegression()),
                 #("LR_CV", LogisticRegressionCV()),
                 #('RidgeClassifier',RidgeClassifier()),
                 #('RidgeClassifierCV', RidgeClassifierCV()),
                 #("LSVC", LinearSVC()),
                 #('BernoulliNB', BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None)),
                 #("GNBayes",GaussianNB()),
                 #("rbf1SVM", SVC(C=1, gamma=.01, cache_size=200, kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)), \
                 #("rbf10SVM", SVC(C=10, gamma=.01, cache_size=200, kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)), \
                 #("polySVM", SVC(C=1, gamma=.01, cache_size=200, coef0=0.0, degree=3, kernel='poly', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)), \
                 #("sigSVM", SVC(C=1, gamma=.01, cache_size=200, coef0=0.0, degree=3, kernel='sigmoid', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)), \
                 #("ada_discrete", AdaBoostClassifier(base_estimator=dt_stump, learning_rate=1, n_estimators=40, algorithm="SAMME")),\
                 #("ada_real", AdaBoostClassifier(base_estimator=dt_stump,learning_rate=1,n_estimators=40,algorithm="SAMME.R")),\
                 #("GBC", GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=50, subsample=1.0, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')),\
                 #("Bagging",BaggingClassifier(base_estimator=dt_stump, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0)),\
                 #("ETC", ExtraTreesClassifier(n_estimators=15, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=False, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False)),\
                 #("RF", RandomForestClassifier(n_estimators=20, criterion='gini',max_depth=3, min_samples_split=2, min_samples_leaf=1, max_features='auto', bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0))\
                 #('DTC', DecisionTreeClassifier()),
                 #("kNeighbors-uniform", KNeighborsClassifier(n_neighbors=5, weights='uniform')),
                 #("kNeighbors-distance", KNeighborsClassifier(n_neighbors=15, weights='distance')),
                 #('GaussianProcessClassifier', GaussianProcessClassifier()),
                 #('LabelPropagation', LabelPropagation()),
                 #('LabelSpreading', LabelSpreading()),
                 ("VotingSoft", VotingClassifier(estimators=[
                     ("LR", LogisticRegression()),
                     ("LR_CV", LogisticRegressionCV()),
                     #('RidgeClassifier',RidgeClassifier()),
                     #('RidgeClassifierCV', RidgeClassifierCV()),
                     #("LSVC", LinearSVC()),
                     ('BernoulliNB', BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None)),
                     ("GNBayes",GaussianNB()),
                     ("rbf1SVM", SVC(C=1, gamma=.01, cache_size=200, kernel='rbf', max_iter=-1, probability=True, random_state=None, shrinking=True, tol=0.001, verbose=False)),
                     ("rbf10SVM", SVC(C=10, gamma=.01, cache_size=200, kernel='rbf', max_iter=-1, probability=True, random_state=None, shrinking=True, tol=0.001, verbose=False)),
                     ("polySVM", SVC(C=1, gamma=.01, cache_size=200, coef0=0.0, degree=3, kernel='poly', max_iter=-1, probability=True, random_state=None, shrinking=True, tol=0.001, verbose=False)),
                     ("sigSVM", SVC(C=1, gamma=.01, cache_size=200, coef0=0.0, degree=3, kernel='sigmoid', max_iter=-1, probability=True, random_state=None, shrinking=True, tol=0.001, verbose=False)),
                     ("ada_discrete", AdaBoostClassifier(base_estimator=dt_stump, learning_rate=1, n_estimators=40, algorithm="SAMME")),
                     ("ada_real", AdaBoostClassifier(base_estimator=dt_stump,learning_rate=1,n_estimators=40,algorithm="SAMME.R")),
                     ("GBC", GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=50, subsample=1.0, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')),
                     ("Bagging",BaggingClassifier(base_estimator=dt_stump, n_estimators=30, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0)),\
                     ("ETC", ExtraTreesClassifier(n_estimators=15, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=False, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False)),
                     ("RF", RandomForestClassifier(n_estimators=20, criterion='gini',max_depth=3, min_samples_split=2, min_samples_leaf=1, max_features='auto', bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0)),
                     ('DTC', DecisionTreeClassifier()),
                     ("kNeighbors-uniform", KNeighborsClassifier(n_neighbors=5, weights='uniform')),
                     ("kNeighbors-distance", KNeighborsClassifier(n_neighbors=15, weights='distance')),
                     ('GaussianProcessClassifier', GaussianProcessClassifier()),
                     ('LabelPropagation', LabelPropagation()),
                     ('LabelSpreading', LabelSpreading()),
                         ], voting='soft', weights=None)),
                 ]
        
        
    

        
    def get_strategy_name(self, string):
        split_name=string.split('_')
        strategy_name=split_name[1]+'_'+split_name[2]  
        return strategy_name
            
    def create_indicators(self):
        '''adds indicators to the data dict'''
        if len(self.signals_dict)<1:
            print('get_signals() first')
            return
        
        #move this to transform
        print('Adding indicators for training set...')
        self.data_dict=self.futures.data_dict.copy()
        self.atr_lookback=10
        self.max_nans=0
        self.nan_index=[]
        for i, sym in enumerate(self.data_dict):
            #get group
            group=futures.dic.loc[sym].Group
            syms_in_group=futures.dic.index[futures.dic.Group==group]
            syms_in_group=[x for x in syms_in_group if x != sym]
            train_df=pd.DataFrame()

            data=self.data_dict[sym][['Open', 'High', 'Low', 'Close',
                               'Volume']].copy()
            if self.max_nans==0:
                lookback=data.shape[0]
            else:
                lookback=self.max_training_length+self.max_nans
            
            #truncate for speed
            data=data[-lookback:]
            
            #price
            train_df['pct_open_1'] = data.Open.pct_change()
            train_df['pct_high_1'] = data.High.pct_change()
            train_df['pct_low_1'] = data.Low.pct_change()
            train_df['pct_close_1'] = data.Close.pct_change()
            train_df['pct_close_2'] = train_df['pct_close_1'].shift(1)
            train_df['pct_close_3'] = train_df['pct_close_1'].shift(2)
            train_df['pct_close_4'] = train_df['pct_close_1'].shift(3)
            train_df['pct_close_5'] = train_df['pct_close_1'].shift(4)
            train_df['pct_close_6'] = train_df['pct_close_1'].shift(5)
            train_df['pct_volume'] = data.Volume.pct_change()
            #train_df['pct_seasonality'] = data.S.pct_change()
            train_df['pct_open_2'] = data.Open.pct_change(periods=2)
            train_df['pct_high_2'] = data.High.pct_change(periods=2)
            train_df['pct_low_2'] = data.Low.pct_change(periods=2)
            train_df['pct_close_2'] = data.Close.pct_change(periods=2)
            train_df['pct_volume_2'] = data.Volume.pct_change(periods=2)
            #train_df['pct_seasonality_2'] = data.S.pct_change(periods=2)
            train_df['pct_open_3'] = data.Open.pct_change(periods=3)
            train_df['pct_high_3'] = data.High.pct_change(periods=3)
            train_df['pct_low_3'] = data.Low.pct_change(periods=3)
            train_df['pct_close_3'] = data.Close.pct_change(periods=3)
            train_df['pct_volume_3'] = data.Volume.pct_change(periods=3)
            #train_df['pct_seasonality_2'] = data.S.pct_change(periods=3)
            
            #% above support/resistance need some lookback here
            
            #% above/below moving averages need lookbacks

            #group sym close/other sym close pct change
            for sym2 in syms_in_group:
                columns=['Open', 'High', 'Low', 'Close', 'Volume']
                data2=self.data_dict[sym2][columns]
                data2=(data/data2).pct_change()
                #print(data2)
                data2.columns=[sym2+'_'+x+'_ratio_1' for x in columns]
                train_df=pd.concat([train_df, data2], axis=1)
                data2.columns=[sym2+'_'+x+'_ratio_2' for x in columns]
                train_df=pd.concat([train_df, data2.shift(1)], axis=1)
                data2.columns=[sym2+'_'+x+'_ratio_3' for x in columns]
                train_df=pd.concat([train_df, data2.shift(2)], axis=1)
                
            #atr % change
            data['ATR']=atr_df(data, self.atr_lookback)
            train_df['pct_atr_1']=data.ATR.pct_change()
            train_df['pct_atr_2']=train_df['pct_atr_1'].shift(1)
            train_df['pct_atr_3']=train_df['pct_atr_1'].shift(2)
            train_df['pct_atr_4']=train_df['pct_atr_1'].shift(3)
            train_df['pct_atr_5']=train_df['pct_atr_1'].shift(4)
            train_df['pct_atr_6']=train_df['pct_atr_1'].shift(5)
            
            
            
            if i==0:
                nan_counts=train_df.isnull().sum()
                self.max_nans=max(nan_counts)
                
            #check data for nans
            train_df=train_df[-self.max_training_length:]

            if train_df.isnull().sum().sum() > 0:
                message='{} nans in {}'.format(nan_counts, sym)
                print(message)
                sys.exit(message)
                
                
            '''
            max_nan_col=nan_counts[nan_counts==self.max_nans].index[0]
            nan_index=train_df[train_df[max_nan_col].isnull()].index.tolist()
            if i == 0:
                self.nan_index=nan_index
            else:
                if nan_index != self.nan_index:
                    message='inconsistent numbers of nans in {}: {}'.format(sym,
                                                             str(nan_index))
                    print(message)
                    sys.exit(message)
                #print(sym,'data checked')
            
            '''
            for length in self.training_lengths:
                self.training_dict[length][sym]=train_df[-length:]
            
        self.train_df=train_df
        
    def get_signals(self):
        self.signals_dict=self.futures.signals.training_signals[self.last_date]
        self.training_lengths=list(self.signals_dict.keys())
        self.max_training_length=max(self.training_lengths)
        for length in self.training_lengths:
            self.training_dict[length]={}
        self.symbols=list(self.signals_dict[self.max_training_length].keys())
        
        for length in self.training_lengths:
            strategies=list(self.signals_dict[length][self.symbols[0]].keys())
            self.strategies[length]=strategies
            
            for strategy in strategies:
                name=self.get_strategy_name(strategy)
                if name not in self.strategy_types:
                    self.strategy_types.append(name)
        
                
    def prepare(self):
        self.get_signals()
        self.create_indicators()
        print('Training data checked and prepared')
        
    def train(self, symbol):
        if len(self.training_dict)<1:
            print('prepare() first')
            return
        
        self.predictions={}
        compression = self.FEATURE_COMPRESSION
        
        print('Training models..')
        #train/predict
        #for sym
        for classifier in self.models:
            #print(classifier)
            self.pipelines = [

                Pipeline(
                [('OVO_'+classifier[0], OneVsOneClassifier(classifier[1],n_jobs=-1)),
                 ]),
                Pipeline(
                [('OVO_RFE_'+classifier[0], OneVsOneClassifier(classifier[1],n_jobs=-1)),
                 ]),
                Pipeline(
                [
                 ('SelectKBest'+str(compression),SelectKBest(f_classif, k=compression)),  
                 ('OVO_'+classifier[0], OneVsOneClassifier(classifier[1],n_jobs=-1)),                      
                    ]),
                Pipeline(
                [
                 ('PCA'+str(compression),PCA(n_components=compression)),
                 ('OVO_'+classifier[0], OneVsOneClassifier(classifier[1],n_jobs=-1)),
                    ]), 
                Pipeline(
                [('OVR_'+classifier[0], OneVsRestClassifier(classifier[1],n_jobs=-1)),
                 ]),
                Pipeline(
                [('OVR_RFE'+classifier[0], OneVsRestClassifier(classifier[1],n_jobs=-1)),
                 ]),
                Pipeline(
                [
                 ('SelectKBest'+str(compression),SelectKBest(f_classif, k=compression)),
                 ('OVR_'+classifier[0], OneVsRestClassifier(classifier[1],n_jobs=-1)),
                    ]),                
                Pipeline(
                [
                 ('PCA'+str(compression),PCA(n_components=compression)),
                 ('OVR_'+classifier[0], OneVsRestClassifier(classifier[1],n_jobs=-1)),
                    ]),                
                    ]
            pipe_counts={
                    'OVO':{},
                    'RFE_OVO':{},
                    'KB_OVO':{},
                    'PCA_OVO':{},
                    'OVR':{},
                    'RFE_OVR':{},
                    'KB_OVR':{},
                    'PCA_OVR':{},
                    
                    }
            for pipe_name, pipeline in zip(pipe_counts.keys(), self.pipelines): 
                print(pipe_name)
                print(pipeline.get_params)
                self.pipeline=pipeline
                start_time = time.time()
                
                for length in self.training_lengths:
                    #pipe_counts[pipe_name][str(length)]={}
                    
                    for strategy in self.strategies[length]:
                        x_train=self.training_dict[length][symbol][:-1].values
                        y_train=self.signals_dict[length][symbol][:-1][strategy].values
                        x_test=self.training_dict[length][symbol][-1:].values
                        
                        if pipe_name.split('_')[0]=='RFE':
                            rfecv=RFECV(self.rfe_estimator[0][1], n_jobs=-1)
                            x_train=rfecv.fit_transform(x_train, y_train)
                            x_test = rfecv.transform(x_test)
                            
                        pipeline.fit(x_train, y_train)
                        signal = pipeline.predict(x_test)
                        print(self.last_date, symbol, length, strategy, signal)
                        
                        name=self.get_strategy_name(strategy)
                        strat_name=str(length)+'_'+name
                        if strat_name in pipe_counts[pipe_name]:
                            pipe_counts[pipe_name][strat_name]+=int(signal[0])
                        else:
                            pipe_counts[pipe_name][strat_name]=int(signal[0])

                        #strat_id = '{}_{}_{}'.format(symbol, length, strategy)
                        
                
                print('Elapsed time: {} minutes. {}'\
                      .format(round(((time.time() - start_time) / 60), 2), dt.now()))
            
            pipe_counts=pd.DataFrame(pipe_counts)
            self.pipe_cols=pipe_counts.columns.tolist()
            
            pipe_counts['ALL']=to_signals(pipe_counts.sum(axis=1))
            
            stratname_dict={}
            for typ in self.strategy_types:
                stratname_dict[typ]=[index for index in self.pipe_counts.index\
                                                          if typ in index]
                
            pipe_counts['SignalDate']=self.last_date
            self.pipe_counts=pipe_counts
            self.pipe_cols.append('ALL')
                
        
    def train_all(self, nextday_futures=None):
        if len(self.training_dict)<1:
            print('prepare() first')
            return
        
        
        '''
        def my_function(q, x):
            q.put(x + 100)
        
        if __name__ == '__main__':
            queue = Queue()
            p = Process(target=my_function, args=(queue, 1))
            p.start()
            p.join() # this blocks until the process terminates
            result = queue.get()
            print result
            
            
        def runInParallel(*fns):
          proc = []
          for fn in fns:
            p = multiprocessing.Process(target=fn)
            p.start()
            proc.append(p)
          for p in proc:
            p.join()
        '''
        #trains all markets subprocess
        for symbol in self.markets:
            self.train(symbol)
            
            if nextday_futures is not None:
                self.check_signal(symbol, nextday_futures)
            
    def check_signal(self, symbol, nextday_futures):
        self.next_date=nextday_futures.last_date
        self.nextday_signals_dict=nextday_futures.signals.training_signals[self.next_date]
        for length in self.training_lengths:
            for strategy in self.nextday_signals_dict[length][symbol].columns:
                signal=self.nextday_signals_dict[length][symbol][strategy][-2]
                signal_date=self.nextday_signals_dict[length][symbol][strategy].index[-2]
                name=self.get_strategy_name(strategy)
                index_name='{}_{}'.format(length, name)
                self.pipe_counts.set_value(index_name, 'ACTDate', signal)
                self.pipe_counts.set_value(index_name, 'ACT', signal_date)
        #add diff
        for col in self.pipe_cols:
            self.pipe_counts['vs_ACT_'+col]=self.pipe_counts['ACT']-self.pipe_counts[col]
            self.pipe_counts['vs_GA_'+col]=self.pipe_counts['ACT']-self.pipe_counts[col]
            
if __name__ == '__main__':
    from gsm import Service
    from nimbus.csidata import Futures
    '''
    if 'myVar' in locals():
      # myVar exists.
    To check the existence of a global variable:
    
    if 'myVar' in globals():
      # myVar exists.
    To check if an object has an attribute:
    
    if hasattr(obj, 'attr_name'):
      # obj.attr_name exists.
    '''
    if 'start_time' not in globals():
        start_time = time.time()
        #
        #gsm.create(4)
        #print(gsm.portfolio.history)
        days=2
        
        futures = Futures()
        futures_generator=futures.create_simulation_data(days)
        futures = next(futures_generator)
        
        nextday_futures = Futures()
        nextday_futures_generator=nextday_futures.create_simulation_data(days-1)
        nextday_futures = next(nextday_futures_generator)
    else:
        start_time = time.time()
    

    #
    self=ML(futures)
    self.prepare() #prepares training data
    self.train('US') #trains models
    self.check_signal('US', nextday_futures)

    #print(pd.DataFrame(self.pipe_counts))
    #self.predict() #appends signals to future.atr
    
    #self.train(sym)
    #for sym in gsm.futures.dic.index:
    #    self.train(sym)
        #trend, mode = self.zigzag_data.get_peaks_valleys(show_plots=True)
        #self.zigzag_data.plot_pivots()
    print('Elapsed time: {} minutes. {}'\
          .format(round(((time.time() - start_time) / 60), 2), dt.now()))


'''
from sklearn.model_selection import GridSearchCV

param_grid = [
        {
            'activation' : ['identity', 'logistic', 'tanh', 'relu'],
            'solver' : ['lbfgs', 'sgd', 'adam'],
            'hidden_layer_sizes': [
             (1,),(2,),(3,),(4,),(5,),(6,),(7,),(8,),(9,),(10,),(11,), (12,),(13,),(14,),(15,),(16,),(17,),(18,),(19,),(20,),(21,)
             ]
        }
       ]
'''
