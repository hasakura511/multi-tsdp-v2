#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 11:29:40 2018

@author: hidemiasakura

machine learning signal processor
inputs futures
output signals
"""

import time
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
import calendar
from datetime import datetime as dt

#from nimbus.process.mean_reversion import MeanReversion
from nimbus.process.zigzag import zigzag
from nimbus.process.transform import to_signals
#sklearn grid_search, cross_validation -->model search
#from sklearn.grid_search import ParameterGrid
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2, f_classif
#classification
#from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import Perceptron, PassiveAggressiveClassifier, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier,\
                        BaggingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC, SVC, NuSVC
from sklearn.neighbors import RadiusNeighborsClassifier, KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB


#import warnings
#warnings.simplefilter('error')

class Signals(object):
    
    def __init__(self):
        #self.SYMBOL = ''
        #self.SUPPORT_RESISTANCE_LOOKBACK = 60
        #self.VALIDATION_START_DATE = None
        #self.VALIDATION_LENGTH = None
        #self.SIGNAL_TYPES = ['gain_ahead', 'zigzag', 'long', 'short']
        #self.ADF_PVALUE = 3.0
        #self.FEATURE_COMPRESSION = 10
        #self.MIN_DATAPOINTS = 3
        #self.FILTER_METRIC = 'net_equity'
        #self.MAX_LOOKBACK = self.SUPPORT_RESISTANCE_LOOKBACK*2
        #self.MAX_READ_LINES = 500
        #self.INNER_ZIGZAG_STD = 2.0
        #self.OUTER_ZIGZAG_STD = 4.0
        
        #ML PARAMS
        self.dt_stump = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
        self.rfe_estimator = [ 
                ("None","None"),\
                #("GradientBoostingRegressor",GradientBoostingRegressor()),\
                #("DecisionTreeRegressor",DecisionTreeRegressor()),\
                #("ExtraTreeRegressor",ExtraTreeRegressor()),\
                #("BayesianRidge", BayesianRidge()),\
                 ]
        
        self.feature_selection_models = [
                 ('PCA'+str(self.FEATURE_COMPRESSION),PCA(n_components=self.FEATURE_COMPRESSION)),\
                 ('SelectKBest'+str(self.FEATURE_COMPRESSION),SelectKBest(f_classif, k=self.FEATURE_COMPRESSION))\
                 ]
        
        self.short_models = [
                 #("LR", LogisticRegression(class_weight={1:1})), \
                 #("PRCEPT", Perceptron(class_weight={1:1})), \
                 #("PAC", PassiveAggressiveClassifier(class_weight={1:1})), \
                 #("LSVC", LinearSVC()), \
                 ("GNBayes",GaussianNB()),\
                 #("LDA", LinearDiscriminantAnalysis()), \
                 #("QDA", QuadraticDiscriminantAnalysis()), \
                 #("MLPC", Classifier([Layer("Sigmoid", units=150), Layer("Softmax")],learning_rate=0.001, n_iter=25, verbose=True)),
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
                     #("kNeighbors-uniform", KNeighborsClassifier(n_neighbors=2, weights='uniform')),\
                     #("MLPC", Classifier([Layer("Sigmoid", units=150), Layer("Softmax")],learning_rate=0.001, n_iter=25, verbose=True)),\
                     #("rbfSVM", SVC(C=1, gamma=.01, cache_size=200, class_weight={1:500}, kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)), \
                     #("kNeighbors-distance", KNeighborsClassifier(n_neighbors=8, weights='distance')),\
                     #("Bagging",BaggingClassifier(base_estimator=dt_stump, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0)),\
                     #("ETC", ExtraTreesClassifier(class_weight={1:1}, n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=False, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False)),\
                     #], voting='hard', weights=None)),
                 #("VotingSoft", VotingClassifier(estimators=[\
                     #("ada_discrete", AdaBoostClassifier(base_estimator=dt_stump, learning_rate=1, n_estimators=400, algorithm="SAMME")),\
                     #("ada_real", AdaBoostClassifier(base_estimator=dt_stump,learning_rate=1,n_estimators=180,algorithm="SAMME.R")),\
                     #("GBC", GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')),\
                     #("QDA", QuadraticDiscriminantAnalysis()),\
                     #("GNBayes",GaussianNB()),\
                     #("MLPC", Classifier([Layer("Sigmoid", units=150), Layer("Softmax")],learning_rate=0.001, n_iter=25, verbose=True)),\
                     #("rbfSVM", SVC(C=1, gamma=.01, cache_size=200, class_weight={1:1}, kernel='rbf', max_iter=-1, probability=True, random_state=None, shrinking=True, tol=0.001, verbose=False)), \
                     #("kNeighbors-distance", KNeighborsClassifier(n_neighbors=8, weights='distance')),\
                     #("Bagging",BaggingClassifier(base_estimator=dt_stump, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0)),\
                     #("ETC", ExtraTreesClassifier(class_weight={1:1}, n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=False, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False)),\
                 #        ], voting='soft', weights=None)),
                 ]
        
                 
        self.models = [
                 #("LR", LogisticRegression(class_weight={1:1})), \
                 #("PRCEPT", Perceptron(class_weight={1:1})), \
                 #("PAC", PassiveAggressiveClassifier(class_weight={1:1})), \
                 #("LSVC", LinearSVC()), \
                 #("GNBayes",GaussianNB()),\
                 #("LDA", LinearDiscriminantAnalysis()), \
                 #("QDA", QuadraticDiscriminantAnalysis()), \
                 #("MLPC", Classifier([Layer("Sigmoid", units=150), Layer("Softmax")],learning_rate=0.001, n_iter=25, verbose=True)),
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
                 ("VotingSoft", VotingClassifier(estimators=[\
                     #("ada_discrete", AdaBoostClassifier(base_estimator=dt_stump, learning_rate=1, n_estimators=400, algorithm="SAMME")),\
                     #("ada_real", AdaBoostClassifier(base_estimator=dt_stump,learning_rate=1,n_estimators=180,algorithm="SAMME.R")),\
                     #("GBC", GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto')),\
                     #("QDA", QuadraticDiscriminantAnalysis()),\
                     ("GNBayes",GaussianNB()),\
                     ("LDA", LinearDiscriminantAnalysis()), \
                     ("kNeighbors-uniform", KNeighborsClassifier(n_neighbors=self.MIN_DATAPOINTS, weights='uniform')),\
                     #("MLPC", Classifier([Layer("Sigmoid", units=150), Layer("Softmax")],learning_rate=0.001, n_iter=25, verbose=True)),\
                     #("rbfSVM", SVC(C=1, gamma=.01, cache_size=200, class_weight={1:1}, kernel='rbf', max_iter=-1, probability=True, random_state=None, shrinking=True, tol=0.001, verbose=False)), \
                     #("kNeighbors-distance", KNeighborsClassifier(n_neighbors=8, weights='distance')),\
                     #("Bagging",BaggingClassifier(base_estimator=dt_stump, n_estimators=10, max_samples=1.0, max_features=1.0, bootstrap=True, bootstrap_features=False, oob_score=False, warm_start=False, n_jobs=1, random_state=None, verbose=0)),\
                     #("ETC", ExtraTreesClassifier(class_weight={1:1}, n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=False, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False)),\
                         ], voting='soft', weights=None)),
                 ]
        #self.SHORT_TREND_SIGNAL_TYPES = self.SIGNAL_TYPES
        self.short_model=self.short_models[0]
        self.short_trend_pipelines=[
                       [self.short_model],
                       [self.feature_selection_models[0],self.short_model],
                       #[self.FEATURE_SELECTION_MODELS[1],SHORTMODEL],
                        ]
                        
        #self.PV2E_SIGNAL_TYPES =self.SIGNAL_TYPES
        self.pv2e_model=self.models[0]
        self.pv2e_pipelines=[
                        [self.pv2e_model],
                        [self.feature_selection_models[0],self.pv2e_model],
                        #[fs_models[1],pv2e_p2pModel],
                        ]
    
        #self.PV3S_SIGNAL_TYPES = self.SIGNAL_TYPES
        self.pv3s_model= self.models[0]               
        self.pv3s_pipelines=[
                        [self.pv3s_model],
                        [self.feature_selection_models[0],self.pv3s_model],
                        #[fs_models[1],pv3s_v2vModel],
                        ]
        
        self.signal_sets={
                'wf_is_short':{},
                'wf_is_pv2e_p2p':{},
                'wf_is_pv3s_v2v':{},
                }
        self.rank_by_metric_best={
                        'best_wf_is_short':{},
                        'best_wf_is_pv2e_p2p':{},
                        'best_wf_is_pv3s_v2v':{},
                        'best_wf_is_all':{},
                        }
        self.rank_by_metric_worst={
                        'worst_wf_is_short':{},
                        'worst_wf_is_pv2e_p2p':{},
                        'worst_wf_is_pv3s_v2v':{},
                        'worst_wf_is_all':{},
                        }
        #self.final_df={}
        self.signal_df={}
        #self.futures_dict={}
    '''
    def check_data(self):
        print('Checking futures data..')
        for sym in self.futures.data_dict.keys():    
            data=self.futures.data_dict[sym].copy()
            
            #check if there's enough data issue warning
            if data.shape[0] < self.MAX_LOOKBACK:
                message =  'Warning! Data length {} < MAX_LOOKBACK {}'.format(
                                             data.shape[0], self.MAX_LOOKBACK)
                print(message)
                
            #if the there's more data than we need, truncate it
            if data.shape[0] > self.MAX_LOOKBACK:
                data = data[-self.MAX_LOOKBACK:]
            
            data['symbol']=sym
            
            data['gain_ahead']=data.Close.pct_change().shift(-1).fillna(0)
            #find where gain ahead was 0, other than the last index
            zero_gains_index=data.reset_index()\
                    [(data.gain_ahead==0).values].index.tolist()[:-1]
            #if gain ahead was 0, then set it to the next value
            for i in zero_gains_index:
                data.set_value(data.iloc[i].name, 'gain_ahead',
                               data.iloc[i+1].gain_ahead)
                #print(data.iloc[i])
                #print(data.iloc[i+1])
                
            data['gain_ahead_signal']=to_signals(data.gain_ahead)
            self.futures_dict[sym]=data
            
        self.debug=data
    '''
    
    def get_signals(self):
        
        for sym in self.data_dict:
            data=self.data_dict[sym]
            signal_cols=[x for x in data.columns if 'signals' in x]
            
    def add_indicators(self):
        '''adds indicators to the data dict'''
        for sym in self.data_dict:
            data=self.data_dict[sym]
            
    
    def create(self, futures):
        self.data_dict=futures.data_dict.copy()
        self.get_signals()
        self.add_indicators()
        
    def train(self, symbol):
        if len(self.futures_dict)<1:
            print('create() first')
            return
        
        print('Training models..')
        SR_LOOKBACK=self.SUPPORT_RESISTANCE_LOOKBACK
        
        
        self.dataset=self.futures_dict[sym][-SR_LOOKBACK:]
        NROWS=self.dataset.shape[0]
        print(NROWS,'datapoints')
        #mr=MeanReversion()
        #mr_modes=mr.adf(dataset.Close, SR_LOOKBACK, threshold=self.ADF_PVALUE,
        #                showPlot=True, ticker=sym)
        self.zigzag_data=zigzag(symbol, self.dataset,self.OUTER_ZIGZAG_STD)
        '''
        training_range=range(SR_LOOKBACK,NROWS-SR_LOOKBACK+1)
        for start,i in enumerate(training_range):
            print(start, i)
    
            #modes = mrClassifier(data.Close, data.shape[0]-1,
            #                     threshold=adfPvalue, showPlot=debug,
            #                               ticker=ticker+contractExpiry)

        '''
        
    def train_all(self, futures):
        #trains all markets sequentially
        pass

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
        futures = Futures()
        futures_generator=futures.create_simulation_data(3)
    else:
        start_time = time.time()
    

    futures = next(futures_generator)
    self=Signals()
    self.create(futures)
    sym='ES'
    self.train(sym)
    #for sym in gsm.futures.dic.index:
    #    self.train(sym)
        #trend, mode = self.zigzag_data.get_peaks_valleys(show_plots=True)
        #self.zigzag_data.plot_pivots()

print('Elapsed time: {} minutes. {}'\
      .format(round(((time.time() - start_time) / 60), 2), dt.now()))

