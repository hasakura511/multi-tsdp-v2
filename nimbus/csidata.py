#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 16:31:36 2018

@author: hidemiasakura

inputs
dictionary file

outputs
atr file
consolidated data files

"""
import sys
import numpy as np
import pandas as pd
#from nimbus.portfolio import Portfolio
from nimbus.signals import Signals
from nimbus.process.transform import average_true_range
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
#import calendar
#from datetime import datetime as dt
import seaborn as sns
#import warnings
#warnings.simplefilter('error')

#dic=pd.read_csv('./nimbus/data/system/dictionary.csv', index_col=0)
#dicfx=pd.read_csv('./nimbus/data/system/dictionary_fx.csv', index_col=0)


def get_hist(filename, maxlookback):
    data = pd.read_csv(filename, index_col=0)[-maxlookback:]
    print(data.shape[0], filename, data)
    return data
    
def get_history(filename, datapoints, total_datapoints):
    data = pd.read_csv(filename, index_col=0, header=None)[-total_datapoints:]
    print('datapoints',datapoints,'total_datapoints',\
          total_datapoints, filename, 'check', data.shape[0]==total_datapoints)
    #print(data.shape[0], filename, data.shape[0]==maxlookback, '\n', data)

    for i, j in enumerate(range(datapoints, data.shape[0]+1)):
        yield data.iloc[i:j]

class FX(object):
    
    def __init__(self):
        self.dic = pd.read_csv('./nimbus/data/system/dictionary_fx.csv', index_col=0)
        self.atr=pd.DataFrame()
        self.data=pd.DataFrame()
        self.correlations=pd.DataFrame()
        self.atr_lookback = 20
        self.datapoints = self.atr_lookback+1
        self.total_datapoints = self.datapoints
        self.data_path='./nimbus/data/csidata/fx/'
        self.debug_path='./nimbus/debug/'
        self.cache_path='./nimbus/process/ta/'
        self.last_date=None
        self.first_date=None
        self.rates={}
        self.rates_history={}
        self.history=pd.DataFrame()
        self.data_generators={}
        self.corr_history={}
        self.days=1

    def create_history(self, **kwargs):
        self.atr_lookback=kwargs.get('atr_lookback', self.atr_lookback)
        iteration=kwargs.get('iteration', 0)
        
        #how much data do you need to prime the first datapoint
        #atrlookback + pc lookback(1)
        self.datapoints=self.atr_lookback+1
        #total primed datapoints, -1 because iteration starts at 0
        self.total_datapoints=self.datapoints+(self.days-1)
        
        cDictCSI=self.dic.to_dict()['filename']
        currencyPairs=cDictCSI.keys()
        
        for i, pair in enumerate(currencyPairs):
            #end at -1 to ignore  new day. 
            filename = self.data_path+cDictCSI[pair]
            if iteration==0:
                self.data_generators[pair] = get_history(filename, self.datapoints,\
                                                        self.total_datapoints)
            #print pair, data.index[-1]
            data=next(self.data_generators[pair])
            #print(data)
            data.index = pd.to_datetime(data.index,format='%Y%m%d')
            data.columns = ['Open','High','Low','Close','S']
            data.index.name = 'Dates'
            first_date = data.index[0]
            last_date = data.index[-1]
            
            if i==0:
                self.last_date=last_date
                self.first_date=first_date
            else:
                if last_date> self.last_date:
                    self.last_date=last_date
                if first_date< self.first_date:
                    self.first_date=first_date
            
            #need 21 datapoints for atr 20 lookback
            atr=average_true_range(data.High.values,data.Low.values,\
                                             data.Close.values,self.atr_lookback)
            #print(len(atr),atr)
            pc=data.Close.pct_change()[-self.atr_lookback:]
            priorSig=np.where(pc<0,-1,1)[-1]
            self.data[pair]=pc
            
            self.atr.set_value(pair,'Open',data.Open.iloc[-1])
            self.atr.set_value(pair,'High',data.High.iloc[-1])
            self.atr.set_value(pair,'Low',data.Low.iloc[-1])
            self.atr.set_value(pair,'Close',data.Close.iloc[-1])
            self.atr.set_value(pair,'S',data.S.iloc[-1])
            #self.atr.set_value(pair,'Last',data.Close.iloc[-1])
            self.atr.set_value(pair,'PC',pc.iloc[-1])
            self.atr.set_value(pair,'ACT',priorSig)
            self.atr.set_value(pair,'ATR'+str(self.atr_lookback),atr[-1])
            self.atr.set_value(pair,'first_date',\
                                        first_date)
            self.atr.set_value(pair,'last_date',\
                                        last_date)
        
        self.atr.index.name='pair'
        self.rates={
            'AUD':1/self.atr.loc['AUDUSD'].Close,
            'CAD':self.atr.loc['USDCAD'].Close,
            'CHF':self.atr.loc['USDCHF'].Close,
            'EUR':1/self.atr.loc['EURUSD'].Close,
            'GBP':1/self.atr.loc['GBPUSD'].Close,
            'HKD':self.atr.loc['USDHKD'].Close,
            'JPY':self.atr.loc['USDJPY'].Close,
            'NZD':1/self.atr.loc['NZDUSD'].Close,
            'SGD':self.atr.loc['USDSGD'].Close,
            'USD':1,
            }
        self.correlations=self.data.corr()
        print('FX data created for', self.last_date)
    '''    
    def correlate(self):
        if self.data.shape[0]<1:
            print('create() first!')
            return
        
        self.correlations=self.data.corr()
        print('correlations created!')
    '''
    def save_correlations(self, show=False):
        if self.correlations.shape[0]<1:
            print('create() first!')
            return
        
        savePath= self.debug_path
        
        fig,ax = plt.subplots(figsize=(13,13))
        ax.set_title('FX Correlation Heatmap '+str(self.first_date)+\
                     ' to '+str(self.last_date))
        sns.heatmap(ax=ax,data=self.correlations)
        plt.yticks(rotation=0) 
        plt.xticks(rotation=90)         
        
        filename = '{}{}_fx_correlations.png'.format(savePath,\
                        self.last_date)
        print('Saving {}'.format(filename))
        fig.savefig(filename, bbox_inches='tight')
            
        if show:
            plt.show()
        plt.close()
        
    def create(self, days=1):
        self.days=days
        self.rates_history={}
        self.history=pd.DataFrame()
        self.corr_history={}
        print('\n\nCreating {} day(s) of FX history'.format(days))
        for i in range(days):
            print('\nDay',i+1)
            self.create_history(iteration=i)
            self.history=self.history.append(self.atr)
            self.rates_history[self.last_date.strftime('%Y%m%d')]=self.rates
            self.corr_history[self.last_date.strftime('%Y%m%d')]=self.correlations
            

class Futures(object):
    
    def __init__(self):
        self.dic = pd.read_csv('./nimbus/data/system/dictionary.csv', index_col=0)\
                        .reset_index().set_index('CSIsym')
        self.fx=FX()
        self.atr=pd.DataFrame()
        self.pct_changes=pd.DataFrame()
        self.correlations=pd.DataFrame()
        #self.sr_lookback = 60
        self.atr_lookback = 20
        self.seasonality_lookback=270
        #self.zigzag_std=4.0
        self.datapoints = max(self.atr_lookback+1, self.seasonality_lookback)
        self.total_datapoints = self.datapoints
        self.data_path='./nimbus/data/csidata/futures/'
        self.debug_path='./nimbus/debug/'
        self.last_date=None
        self.first_date=None
        self.months = {
                1:'F',
                2:'G',
                3:'H',
                4:'J',
                5:'K',
                6:'M',
                7:'N',
                8:'Q',
                9:'U',
                10:'V',
                11:'X',
                12:'Z'
                }
        self.files = [ f for f in listdir(self.data_path)\
                      if isfile(join(self.data_path,f)) ]
        #self.markets = [x.split('_')[0] for x in self.files]
        self.markets = self.dic.reset_index().set_index('CSIsym2').index.tolist()
        self.history=pd.DataFrame()
        self.data_generators={}
        self.data_dict={}
        self.corr_history={}
        self.days=1
        #self.training_signals={}
   
    def check_dates(self, sym, data, symbol_iteration):
        first_date = data.index[0]
        last_date = data.index[-1]
        
        #date check
        if symbol_iteration==0:
            self.last_date=last_date
            self.first_date=first_date
        else:
            #add ffill/bfill code here if necessary
            if last_date> self.last_date:
                
                message = '{} Last date ahead that of previous sym!\
                            data misaligned? potential future leak'\
                            .format(sym)
                print(message)
                sys.exit(message)
                #self.last_date=last_date
            if last_date< self.last_date:
                message = '{} Last date behind that of previous sym!\
                            data misaligned? Check data.'\
                            .format(sym)
                print(message)
                sys.exit(message)
            
            if first_date< self.first_date:
                self.first_date=first_date
    '''
    def add_ml_signals(self, sym, data):
        #data['symbol']=sym
        
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
        data['zigzag_signal']=self.zigzag.signals()
        self.data_dict[sym]=data
    
        
    def zigzag_mode_signals(self, data, lookback=120, train=120):
        NROWS=data.shape[0]
        START=NROWS-train
        self.zigzag=zigzag(sym, data, self.zigzag_std)
        signal_ZZTREND_120, signal_ZZMODE_120 = self.zigzag.get_peaks_valleys()
        pass
    '''
    
    def create_history(self, **kwargs):
        self.atr_lookback=kwargs.get('atr_lookback', self.atr_lookback)
        self.seasonality_lookback=kwargs.get('atr_lookback', self.seasonality_lookback)
        iteration=kwargs.get('iteration', 0)
        
        #how much data do you need to prime the first datapoint
        #atrlookback + pc lookback(1)
        #self.datapoints=self.atr_lookback+1
        self.datapoints = max(self.atr_lookback+1, self.seasonality_lookback)
        #total primed datapoints, -1 because iteration starts at 0
        self.total_datapoints=self.datapoints+(self.days-1)
        
        #print('Creating Futures ATR file...')            
        futuresDF=self.dic.copy()
        self.pct_changes=pd.DataFrame()
        
        for symbol_iteration, contract in enumerate(self.markets):
            if 'YT' not in contract:
                sym = ''.join([i for i in contract.split('_')[0] if not i.isdigit()])
            else:
                sym=contract
                
            filename=self.data_path+contract+'_B.csv'
            if iteration==0:
                #initialize data generator
                self.data_generators[sym] = get_history(filename, self.datapoints,\
                                                        self.total_datapoints)
            #print pair, data.index[-1]
            data=next(self.data_generators[sym])
            #print(data.shape)
            data.index = pd.to_datetime(data.index,format='%Y%m%d')
            data.columns = ['Open','High','Low','Close','Volume','OI','R','S']
            data.index.name = 'Dates'

            #self.data_dict[sym]=data.copy()
            self.debug=data.copy()
            
            #check if all the last_dates are the same
            self.check_dates(sym, data, symbol_iteration)
            self.data_dict[sym]=data

            #truncate date if longer than the max lookback at this point.
            data=data[-self.atr_lookback-1:]
            #print(data)
            atr=average_true_range(data.High.values, data.Low.values,
                                   data.Close.values, self.atr_lookback)
            
            #correlation needs 20 datapoints
            self.pct_changes[sym]=data.Close.pct_change()[1:]
            
            #contract data                
            contractYear=str(data.R[-1])[3]
            contractMonth=str(data.R[-1])[-2:]
            #print(sym+self.months[int(contractMonth)]+contractYear)
            Contract=sym+self.months[int(contractMonth)]+contractYear
            currency=self.dic.loc[sym].currency
            rate=self.fx.rates_history[self.last_date.strftime('%Y%m%d')][currency]
            multiplier=self.dic.loc[sym].CSImultiplier
            multiplier2=multiplier/rate
            usdATR = atr[-1]*multiplier2
            #print(usdATR)
            contract_value = data.Close[-1]*multiplier2
            #print(contract_value)
            
            #contract data
            futuresDF.set_value(sym,'Contract',Contract)
            futuresDF.set_value(sym,'Open',data.Close[-1])
            futuresDF.set_value(sym,'High',data.High[-1])
            futuresDF.set_value(sym,'Low',data.Low[-1])
            futuresDF.set_value(sym,'Close',data.Close[-1])
            futuresDF.set_value(sym,'Volume',data.Volume[-1])
            futuresDF.set_value(sym,'OI',data.OI[-1])
            futuresDF.set_value(sym,'R',data.R[-1])
            futuresDF.set_value(sym,'S',data.S[-1])
            futuresDF.set_value(sym,'ATR'+str(self.atr_lookback),atr[-1])
            futuresDF.set_value(sym,'LastPctChg',self.pct_changes[sym][-1])
            futuresDF.set_value(sym,'usdATR',usdATR)
            futuresDF.set_value(sym,'contract_value',contract_value)
            futuresDF.set_value(sym,'FX',rate)

            #dates
            futuresDF.set_value(sym,'first_date', self.first_date)
            futuresDF.set_value(sym,'last_date', self.last_date)
            #futuresDF.set_value(sym,'validation_start_date',validation_start_date)
            
        
        self.atr=futuresDF
        #self.training_signals[self.last_date]={}
        #add signals
        if iteration==0:
            self.signals=Signals(self)
        self.signals.add_previous_signals()
        self.signals.add_seasonality_signals()
        self.signals.add_ml_signals()

        
            
        
        
        #self.last_date = self.data.index[-1].strftime('%Y%m%d')
        #self.first_date = self.data.index[0].strftime('%Y%m%d')
        if self.atr.shape[0] != self.atr.dropna().shape[0]:
            print('Nans found in {}. atr shape {}, dropna {}\n'.format(self.last_date,\
                          self.atr.shape[0], self.atr.dropna().shape[0]))
        else:
            print('Futures data created for {} {}\n'.format(self.last_date,\
                                                              self.atr.shape))
        #print(pc)
        #print(self.pct_changes)    
        self.correlations=self.pct_changes.corr()
        if np.isnan(self.correlations).sum().sum()>0:
            sys.exit('nans in self.correlations. check data')
    
    def create(self, portfolio, params, days):
        #start_index=self.lookback+days
        self.days=days
        self.history=pd.DataFrame()
        self.corr_history={}
        self.fx.create(days+10)
        print('\n\nCreating {} day(s) of Futures history'.format(days))
        for i in range(days):
            print('\nDay',i+1,)
            self.create_history(iteration=i)
            self.history=self.history.append(self.atr)
            #self.rates_history[self.last_date.strftime('%Y%m%d')]=self.rates
            self.corr_history[self.last_date.strftime('%Y%m%d')]=self.correlations
            
            if i==0:
                #signals=self.atr[system]
                portfolio.create(self, **params)
            else:
                '''
                p2=  dict(
                        account_value=500000,
                        margin_percent=0.5,
                        benchmark_sym='GC',
                        correlation_cutoff=0.7,
                        recreate_if_margin_call=False,
                        increment=250,    
                      )
                portfolio.update(self, self.atr.signals_EXCESS, **p2)
                '''
                #signals=self.atr[system]
                portfolio.update(self)
                
    def create_simulation_data(self, days):
        #start_index=self.lookback+days
        self.days=days
        self.history=pd.DataFrame()
        self.corr_history={}
        self.fx.create(days+10)
        print('\n\nCreating {} day(s) of Futures history'.format(days))
        for i in range(days):
            print('\nDay',i+1,)
            self.create_history(iteration=i)
            self.history=self.history.append(self.atr)
            #self.rates_history[self.last_date.strftime('%Y%m%d')]=self.rates
            self.corr_history[self.last_date.strftime('%Y%m%d')]=self.correlations
            yield self
            
                
    def save_correlations(self, show=False):
        if self.correlations.shape[0]<1:
            print('create() first!')
            return
        
    def get_strategies(self):
        if self.atr.shape[0]<1:
            print('create() first!')
            return
        strategies = [x.replace('signals_','') for x in self.atr.columns\
                                                          if 'signals_' in x]
        return strategies


