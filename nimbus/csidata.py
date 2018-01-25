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

import numpy as np
import pandas as pd
#from nimbus.portfolio import Portfolio
from nimbus.process.transform import ATR2
import matplotlib.pyplot as plt
import seaborn as sns
from os import listdir
from os.path import isfile, join
import calendar
from datetime import datetime as dt
#import warnings
#warnings.simplefilter('error')

#dic=pd.read_csv('./nimbus/data/system/dictionary.csv', index_col=0)
#dicfx=pd.read_csv('./nimbus/data/system/dictionary_fx.csv', index_col=0)


def get_hist(filename, maxlookback):
    data = pd.read_csv(filename, index_col=0)[-maxlookback:]
    print(data.shape[0], filename, data)
    return data
    
def get_history(filename, lookback, maxlookback):
    data = pd.read_csv(filename, index_col=0, header=None)[-maxlookback:]
    print('lookback',lookback,'maxlookback', maxlookback, filename,\
          'check', data.shape[0]==maxlookback)
    #print(data.shape[0], filename, data.shape[0]==maxlookback, '\n', data)

    for i, j in enumerate(range(lookback, data.shape[0]+1)):
        yield data.iloc[i:j]

class FX(object):
    
    def __init__(self, lookback=20):
        self.dic = pd.read_csv('./nimbus/data/system/dictionary_fx.csv', index_col=0)
        self.atr=pd.DataFrame()
        self.data=pd.DataFrame()
        self.correlations=pd.DataFrame()
        self.lookback = lookback
        self.data_path='./nimbus/data/csidata/fx/'
        self.debug_path='./nimbus/debug/'
        self.last_date=None
        self.first_date=None
        self.rates={}
        self.rates_history={}
        self.history=pd.DataFrame()
        self.data_generators={}
        self.corr_history={}
        self.days=1

    def create_history(self, **kwargs):
        self.lookback=kwargs.get('lookback', self.lookback)
        days=kwargs.get('days', 0)
        
        #atrlookback + pc lookback(1)
        maxlookback=kwargs.get('maxlookback', self.lookback+1)
        lookback2=self.lookback+1
        
        lookback=self.lookback

        cDictCSI=self.dic.to_dict()['filename']
        currencyPairs=cDictCSI.keys()
        
        for i, pair in enumerate(currencyPairs):
            #end at -1 to ignore  new day. 
            filename = self.data_path+cDictCSI[pair]
            if days==0:
                self.data_generators[pair] = get_history(filename, lookback2,\
                                                        maxlookback)
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
            atr=ATR2(data.High.values,data.Low.values,\
                                             data.Close.values,lookback)
            pc=data.Close.pct_change()[1:]
            priorSig=np.where(pc<0,-1,1)[-1]
            self.data[pair]=pc
            
            self.atr.set_value(pair,'Close',data.Close.iloc[-1])
            #self.atr.set_value(pair,'Last',data.Close.iloc[-1])
            self.atr.set_value(pair,'PC',pc.iloc[-1])
            self.atr.set_value(pair,'ACT',priorSig)
            self.atr.set_value(pair,'ATR'+str(lookback),atr[-1])
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
        start_index=self.lookback+days
        self.rates_history={}
        self.history=pd.DataFrame()
        self.corr_history={}
        print('\n\nCreating {} day(s) of FX history'.format(days))
        for i in range(days):
            print('\nDay',i+1)
            self.create_history(maxlookback=start_index, days=i)
            self.history=self.history.append(self.atr)
            self.rates_history[self.last_date.strftime('%Y%m%d')]=self.rates
            self.corr_history[self.last_date.strftime('%Y%m%d')]=self.correlations
            

class Futures(object):
    
    def __init__(self, lookback=20):
        self.dic = pd.read_csv('./nimbus/data/system/dictionary.csv', index_col=0)\
                        .reset_index().set_index('CSIsym')
        self.fx=FX()
        self.atr=pd.DataFrame()
        self.data=pd.DataFrame()
        self.correlations=pd.DataFrame()
        self.lookback = lookback
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
        self.corr_history={}
        self.days=1
        
    def create_history(self, **kwargs):
        self.lookback=kwargs.get('lookback', self.lookback)
        days=kwargs.get('days', 0)
        
        #atrlookback + pc lookback(1)
        maxlookback=kwargs.get('maxlookback', self.lookback+1)
        lookback2=self.lookback+1
        lookback=self.lookback        
        
        #print('Creating Futures ATR file...')            
        futuresDF=self.dic.copy()
        corrDF=pd.DataFrame()

        for i,contract in enumerate(self.markets):
            
            #data = pd.read_csv(self.data_path+contract+'_B.csv',\
            #                   index_col=0, header=None)[-lookback-1:]
            filename=self.data_path+contract+'_B.csv'
            if days==0:
                self.data_generators[contract] = get_history(filename, lookback2,\
                                                        maxlookback)
            #print pair, data.index[-1]
            data=next(self.data_generators[contract])
            data.index = pd.to_datetime(data.index,format='%Y%m%d')
            data.columns = ['Open','High','Low','Close','Volume','OI','R','S']
            data.index.name = 'Dates'
            #data.R = data.R.astype(int)
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
                
            atr=ATR2(data.High.values,data.Low.values,data.Close.values,lookback)
            pc=data.Close.pct_change()
            act=np.where(pc<0,-1,1)
            
            #print (sym, pc, atr, data.tail())
            if 'YT' not in contract:
                sym = ''.join([i for i in contract.split('_')[0] if not i.isdigit()])
            else:
                sym=contract
                
            contractYear=str(data.R[-1])[3]
            contractMonth=str(data.R[-1])[-2:]
            #print(sym+self.months[int(contractMonth)]+contractYear)
            Contract=sym+self.months[int(contractMonth)]+contractYear
            #print sym, atr[-1], c2contractSpec[sym][2], c2contractSpec[sym][1]
            #usdATR = atr[-1]*c2contractSpec[sym][2]/c2contractSpec[sym][1]
            currency=self.dic.loc[sym].currency
            rate=self.fx.rates_history[last_date.strftime('%Y%m%d')][currency]
            multiplier=self.dic.loc[sym].CSImultiplier
            multiplier2=multiplier/rate
            usdATR = atr[-1]*multiplier2
            #print(usdATR)
            contract_value = data.Close[-1]*multiplier2
            #print(contract_value)
            #group=self.dic.loc['AD'].Group
            
            self.data[sym]=pc[1:]
            futuresDF.set_value(sym,'Contract',Contract)
            futuresDF.set_value(sym,'Close',data.Close[-1])
            futuresDF.set_value(sym,'ATR'+str(lookback),atr[-1])
            futuresDF.set_value(sym,'LastPctChg',pc[-1])
            futuresDF.set_value(sym,'ACT',act[-1])
            #futuresDF.set_value(sym,'prevACT',act[-2])
            futuresDF.set_value(sym,'usdATR',usdATR)
            #futuresDF.set_value(sym,'QTY',qty)
            #futuresDF.set_value(sym,'QTY_MINI',qty_mini)
            #futuresDF.set_value(sym,'QTY_MICRO',qty_micro)
            futuresDF.set_value(sym,'contract_value',contract_value)
            #futuresDF.set_value(sym,'FX',c2contractSpec[sym][1])
            #futuresDF.set_value(sym,'PC'+str(data.index[-1]),pc[-1])
            #futuresDF.set_value(sym,'Close'+str(data.index[-1]),data.Close[-1])
            #futuresDF.set_value(sym,'group',c2contractSpec[sym][3])
            #futuresDF.set_value(sym,'RiskOn',c2contractSpec[sym][4])
            #futuresDF.set_value(sym,'Custom',c2contractSpec[sym][5])
            #date
            futuresDF.set_value(sym,'first_date',first_date)
            futuresDF.set_value(sym,'last_date',last_date)
            
        self.atr=futuresDF
        #self.last_date = self.data.index[-1].strftime('%Y%m%d')
        #self.first_date = self.data.index[0].strftime('%Y%m%d')
        if self.atr.shape[0] != self.atr.dropna().shape[0]:
            print('Nans found in {}. atr shape {}, dropna {}\n'.format(self.last_date,\
                          self.atr.shape[0], self.atr.dropna().shape[0]))
        else:
            print('Futures data created for {} {}\n'.format(self.last_date,\
                                                              self.atr.shape))
            
        self.correlations=self.data.corr()
    
    def create(self, portfolio, system, params, days):
        start_index=self.lookback+days
        self.days=days
        self.history=pd.DataFrame()
        self.corr_history={}
        self.fx.create(days+5)
        print('\n\nCreating {} day(s) of Futures history'.format(days))
        for i in range(days):
            print('\nDay',i+1,)
            self.create_history(maxlookback=start_index, days=i)
            self.history=self.history.append(self.atr)
            #self.rates_history[self.last_date.strftime('%Y%m%d')]=self.rates
            self.corr_history[self.last_date.strftime('%Y%m%d')]=self.correlations
            
            if i==0:
                signals=self.atr[system]
                portfolio.create(self, signals, **params)
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
                signals=self.atr[system]
                portfolio.update(self, signals)
            
    def save_correlations(self, show=False):
        if self.correlations.shape[0]<1:
            print('create() first!')
            return


#fut=Futures()
#fut.create(days=2)
'''
#fx=FX()
#fx.create()
account=  {
    'id': "25k-chip",
    'display': "25K",
    'amount': 500000
  }
portfolio = Portfolio(account)
portfolio.create()
fut=Futures(portfolio)
fut.create()
portfolio.update_target(fut)
#portfolio.save()
response=portfolio.update_margin(fut, fut.atr.signals_Excess)
'''

