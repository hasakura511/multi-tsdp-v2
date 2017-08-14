# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 10:46:08 2016

@author: Hidemi
"""

import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
import io
import traceback
import json
import imp
import urllib
import urllib2
import webbrowser
import re
import datetime
from datetime import datetime as dt
import calendar
import time
import inspect
import os
import os.path
import sys
import ssl
from copy import deepcopy
from suztoolz.transform import ATR2
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import logging

start_time = time.time()
barSizeSetting='4h'
tradingEquity=1000000
riskPerTrade=0.01
riskEquity=tradingEquity*riskPerTrade
lookback=20
refresh=False

if len(sys.argv)==1:
    showPlots = False
    dataPath='D:/ML-TSDP/data/csidata/fx/'
    savePath= 'C:/Users/Hidemi/Desktop/Python/TSDP/ml/data/results/' 
    savePath2 = 'C:/Users/Hidemi/Desktop/Python/TSDP/ml/data/results/' 
    signalPath = 'D:/ML-TSDP/data/signals/' 
    pairPath =  'D:/ML-TSDP/data/' 
    dbPath='C:/Users/Hidemi/Desktop/Python/TSDP/ml/data/futures.sqlite3' 
    logging.basicConfig(filename='C:/logs/vol_adjsize_curr_error.log',level=logging.DEBUG)
    
else:
    showPlots = False
    dataPath='./data/csidata/fx/'
    savePath='./data/'
    signalPath = './data/signals/' 
    savePath2 = './data/results/'
    pairPath =  './data/' 
    dbPath='./data/futures.sqlite3'
    logging.basicConfig(filename='/logs/vol_adjsize_curr_error.log',level=logging.DEBUG)
    
    
#with open(pairPath+'currencies.txt') as f:
#    currencyPairs = f.read().splitlines()

cDictCSI = {
            'AUDNZD':'AD!0000$.CSV',
            'GBPAUD':'AD30000$.CSV',
            'AUDJPY':'AD40000$.CSV',
            'AUDCHF':'AD50000$.CSV',
            'AUDUSD':'AD60000$.CSV',
            'AUDCAD':'AD70000$.CSV',
            'NZDCAD':'CAD90000$.CSV',
            'CADCHF':'CHFX00000$.CSV',
            'NZDCHF':'CHFXC0000$.CSV',
            'GBPNZD':'DR%0000$.CSV',
            'GBPUSD':'DR50000$.CSV',
            'GBPJPY':'DR90000$.CSV',
            'GBPCAD':'DRC0000$.CSV',
            'GBPCHF':'DR^0000$.CSV',
            'EURNZD':'EU#0000$.CSV',
            'EURAUD':'EU10000$.CSV',
            'EURCAD':'EU20000$.CSV',
            'EURJPY':'EU50000$.CSV',
            'EURCHF':'EU60000$.CSV',
            'EURGBP':'EU70000$.CSV',
            'EURUSD':'EU90000$.CSV',
            'CADJPY':'JPYX00000$.CSV',
            'NZDJPY':'JPYXC0000$.CSV',
            'CHFJPY':'JPYXE0000$.CSV',
            'NZDUSD':'QE%0000$.CSV',
            'USDCAD':'QE20000$.CSV',
            'USDJPY':'QE90000$.CSV',
            'USDCHF':'QE}0000$.CSV',
            'USDSGD':'QE{0000$.CSV',
            'USDHKD':'QE70000$.CSV',
            }
conn = sqlite3.connect(dbPath)
currencyPairs=cDictCSI.keys()
try:
    if isfile(savePath+'currenciesATR.csv'):
        currenciesDF_old=pd.read_csv(savePath+'currenciesATR.csv', index_col=0)
        #oldDate=dt(1990,1,1)
        oldDate=dt.strptime(currenciesDF_old.index.name,"%Y-%m-%d %H:%M:%S")
    else:
        oldDate=dt(1990,1,1)
        
    currenciesDF=pd.DataFrame()
    corrDF=pd.DataFrame()
    for pair in currencyPairs:
        #end at -1 to ignore  new day. 
        data = pd.read_csv(dataPath+cDictCSI[pair], index_col=0)[-lookback-1:]
        #print pair, data.index[-1]
        data.index = pd.to_datetime(data.index,format='%Y%m%d')
        data.columns = ['Open','High','Low','Close','S']
        data.index.name = 'Dates'
        atr=ATR2(data.High.values,data.Low.values,data.Close.values,lookback)
        pc=data.Close.pct_change()
        priorSig=np.where(pc<0,-1,1)[-1]
        #print pc
        #print pair, atr,data.tail()
        #signalFilename='v4_'+pair+'.csv'
        corrDF[pair]=pc
        
        currenciesDF.set_value(pair,'Last',data.Close.iloc[-1])
        currenciesDF.set_value(pair,'ATR'+str(lookback),atr[-1])
        currenciesDF.set_value(pair,'Close',data.Close.iloc[-1])
        currenciesDF.set_value(pair,'PC',pc.iloc[-1])
        currenciesDF.set_value(pair,'ACT',priorSig)
        currenciesDF.set_value(pair,'Date',data.index[-1].strftime('%Y%m%d'))
    currenciesDF.index.name=lastDate=data.index[-1]
    #corrDF.to_csv(savePath+'currenciesPCcsv')
    #corrDF.corr().to_csv(savePath+'currenciesCorr.csv')

    if lastDate >oldDate:
        print 'first time run.. generating new currency ATR file.'
        corrDF=corrDF.corr()
        fig,ax = plt.subplots(figsize=(13,13))
        ax.set_title('Correlation '+str(data.index[0])+' to '+str(data.index[-1]))
        sns.heatmap(ax=ax,data=corrDF)
        plt.yticks(rotation=0) 
        plt.xticks(rotation=90) 
        corrDF.to_html(savePath2+'currencies_5.html')

        if savePath != None:
            print 'Saving '+savePath2+'currencies_4.png'
            fig.savefig(savePath2+'currencies_4.png', bbox_inches='tight')
            
        if len(sys.argv)==1 and showPlots:
            print data.index[0],'to',data.index[-1]
            plt.show()
        plt.close()

        for col in corrDF:
            plt.figure(figsize=(8,10))
            corrDF[col].sort_values().plot.barh(color='r')
            plt.axvline(0, color='k')
            plt.title(col+' '+str(lookback)+' Day Correlation '+str(data.index[0])+' to '+str(data.index[-1]))
            plt.xlim(-1,1)
            plt.xticks(np.arange(-1,1.25,.25))
            plt.grid(True)
            filename='v4_'+col+'_CORREL'+'.png'
            if savePath2 != None:
                print 'Saving '+savePath2+filename
                plt.savefig(savePath2+filename, bbox_inches='tight')
            
            if len(sys.argv)==1 and showPlots:
                #print data.index[0],'to',data.index[-1]
                plt.show()
            plt.close()


        '''
        for pair in currencyPairs:
            signalFilename='v4_'+pair+'.csv'
            if isfile(signalPath+signalFilename):
                #print signalFilename
                data = pd.read_csv(signalPath+signalFilename, index_col=0)
                #currenciesDF.set_value(pair,'SIG'+str(data.index[-3]),data.signals.iloc[-3])
                currenciesDF.set_value(pair,'SIG'+str(data.index[-1]),data.signals.iloc[-1])
        '''
        #currenciesDF=currenciesDF.sort_index()
        print currenciesDF
        print 'Saving...', savePath+'currenciesATR.csv'
        currenciesDF.sort_index().to_csv(savePath+'currenciesATR.csv')
        #currenciesDF['Date']=lastDate.strftime('%Y%m%d')
        currenciesDF['timestamp']=int(calendar.timegm(dt.utcnow().utctimetuple()))
        currenciesDF.to_sql(name='currenciesDF',con=conn, index=True, if_exists='append', index_label='CSIsym')
        print 'Saved currenciesDF to', dbPath
    else:
        print 'second time run.. skipping job..'
        pass

    #signalDF=signalDF.sort_index()
    #print signalDF
    #signalDF.to_csv(savePath+'currenciesSignals.csv')
    print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes ', dt.now()
except Exception as e:
    logging.exception("message")
    logging.info( str(dt.now()))