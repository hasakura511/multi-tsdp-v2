# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 10:46:08 2016

@author: Hidemi
"""
import time
import math
import numpy as np
import pandas as pd
import sqlite3
from pandas.io import sql
from os import listdir
from os.path import isfile, join
import calendar
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
import time
import calendar
import inspect
import os
import os.path
import sys
import ssl
import logging
from copy import deepcopy
from suztoolz.transform import ATR2
import matplotlib.pyplot as plt
import seaborn as sns
from suztoolz.datatools.seasonalClass import seasonalClassifier

def fixTypes(original, transformed):
    for x in original.index:
        #print x, type(series[x]),
        transformed[x]=transformed[x].astype(type(original[x]))
    return transformed
    

def vol_adjsize_live(debug, threadlist):
    try:
        start_time = time.time()
        #signal file version
        version='v4'
        safefAdjustment=0
        #atr lookback
        lookback=20
        #c2safef=1

        if debug:
            mode = 'append'
            #marketList=[sys.argv[1]]
            showPlots=False
            dbPath='C:/Users/Hidemi/Desktop/Python/TSDP/ml/data/futures.sqlite3' 
            dbPath2='D:/ML-TSDP/data/futures.sqlite3' 
            dbPathWeb = 'D:/ML-TSDP/web/tsdp/db.sqlite3'
            dataPath='D:/ML-TSDP/data/csidata/v4futures4/'
            savePath= 'C:/Users/Hidemi/Desktop/Python/TSDP/ml/data/results/' 
            pngPath = 'C:/Users/Hidemi/Desktop/Python/TSDP/ml/data/results/' 
            feedfile='D:/ML-TSDP/data/systems/system_ibfeed.csv'
            #test last>old
            #dataPath2=pngPath
            #signalPath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/signals/' 
            
            #test last=old
            dataPath2='D:/ML-TSDP/data/'
            
            #signalPath ='D:/ML-TSDP/data/signals2/'
            signalPath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/signals/' 
            signalSavePath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/signals/' 
            systemPath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/systems/' 
            readConn = sqlite3.connect(dbPath2)
            writeConn= sqlite3.connect(dbPath)
            readWebConn = sqlite3.connect(dbPathWeb)
            logging.basicConfig(filename='C:/logs/vol_adjsize_live_func_error.log',level=logging.DEBUG)
        else:
            mode= 'append'
            #marketList=[sys.argv[1]]
            showPlots=False
            feedfile='./data/systems/system_ibfeed.csv'
            dbPath='./data/futures.sqlite3'
            dbPathWeb ='./web/tsdp/db.sqlite3'
            #dataPath='./data/csidata/v4futures4/'
            dataPath='./data/csidata/v4futures4/'
            dataPath2='./data/'
            savePath='./data/'
            signalPath = './data/signals2/' 
            signalSavePath = './data/signals2/' 
            pngPath = './web/tsdp/betting/static/images/'
            systemPath =  './data/systems/'
            readConn = writeConn= sqlite3.connect(dbPath)
            readWebConn = sqlite3.connect(dbPathWeb)
            logging.basicConfig(filename='/logs/vol_adjsize_live_func_error.log',level=logging.DEBUG)
        
        updatedSymbols = [x[0] for x in threadlist]
        ff = pd.read_csv(feedfile, index_col='CSIsym')
        files = [ f for f in listdir(dataPath) if isfile(join(dataPath,f)) ]
        marketList = [x.split('_')[0] for x in files]
        marketList = [x for x in marketList if x in ff.ix[updatedSymbols].CSIsym2.values]
        
        if len(marketList)>0:
            print 'Found symbols to update', marketList
        else:
            print 'Found no symbols to update:', updatedSymbols
            return
        #fxRates=pd.read_csv(dataPath2+currencyFile, index_col=0)
        #futuresDF_all=pd.read_csv(dataPath2+'futuresATR.csv', index_col=0)
        futuresDF_all=pd.read_sql('select * from futuresDF_all where timestamp=\
                    (select max(timestamp) from futuresDF_all as maxtimestamp)', con=readConn,  index_col='CSIsym')
        fxRates=pd.read_sql('select * from currenciesDF where timestamp=\
                    (select max(timestamp) from currenciesDF as maxtimestamp)', con=readConn,  index_col='CSIsym')
        accountInfo=pd.read_sql('select * from accountInfo where timestamp=\
                    (select max(timestamp) from accountInfo as maxtimestamp)', con=readConn,  index_col='index')
        systems = [x for x in accountInfo.columns if x not in ['Date','timestamp']]
        riskEquity=int(accountInfo.v4futures.riskEquity)
        riskEquity_mini=int(accountInfo.v4mini.riskEquity)
        riskEquity_micro=int(accountInfo.v4micro.riskEquity)  
         
        #for csv system files
        systemFilename='system_v4futures.csv'
        systemFilename2='system_v4mini.csv'
        systemFilename3='system_v4micro.csv'
        systemFilename_tosave='system_v4futures_live.csv'
        systemFilename2_tosave='system_v4mini_live.csv'
        systemFilename3_tosave='system_v4micro_live.csv'
             
        offline =eval(accountInfo.v4futures.offline)
        offline_mini = eval(accountInfo.v4mini.offline)
        offline_micro =eval(accountInfo.v4micro.offline)

        c2system=accountInfo.v4futures.selection
        c2system_mini=accountInfo.v4mini.selection
        c2system_micro=accountInfo.v4micro.selection

        c2id_macro=int(accountInfo.v4futures.c2id)
        c2id_mini=int(accountInfo.v4mini.c2id)
        c2id_micro=int(accountInfo.v4micro.c2id)

        signals = ['ACT','prevACT','AntiPrevACT','RiskOn','RiskOff','Custom','AntiCustom',\
                        'LastSIG', '0.75LastSIG','0.5LastSIG','1LastSIG','Anti1LastSIG','Anti0.75LastSIG','Anti0.5LastSIG',\
                        'LastSEA','AntiSEA','AdjSEA','AntiAdjSEA',\
                        'Voting','Voting2','Voting3','Voting4','Voting5','Voting6','Voting7','Voting8','Voting9',\
                        'Voting10','Voting11','Voting12','Voting13','Voting14','Voting15']
        #refresh=False
        #currencyFile = 'currenciesATR.csv'





        for i,col in enumerate(fxRates.columns):
            if 'Last' in col:
                fxRates = fxRates[fxRates.columns[i]]
                break

        fxDict={
            'AUD':1/fxRates.ix['AUDUSD'],
            'CAD':fxRates.ix['USDCAD'],
            'CHF':fxRates.ix['USDCHF'],
            'EUR':1/fxRates.ix['EURUSD'],
            'GBP':1/fxRates.ix['GBPUSD'],
            'HKD':fxRates.ix['USDHKD'],
            'JPY':fxRates.ix['USDJPY'],
            'NZD':1/fxRates.ix['NZDUSD'],
            'SGD':fxRates.ix['USDSGD'],
            'USD':1,
            }
        #csisym:[c2sym,usdFXrate,multiplier,riskON signal,Custom]
        c2contractSpec = {
        'AC':['@AC',fxDict['USD'],29000,'energy',1,1],
        'AD':['@AD',fxDict['USD'],100000,'currency',1,1],
        'AEX':['AEX',fxDict['EUR'],200,'index',1,1],
        'BO':['@BO',fxDict['USD'],600,'grain',1,1],
        'BP':['@BP',fxDict['USD'],62500,'currency',1,1],
        'C':['@C',fxDict['USD'],50,'grain',1,1],
        'CC':['@CC',fxDict['USD'],10,'soft',1,1],
        'CD':['@CD',fxDict['USD'],100000,'currency',1,1],
        'CGB':['CB',fxDict['CAD'],1000,'rates',-1,1],
        'CL':['QCL',fxDict['USD'],1000,'energy',1,1],
        'CT':['@CT',fxDict['USD'],500,'soft',1,1],
        'CU':['@EU',fxDict['USD'],125000,'currency',1,1],
        'DX':['@DX',fxDict['USD'],1000,'currency',-1,-1],
        'EBL':['BD',fxDict['EUR'],1000,'rates',-1,1],
        'EBM':['BL',fxDict['EUR'],1000,'rates',-1,1],
        'EBS':['EZ',fxDict['EUR'],1000,'rates',-1,1],
        'ED':['@ED',fxDict['USD'],2500,'rates',-1,1],
        'EMD':['@EMD',fxDict['USD'],100,'index',1,1],
        'ES':['@ES',fxDict['USD'],50,'index',1,1],
        'FC':['@GF',fxDict['USD'],500,'meat',1,-1],
        'FCH':['MT',fxDict['EUR'],10,'index',1,1],
        'FDX':['DXM',fxDict['EUR'],5,'index',1,1],
        'FEI':['IE',fxDict['EUR'],2500,'rates',-1,1],
        'FFI':['LF',fxDict['GBP'],10,'index',1,1],
        'FLG':['LG',fxDict['GBP'],1000,'rates',-1,1],
        'FSS':['LL',fxDict['GBP'],1250,'rates',-1,-1],
        'FV':['@FV',fxDict['USD'],1000,'rates',-1,1],
        'GC':['QGC',fxDict['USD'],100,'metal',-1,1],
        'HCM':['HHI',fxDict['HKD'],50,'index',1,1],
        'HG':['QHG',fxDict['USD'],250,'metal',1,1],
        'HIC':['HSI',fxDict['HKD'],50,'index',1,1],
        'HO':['QHO',fxDict['USD'],42000,'energy',1,1],
        'JY':['@JY',fxDict['USD'],125000,'currency',-1,1],
        'KC':['@KC',fxDict['USD'],375,'soft',1,1],
        'KW':['@KW',fxDict['USD'],50,'grain',1,1],
        'LB':['@LB',fxDict['USD'],110,'soft',1,-1],
        'LC':['@LE',fxDict['USD'],400,'meat',1,-1],
        'LCO':['EB',fxDict['USD'],1000,'energy',1,1],
        'LGO':['GAS',fxDict['USD'],100,'energy',1,1],
        'LH':['@HE',fxDict['USD'],400,'meat',1,-1],
        'LRC':['LRC',fxDict['USD'],10,'soft',1,1],
        'LSU':['QW',fxDict['USD'],50,'soft',1,1],
        'MEM':['@MME',fxDict['USD'],50,'index',1,1],
        'MFX':['IB',fxDict['EUR'],10,'index',1,1],
        'MP':['@PX',fxDict['USD'],500000,'currency',1,1],
        'MW':['@MW',fxDict['USD'],50,'grain',1,1],
        'NE':['@NE',fxDict['USD'],100000,'currency',1,1],
        'NG':['QNG',fxDict['USD'],10000,'energy',1,1],
        'NIY':['@NKD',fxDict['JPY'],500,'index',1,-1],
        'NQ':['@NQ',fxDict['USD'],20,'index',1,1],
        'O':['@O',fxDict['USD'],50,'grain',1,-1],
        'OJ':['@OJ',fxDict['USD'],150,'soft',1,1],
        'PA':['QPA',fxDict['USD'],100,'metal',1,1],
        'PL':['QPL',fxDict['USD'],50,'metal',-1,1],
        'RB':['QRB',fxDict['USD'],42000,'energy',1,1],
        'RR':['@RR',fxDict['USD'],2000,'grain',1,-1],
        'RS':['@RS',fxDict['CAD'],20,'grain',1,-1],
        'S':['@S',fxDict['USD'],50,'grain',1,-1],
        'SB':['@SB',fxDict['USD'],1120,'soft',1,1],
        'SF':['@SF',fxDict['USD'],125000,'currency',1,1],
        'SI':['QSI',fxDict['USD'],50,'metal',-1,1],
        'SIN':['IN',fxDict['USD'],2,'index',1,1],
        'SJB':['BB',fxDict['JPY'],100000,'rates',-1,1],
        'SM':['@SM',fxDict['USD'],100,'grain',1,-1],
        'SMI':['SW',fxDict['CHF'],10,'index',1,1],
        'SSG':['SS',fxDict['SGD'],200,'index',1,-1],
        'STW':['TW',fxDict['USD'],100,'index',1,1],
        'SXE':['EX',fxDict['EUR'],10,'index',1,1],
        'TF':['@TFS',fxDict['USD'],100,'index',1,1],
        'TU':['@TU',fxDict['USD'],2000,'rates',-1,1],
        'TY':['@TY',fxDict['USD'],1000,'rates',-1,1],
        'US':['@US',fxDict['USD'],1000,'rates',-1,1],
        'VX':['@VX',fxDict['USD'],1000,'index',-1,-1],
        'W':['@W',fxDict['USD'],50,'grain',1,1],
        'YA':['AP',fxDict['AUD'],25,'index',1,-1],
        'YB':['HBS',fxDict['AUD'],2400,'rates',-1,1],
        'YM':['@YM',fxDict['USD'],5,'index',1,1],
        'YT2':['HTS',fxDict['AUD'],2800,'rates',-1,1],
        'YT3':['HXS',fxDict['AUD'],8000,'rates',-1,1],
            }
            
        months = {
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


            






        #futuresDF_all=pd.read_csv(dataPath2+'futuresATR_Signals.csv', index_col=0)

        #oldDate=dt.strptime(futuresDF_all.index.name,"%Y-%m-%d %H:%M:%S")
        futuresDF=pd.DataFrame()
        corrDF=pd.DataFrame()

        for i,contract in enumerate(marketList):
            data = pd.read_csv(dataPath+contract+'_B.csv', index_col=0, header=None)[-lookback-1:]
            data.index = pd.to_datetime(data.index,format='%Y%m%d')
            data.columns = ['Open','High','Low','Close','Volume','OI','R','S']
            data.index.name = 'Dates'
            data.R = data.R.astype(int)
            atr=ATR2(data.High.values,data.Low.values,data.Close.values,lookback)
            pc=data.Close.pct_change()
            act=np.where(pc<0,-1,1)
            
            if i==0:
                lastDate = data.index[-1]
            else:
                if data.index[-1]> lastDate:
                    lastDate=data.index[-1]
            #print pc
            #print sym, atr,data.tail()
            if 'YT' not in contract:
                sym = ''.join([i for i in contract.split('_')[0] if not i.isdigit()])
            else:
                sym=contract
            contractYear=str(data.R[-1])[3]
            contractMonth=str(data.R[-1])[-2:]
            contractName=c2contractSpec[sym][0]+months[int(contractMonth)]+contractYear
            #print sym, atr[-1], c2contractSpec[sym][2], c2contractSpec[sym][1]
            usdATR = atr[-1]*c2contractSpec[sym][2]/c2contractSpec[sym][1]
            cValue = data.Close[-1]*c2contractSpec[sym][2]/c2contractSpec[sym][1]
            qty = int(max(math.floor(riskEquity/usdATR),1))
            qty_mini = int(math.ceil(riskEquity_mini/usdATR))
            qty_micro = int(math.ceil(riskEquity_micro/usdATR))
            #print sym, data.R[-1], contractName
            #signalFilename='v4_'+sym+'.csv'
            corrDF[sym]=pc
            futuresDF.set_value(sym,'Contract',contractName)
            futuresDF.set_value(sym,'LastClose',data.Close[-1])
            futuresDF.set_value(sym,'ATR'+str(lookback),atr[-1])
            futuresDF.set_value(sym,'LastPctChg',pc[-1])
            futuresDF.set_value(sym,'ACT',act[-1])
            #futuresDF.set_value(sym,'prevACT',act[-2])
            futuresDF.set_value(sym,'usdATR',usdATR)
            futuresDF.set_value(sym,'QTY',qty)
            futuresDF.set_value(sym,'QTY_MINI',qty_mini)
            futuresDF.set_value(sym,'QTY_MICRO',qty_micro)
            futuresDF.set_value(sym,'contractValue',cValue)
            futuresDF.set_value(sym,'FX',c2contractSpec[sym][1])
            #futuresDF.set_value(sym,'PC'+str(data.index[-1]),pc[-1])
            #futuresDF.set_value(sym,'Close'+str(data.index[-1]),data.Close[-1])
            futuresDF.set_value(sym,'group',c2contractSpec[sym][3])
            futuresDF.set_value(sym,'RiskOn',c2contractSpec[sym][4])
            futuresDF.set_value(sym,'Custom',c2contractSpec[sym][5])
            
        futuresDF.index.name = lastDate


        print ".. skipping seasonalClassifier, copying seasonality from futuresDF_all"
        #second time load  "old" file for seasonality signals.
        nextColOrder = ['0.75LastSIG','0.5LastSIG','1LastSIG','LastSEA','LastSRUN','vSTART']
        futuresDF['vSTART']=futuresDF_all.vSTART
        futuresDF['LastSEA']=futuresDF_all.LastSEA
        futuresDF['LastSRUN']=futuresDF_all.LastSRUN


            
        for i2,contract in enumerate(marketList):
            #print i,
            if 'YT' not in contract:
                sym = ''.join([i for i in contract.split('_')[0] if not i.isdigit()])
            else:
                sym=contract
            signalFilename=version+'_'+sym+'.csv'

            #print signalFilename
            data = pd.read_csv(signalPath+signalFilename, index_col=0)

                    
            if sym in offline:
                adjQty=0
            else:
                if data.safef.iloc[-1] ==1:
                    adjQty = int(round(futuresDF.ix[sym].QTY*(1+safefAdjustment)))
                else:
                    adjQty = int(round(futuresDF.ix[sym].QTY*(1-safefAdjustment)))
                
            futuresDF.set_value(sym,'LastSIG',data.signals.iloc[-1])
            futuresDF.set_value(sym,'LastSAFEf',data.safef.iloc[-1])
            futuresDF.set_value(sym,'finalQTY',adjQty)
            #futuresDF.set_value(sym,'SIG'+str(data.index[-1]),data.signals.iloc[-1])
           

        for i2,contract in enumerate(marketList):
            #print i,
            if 'YT' not in contract:
                sym = ''.join([i for i in contract.split('_')[0] if not i.isdigit()])
            else:
                sym=contract
            signalFilename='0.5_'+sym+'_1D.csv'
            #print signalFilename
            data = pd.read_csv(signalPath+signalFilename, index_col=0)
            data.index = pd.to_datetime(data.index,format='%Y-%m-%d')
            if i2==0:
                sigDate = data.index[-1]
            else:
                if data.index[-1]> sigDate:
                    sigDate=data.index[-1]
                    
            if sym in offline:
                adjQty=0
            else:
                if data.dpsSafef.iloc[-1] ==1:
                    adjQty = int(round(futuresDF.ix[sym].QTY*(1+safefAdjustment)))
                else:
                    adjQty = int(round(futuresDF.ix[sym].QTY*(1-safefAdjustment)))
                    
            futuresDF.set_value(sym,'0.5LastSIG',data.signals.iloc[-1])
            futuresDF.set_value(sym,'0.5LastSAFEf',data.dpsSafef.iloc[-1])
            futuresDF.set_value(sym,'0.5finalQTY',adjQty)
            #futuresDF.set_value(sym,'0.5SIG'+str(data.index[-1]),data.signals.iloc[-1])

        for i2,contract in enumerate(marketList):
            #print i,
            if 'YT' not in contract:
                sym = ''.join([i for i in contract.split('_')[0] if not i.isdigit()])
            else:
                sym=contract
            signalFilename='0.75_'+sym+'_1D.csv'
            #print signalFilename
            data = pd.read_csv(signalPath+signalFilename, index_col=0)
            if sym in offline:
                adjQty=0
            else:
                if data.dpsSafef.iloc[-1] ==1:
                    adjQty = int(round(futuresDF.ix[sym].QTY*(1+safefAdjustment)))
                else:
                    adjQty = int(round(futuresDF.ix[sym].QTY*(1-safefAdjustment)))
                    
            futuresDF.set_value(sym,'0.75LastSIG',data.signals.iloc[-1])
            futuresDF.set_value(sym,'0.75LastSAFEf',data.dpsSafef.iloc[-1])
            futuresDF.set_value(sym,'0.75finalQTY',adjQty)
            #futuresDF.set_value(sym,'0.75SIG'+str(data.index[-1]),data.signals.iloc[-1])

        for i2,contract in enumerate(marketList):
            #print i,
            if 'YT' not in contract:
                sym = ''.join([i for i in contract.split('_')[0] if not i.isdigit()])
            else:
                sym=contract
            signalFilename='1_'+sym+'_1D.csv'
            #print signalFilename
            data = pd.read_csv(signalPath+signalFilename, index_col=0)
            if sym in offline:
                adjQty=0
            else:
                if data.dpsSafef.iloc[-1] ==1:
                    adjQty = int(round(futuresDF.ix[sym].QTY*(1+safefAdjustment)))
                else:
                    adjQty = int(round(futuresDF.ix[sym].QTY*(1-safefAdjustment)))
                    
            futuresDF.set_value(sym,'1LastSIG',data.signals.iloc[-1])
            futuresDF.set_value(sym,'1LastSAFEf',data.dpsSafef.iloc[-1])
            futuresDF.set_value(sym,'1finalQTY',adjQty)
            #futuresDF.set_value(sym,'1SIG'+str(data.index[-1]),data.signals.iloc[-1])

        futuresDF=futuresDF.sort_index()
        columns = futuresDF.columns.tolist()
        #futuresDF.LastPctChg=futuresDF[sorted([x for x in columns if 'PC' in x])[-1]].values
        start_idx =columns.index('ACT')+1
        new_order = columns[:start_idx]+nextColOrder
        new_order =  new_order+[x for x in columns if x not in new_order]
        futuresDF = futuresDF[new_order]
        print futuresDF.iloc[:,:4]
        print 'Creating voting signals..'

        AdjSEACols= ['RiskOn','AntiPrevACT','LastSEA']
        votingCols =['Anti1LastSIG','prevACT','LastSEA']
        voting2Cols = ['0.5LastSIG','AntiPrevACT','AdjSEA']
        voting3Cols = ['Anti0.75LastSIG','AntiPrevACT','AntiSEA']
        voting4Cols=['Voting','Voting2','Voting3','Voting8']
        #voting5Cols=['prevACT','Anti1LastSIG','AntiAdjSEA']
        voting5Cols=['Voting','Voting2','Voting3','Voting14']
        #voting6Cols = ['0.5LastSIG','prevACT','AntiSEA']
        voting6Cols =['Voting','Voting2','Voting3','Voting15']
        voting7Cols = ['RiskOn','0.5LastSIG','AntiSEA']
        voting8Cols = ['RiskOn','0.5LastSIG','AntiPrevACT']
        #voting9Cols = ['RiskOn','0.5LastSIG','AntiSEA']
        voting9Cols= ['RiskOn','AntiPrevACT','AntiSEA']
        voting10Cols = ['RiskOn','0.5LastSIG','AntiSEA','AntiPrevACT']
        voting11Cols = ['RiskOn','Anti0.75LastSIG','AntiSEA','AntiPrevACT']
        voting12Cols = ['1LastSIG','AntiPrevACT','AdjSEA']
        voting13Cols = ['RiskOn','0.75LastSIG','AntiSEA','prevACT']
        voting14Cols = ['RiskOn','Anti1LastSIG','LastSEA','prevACT']
        voting15Cols = ['RiskOff','Anti0.75LastSIG','AntiSEA']

        #voting9Cols=['Anti1LastSIG','AntiSEA']
        #voting4Cols= votingCols+voting2Cols+voting3Cols
        futuresDF['None']=0
        futuresDF['RiskOff']=np.where(futuresDF.RiskOn<0,1,-1)
        futuresDF['AntiCustom']=np.where(futuresDF.Custom<0,1,-1)
        futuresDF['Anti1LastSIG'] = np.where(futuresDF['1LastSIG']==1,-1,1)
        futuresDF['Anti0.75LastSIG'] = np.where(futuresDF['0.75LastSIG']==1,-1,1)
        futuresDF['Anti0.5LastSIG'] = np.where(futuresDF['0.5LastSIG']==1,-1,1)
        futuresDF['AntiSEA'] = np.where(futuresDF.LastSEA==1,-1,1)
        futuresDF['prevACT'] = futuresDF.ACT
        futuresDF['AntiPrevACT'] = np.where(futuresDF.ACT==1,-1,1)
        #futuresDF['AdjSEA'] = np.where(futuresDF.LastSRUN <0, futuresDF.LastSEA*-1, futuresDF.LastSEA)
        futuresDF['AdjSEA']=np.where(futuresDF[AdjSEACols].sum(axis=1)<0,-1,1)
        futuresDF['AntiAdjSEA'] = np.where(futuresDF.AdjSEA==1,-1,1)
        futuresDF['Voting']=np.where(futuresDF[votingCols].sum(axis=1)<0,-1,1)
        futuresDF['Voting2']=np.where(futuresDF[voting2Cols].sum(axis=1)<0,-1,1)
        futuresDF['Voting3']=np.where(futuresDF[voting3Cols].sum(axis=1)<0,-1,1)
        #futuresDF['Voting4']=np.where(futuresDF[voting4Cols].sum(axis=1)<0,-1,1)
        #futuresDF['Voting5']=np.where(futuresDF[voting5Cols].sum(axis=1)<0,-1,1)
        #futuresDF['Voting6']=np.where(futuresDF[voting6Cols].sum(axis=1)<0,-1,1)
        futuresDF['Voting7']=np.where(futuresDF[voting7Cols].sum(axis=1)<0,-1,1)
        futuresDF['Voting8']=np.where(futuresDF[voting8Cols].sum(axis=1)<0,-1,1)
        #futuresDF['Voting9']=np.where(futuresDF[voting9Cols].sum(axis=1)<0,-1,1)
        v9=futuresDF[voting9Cols].sum(axis=1)
        v9[v9<0]=-1
        v9[v9>0]=1
        futuresDF['Voting9']=v9.values
        #futuresDF['Voting10']=np.where(futuresDF[voting10Cols].sum(axis=1)<0,-1,1)
        v10=futuresDF[voting10Cols].sum(axis=1)
        v10[v10<0]=-1
        v10[v10>0]=1
        futuresDF['Voting10']=v10.values
        #futuresDF['Voting11']=np.where(futuresDF[voting11Cols].sum(axis=1)<0,-1,1)
        v11=futuresDF[voting11Cols].sum(axis=1)
        v11[v11<0]=-1
        v11[v11>0]=1
        futuresDF['Voting11']=v11.values
        #futuresDF['Voting12']=np.where(futuresDF[voting12Cols].sum(axis=1)<0,-1,1)
        v12=futuresDF[voting12Cols].sum(axis=1)
        v12[v12<0]=-1
        v12[v12>0]=1
        futuresDF['Voting12']=v12.values
        #futuresDF['Voting13']=np.where(futuresDF[voting13Cols].sum(axis=1)<0,-1,1)
        v13=futuresDF[voting13Cols].sum(axis=1)
        v13[v13<0]=-1
        v13[v13>0]=1
        futuresDF['Voting13']=v13.values
        #futuresDF['Voting14']=np.where(futuresDF[voting14Cols].sum(axis=1)<0,-1,1)
        v14=futuresDF[voting14Cols].sum(axis=1)
        v14[v14<0]=-1
        v14[v14>0]=1
        futuresDF['Voting14']=v14.values
        futuresDF['Voting15']=np.where(futuresDF[voting15Cols].sum(axis=1)<0,-1,1)
        #futuresDF['Voting4']=np.where(futuresDF[voting4Cols].sum(axis=1)<0,-1,1)

        #Voting of Voting
        v4=futuresDF[voting4Cols].sum(axis=1)
        v4[v4<0]=-1
        v4[v4>0]=1
        futuresDF['Voting4']=v4.values

        v5=futuresDF[voting5Cols].sum(axis=1)
        v5[v5<0]=-1
        v5[v5>0]=1
        futuresDF['Voting5']=v5.values 

        v6=futuresDF[voting6Cols].sum(axis=1)
        v6[v6<0]=-1
        v6[v6>0]=1
        futuresDF['Voting6']=v6.values



        for group in futuresDF.groupby(by='group').Custom:
            #print group
            colors=np.array(['r']*group[1].shape[0])
            mask = group[1]>0
            colors[mask.values]='g'
            Lper = round(sum(mask.values)/float(group[1].shape[0])*100,1)
            title = 'Custom '+group[0] + ' L% '+ str(Lper)
            group[1].plot(kind='barh', title =title, color=colors)
            plt.xlim(-1,1)

            filename='custom_'+group[0]+'.png'
            plt.savefig(pngPath+filename, bbogroup_inches='tight')
            print 'Saved '+pngPath+filename
            
            #if debug:
            #    plt.show()
            plt.close()
        #print 'Saving signals from', c2system
        #1biv. Run v4size (signals and size) and check system.csv for qty,contracts with futuresATR
        #save signals to v4_ signal files for order processing
        lastDate=int(lastDate.strftime('%Y%m%d'))
        '''
        nsig=0
        for ticker in futuresDF.index:
            #nsig+=1
            #signalFile=pd.read_csv(signalSavePath+ version+'_'+ ticker+ '.csv', index_col=['dates'])
            #addLine = signalFile.iloc[-1]
            #addLine = addLine.append(pd.Series(data=timenow.strftime("%Y%m%d %H:%M:%S %Z"), index=['timestamp']))
            #addLine = addLine.append(pd.Series(data=cycleTime, index=['cycleTime']))
            #addLine.name = sst.iloc[-1].name
            addLine = pd.Series(name=lastDate)
            addLine['CSIsym']=ticker
            #addLine['c2sym']=ticker
            addLine['signals']=futuresDF.ix[ticker][c2system]
            addLine['safef']=c2safef
            addLine['timestamp']=int(calendar.timegm(dt.utcnow().utctimetuple()))
            #signalFile = signalFile.append(addLine)
            #filename=signalSavePath + version+'_'+ ticker+ '.csv'
            
            addLine.to_frame().transpose().to_sql(name='Signals', if_exists='append', con=conn, index=True, index_label='Date')
            print 'Saved sql',  addLine['signals'], addLine['safef']
            #signalFile.to_csv(filename, index=True)
        #print nsig, 'files updated'
        '''

        futuresDF['Date']=lastDate
        futuresDF['timestamp']=int(calendar.timegm(dt.utcnow().utctimetuple()))
        futuresDF.index.name = 'CSIsym'

        try:
            futuresDF.to_sql(name='futuresATRhist', if_exists=mode, con=writeConn, index=True, index_label='CSIsym')
            futuresDF.to_sql(name='futuresATR', if_exists='replace', con=writeConn, index=True, index_label='CSIsym')
            print 'Saved to sql db table', 'futuresATR'
        except Exception as e:
            #print e
            traceback.print_exc()
            
            


        #system file update.
        #load from daily run, save to live (to update pivot dates)


        system = pd.read_csv(systemPath+systemFilename, index_col=0)
        system.index = [x.split('_')[1] for x in system.System]
        system.index.name = 'CSIsym'
        system = system.ix[ff.index]

        system_mini = pd.read_csv(systemPath+systemFilename2, index_col=0)
        system_mini.index = [x.split('_')[1] for x in system_mini.System]
        system_mini.index.name = 'CSIsym'
        system_mini = system_mini.ix[ff.index]

        system_micro = pd.read_csv(systemPath+systemFilename3, index_col=0)
        system_micro.index = [x.split('_')[1] for x in system_micro.System]
        system_micro.index.name = 'CSIsym'
        system_micro = system_micro.ix[ff.index]

        #macro
        for sys in system.System:
            sym=sys.split('_')[1]
            if sym in futuresDF.index:
                idx=system[system.System==sys].index[0]
                print 'MACRO', sys, sym, system.ix[idx].c2qty,
                system.set_value(idx,'c2qty',int(futuresDF.ix[sym]['finalQTY']))
                print system.ix[idx].c2qty, system.ix[idx].c2sym,
                system.set_value(idx,'c2sym',futuresDF.ix[sym]['Contract'])
                print system.ix[idx].c2sym
        #system.Name=systemFilename.split('_')[1][:-4]
        system.c2id=c2id_macro

        #mini
        for sys in system_mini.System:
            sym=sys.split('_')[1]
            if sym in futuresDF.index:
                idx=system_mini[system_mini.System==sys].index[0]
                print 'MINI', sys, sym, system_mini.ix[idx].c2qty,
                if sym in offline_mini:
                    system_mini.set_value(idx,'c2qty',0)
                else:
                    system_mini.set_value(idx,'c2qty',int(futuresDF.ix[sym]['QTY_MINI']))
                print system_mini.ix[idx].c2qty, system_mini.ix[idx].c2sym,
                system_mini.set_value(idx,'c2sym',futuresDF.ix[sym]['Contract'])
                print system_mini.ix[idx].c2sym
        #system_mini.Name=systemFilename2.split('_')[1][:-4]
        system_mini.c2id=c2id_mini

        #micro
        for sys in system_micro.System:
            sym=sys.split('_')[1]
            if sym in futuresDF.index:
                idx=system_micro[system_micro.System==sys].index[0]
                print 'MICRO', sys, sym, system_micro.ix[idx].c2qty,
                if sym in offline_micro:
                    system_micro.set_value(idx,'c2qty',0)
                else:
                    system_micro.set_value(idx,'c2qty',int(futuresDF.ix[sym]['QTY_MICRO']))
                print system_micro.ix[idx].c2qty, system_micro.ix[idx].c2sym,
                system_micro.set_value(idx,'c2sym',futuresDF.ix[sym]['Contract'])
                print system_micro.ix[idx].c2sym
        #system_micro.Name=systemFilename3.split('_')[1][:-4]
        system_micro.c2id=c2id_micro 
            

            
        for i,sym in enumerate([x.split('_')[1] for x in system.System]):
            if sym in futuresDF.index:
                system.set_value(sym,'signal',futuresDF[c2system].ix[sym])
                #series=system.ix[i]
                #series.name=sym
                #series=fixTypes(series,series.to_frame().transpose())

        #system.to_csv(systemPath+systemFilename, index=False)
        system['Date']=lastDate
        system['timestamp']=int(calendar.timegm(dt.utcnow().utctimetuple()))
        tablename = 'v4futures_live'
        system.ix[futuresDF.index].to_sql(name=tablename, if_exists='replace', con=writeConn, index=True, index_label='CSIsym')
        system.ix[futuresDF.index].to_sql(name='signals_live', if_exists=mode, con=writeConn, index=True, index_label='CSIsym')
        print tablename,c2system
        print 'Saved to sql db and',  systemPath+systemFilename_tosave
        system.to_csv(systemPath+systemFilename_tosave, index=True)

        for i,sym in enumerate([x.split('_')[1] for x in system_mini.System]):
            if sym in futuresDF.index:
                system_mini.set_value(sym,'signal',futuresDF[c2system_mini].ix[sym])
                #series=system_mini.ix[i]
                #series.name=sym
                #series=fixTypes(series,series.to_frame().transpose())
                
        #system_mini.to_csv(systemPath+systemFilename2, index=False)
        system_mini['Date']=lastDate
        system_mini['timestamp']=int(calendar.timegm(dt.utcnow().utctimetuple()))
        tablename = 'v4mini_live'
        system_mini.ix[futuresDF.index].to_sql(name=tablename, if_exists='replace', con=writeConn, index=True, index_label='CSIsym')
        system_mini.ix[futuresDF.index].to_sql(name='signals_live', if_exists=mode, con=writeConn, index=True, index_label='CSIsym')
        print tablename,c2system_mini
        print 'Saved to sql db and', systemPath+systemFilename2_tosave
        system_mini.to_csv(systemPath+systemFilename2_tosave, index=True)

        for i,sym in enumerate([x.split('_')[1] for x in system_micro.System]):
            if sym in futuresDF.index:
                system_micro.set_value(sym,'signal',futuresDF[c2system_micro].ix[sym])
                #series=system_micro.ix[i]
                #series.name=sym
                #series=fixTypes(series,series.to_frame().transpose())
                
        system_micro['Date']=lastDate
        system_micro['timestamp']=int(calendar.timegm(dt.utcnow().utctimetuple()))
        tablename = 'v4micro_live'
        system_micro.ix[futuresDF.index].to_sql(name= tablename, if_exists='replace', con=writeConn, index=True, index_label='CSIsym')
        system_micro.ix[futuresDF.index].to_sql(name='signals_live', if_exists=mode, con=writeConn, index=True, index_label='CSIsym')
        print tablename, c2system_micro
        print  'Saved to sql db and', systemPath+systemFilename3_tosave
        system_micro.to_csv(systemPath+systemFilename3_tosave, index=True)

        #futuresDF.to_csv(savePath+'futuresATR.csv')
        #print 'Saved', savePath+'futuresATR.csv'
        
        
        print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes ', dt.now()
        
        ordersDict= { 
                        'v4futures':system.ix[futuresDF.index],
                        'v4mini':system_mini.ix[futuresDF.index],
                        'v4micro':system_micro.ix[futuresDF.index],
                        }
        return ordersDict
    except Exception as e:
        logging.exception("message")
        logging.info( str(dt.now()))