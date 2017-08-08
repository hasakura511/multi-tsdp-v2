# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 10:46:08 2016

@author: Hidemi
"""
import time
import math
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
import sqlite3
import io
import traceback
import json
import imp
import urllib
import urllib2
import requests
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
from suztoolz.datatools.seasonalClass import seasonalClassifier
import logging

start_time = time.time()
version='v4'
systems = ['v4futures','v4mini','v4micro']
lookback=20
refresh=False
currencyFile = 'currenciesATR.csv'
systemFilename='system_v4futures.csv'
systemFilename2='system_v4mini.csv'
systemFilename3='system_v4micro.csv'

c2id_macro=110126294
c2id_mini=110125449
c2id_micro=110125347
c2key=c2key_micro=c2key_mini=c2key_macro='O9WoxVj7DNXkpifMY_blqHpFg5cp3Fjqc7Aiu4KQjb8mXQlEVx'



if len(sys.argv)==1:
    debug=True
    showPlots=False
    #refreshSea tries to recreate the first run futuresATR file after new signals have been generated
    refreshSea=False
    dbPath='./data/futures.sqlite3' 
    dbPath2='./data/futures.sqlite3'
    dbPathWeb = './web/tsdp/db.sqlite3'
    dataPath='./data/csidata/v4futures2/'
    savePath= './data/results/' 
    pngPath=savePath2 = './data/results/' 
    feedfile='.data/systems/system_ibfeed.csv'
    #test last>old
    #dataPath2=savePath2
    #signalPath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/signals/' 
    
    #test last=old
    dataPath2='./data/'
    
    signalPath ='./data/signals/'
    signalSavePath = './data/signals/' 
    systemPath = './data/systems/' 
    logging.basicConfig(filename='/logs/vol_adjsize_error.log',level=logging.DEBUG)
    
else:
    debug=False
    showPlots=False
    #if set to on its probably going to mess up the db.
    refreshSea=False
    dbPathWeb ='./web/tsdp/db.sqlite3'
    dbPath=dbPath2='./data/futures.sqlite3'
    dataPath='./data/csidata/v4futures2/'
    dataPath2='./data/'
    savePath='./data/'
    signalPath = './data/signals/' 
    signalSavePath = './data/signals/' 
    savePath2 = './data/results/'
    pngPath= './web/tsdp/betting/static/images/'
    systemPath =  './data/systems/'
    feedfile='./data/systems/system_ibfeed.csv'
    logging.basicConfig(filename='/logs/vol_adjsize_error.log',level=logging.DEBUG)
    
writeConn = sqlite3.connect(dbPath)
readConn =  sqlite3.connect(dbPath2)
readWebConn = sqlite3.connect(dbPathWeb)
webready = False
if webready:
    #request='http://www.globalsystemsmanagement.net/last_userselection/'
    #selectionDF = pd.DataFrame(requests.get(request).json())
    #selectionDict=eval(selectionDF.selection[0])
    selectionDF=pd.read_sql('select * from betting_userselection where timestamp=\
            (select max(timestamp) from betting_userselection as maxtimestamp)', con=readWebConn, index_col='userID')
    selectionDict=eval(selectionDF.selection.values[0])
    c2system_macro=c2system=selectionDict["v4futures"][0]
    c2system_mini=selectionDict["v4mini"][0]
    c2system_micro=selectionDict["v4micro"][0]
else:
    c2system_macro=c2system='RiskOn'
    c2system_mini='RiskOn'
    c2system_micro='RiskOn'
    
c2safef=1

ComponentsDict ={
                            'Off':'None',
                            'Previous':'prevACT',
                            'Anti-Previous':'AntiPrevACT',
                            'RiskOn':'RiskOn',
                            'RiskOff':'RiskOff',
                            'Custom':'Custom',
                            'Anti-Custom':'AntiCustom',
                            '50/50':'0.75LastSIG',
                            'LowestEquity':'0.5LastSIG',
                            'HighestEquity':'1LastSIG',
                            'AntiHighestEquity':'Anti1LastSIG',
                            'Anti50/50':'Anti0.75LastSIG',
                            'AntiLowestEquity':'Anti0.5LastSIG',
                            'Seasonality':'LastSEA',
                            'Anti-Seasonality':'AntiSEA',
                            }


fxRates=pd.read_csv(dataPath2+currencyFile, index_col=0)
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

##ACCTINFO
acctinfofile='./web/tsdp/accountinfo_data.json'
with open(acctinfofile, 'r') as f:
     acctinfo=json.load(f)
     

riskEquity=int(acctinfo['v4futures']['riskEquity'])
riskEquity_mini=int(acctinfo['v4mini']['riskEquity'])
riskEquity_micro=int(acctinfo['v4micro']['riskEquity'])

#offline =['AC','AEX','CC','CGB','CT','DX','EBL','EBM','EBS','ED','FCH','FDX','FEI','FFI','FLG','FSS','HCM','HIC','KC','KW','LB','LCO','LGO','LRC','LSU','MEM','MFX','MW','O','OJ','RR','RS','SB','SIN','SJB','SMI','SSG','STW','SXE','TF','VX','YA','YB','YT2','YT3',]
#offline_mini = ['AC','AD','AEX','BO','BP','CC','CD','CGB','CT','DX','EBL','EBM','EBS','ED','FC','FCH','FDX','FEI','FFI','FLG','FSS','FV','GC','HCM','HIC','HO','KC','KW','LB','LC','LCO','LGO','LH','LRC','LSU','MEM','MFX','MP','MW','NE','NIY','NQ','O','OJ','PA','PL','RB','RR','RS','S','SB','SF','SI','SIN','SJB','SMI','SSG','STW','SXE','TF','US','VX','YA','YB','YM','YT2','YT3',]
#offline_micro =['AC','AD','AEX','BP','C','CC','CD','CGB','CL','CT','CU','DX','EBL','EBM','EBS','ED','EMD','FC','FCH','FDX','FEI','FFI','FLG','FSS','FV','GC','HCM','HIC','HO','JY','KC','KW','LB','LC','LCO','LGO','LH','LRC','LSU','MEM','MFX','MP','MW','NE','NIY','NQ','O','OJ','PA','PL','RB','RR','RS','S','SB','SF','SI','SIN','SJB','SM','SMI','SSG','STW','SXE','TF','TU','US','VX','W','YA','YB','YM','YT2','YT3',]

offline=[sym for sym in c2contractSpec.keys() if sym not in eval(acctinfo['v4futures']['online'])]
offline_mini=[sym for sym in c2contractSpec.keys() if sym not in eval(acctinfo['v4mini']['online'])]
offline_micro=[sym for sym in c2contractSpec.keys() if sym not in eval(acctinfo['v4micro']['online'])]
'''
c2id_macro=acctinfo['v4futures']['c2id']
c2id_mini=acctinfo['v4mini']['c2id']
c2id_micro=acctinfo['v4micro']['c2id']
c2key_macro=acctinfo['v4futures']['c2key']
c2key_mini=acctinfo['v4mini']['c2key']
c2key_micro=acctinfo['v4micro']['c2key']
'''
accountInfo = pd.DataFrame(data=[[c2system, c2system_mini, c2system_micro]],columns=systems,index=['selection'])
accountInfo = accountInfo.append(pd.DataFrame(data=[[c2id_macro, c2id_mini, c2id_micro]],columns=systems,index=['c2id']))
accountInfo = accountInfo.append(pd.DataFrame(data=[[c2key_macro, c2key_mini, c2key_micro]],columns=systems,index=['c2key']))
accountInfo = accountInfo.append(pd.DataFrame(data=[[riskEquity, riskEquity_mini, riskEquity_micro]],columns=systems,index=['riskEquity']))
accountInfo = accountInfo.append(pd.DataFrame(data=[[str(offline), str(offline_mini), str(offline_micro)]],columns=systems,index=['offline']))
print 'loaded and updated account info from', acctinfofile
#range (-1 to 1) postive for counter-trend negative for trend i.e.
#-1 would 0 safef ==1 and double safef==2
#1 would 0 safef ==2 and double safef==1
safefAdjustment=0

filename='./web/tsdp/custom_signals_data.json'
if isfile(filename):
    with open(filename, 'r') as f:
        custom_signals_data = json.load(f)
        custom_signals_data=custom_signals_data['customsignals']
        custom_signals={sym:custom_signals_data[sym]['signals'] for sym in custom_signals_data.keys()}
#print custom_signals
for sym in custom_signals:
    #print sym, custom_signals[sym]
    if custom_signals[sym] is not None:
        c2contractSpec[sym][5]=int(custom_signals[sym])
print 'loaded and updated custom signals from', filename

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

    
files = [ f for f in listdir(dataPath) if isfile(join(dataPath,f)) ]
marketList = [x.split('_')[0] for x in files]

#futuresDF_old=pd.read_csv(dataPath2+'futuresATR_Signals.csv', index_col=0)


futuresDF=pd.DataFrame()
corrDF=pd.DataFrame()



try:
    for i,contract in enumerate(marketList):
        data = pd.read_csv(dataPath+contract+'_B.csv', index_col=0, header=None)[-lookback-1:]
        data.index = pd.to_datetime(data.index,format='%Y%m%d')
        data.columns = ['Open','High','Low','Close','Volume','OI','R','S']
        data.index.name = 'Dates'
        data.R = data.R.astype(int)
        atr=ATR2(data.High.values,data.Low.values,data.Close.values,lookback)
        pc=data.Close.pct_change()
        #no zeros in act because ACT used in prevACT.
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
        var=riskEquity/usdATR
        qty = int(math.floor(var) if var>1 else math.ceil(var))
        var=riskEquity_mini/usdATR
        qty_mini = int(math.floor(var) if var>1 else math.ceil(var))
        var=riskEquity_micro/usdATR
        qty_micro = int(math.floor(var) if var>1 else math.ceil(var))
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
        futuresDF.set_value(sym,'PC'+str(data.index[-1]),pc[-1])
        futuresDF.set_value(sym,'Close'+str(data.index[-1]),data.Close[-1])
        futuresDF.set_value(sym,'group',c2contractSpec[sym][3])
        futuresDF.set_value(sym,'RiskOn',c2contractSpec[sym][4])
        futuresDF.set_value(sym,'Custom',c2contractSpec[sym][5])
        
    futuresDF.index.name = lastDate
    #feeddata=pd.read_csv(feedfile,index_col='CSIsym')
    #feeddata['Date']=int(lastDate.strftime('%Y%m%d'))
    #feeddata['timestamp']=int(calendar.timegm(dt.utcnow().utctimetuple()))
    #feeddata.to_sql(name='feeddata', con=writeConn, index=True, if_exists='append', index_label='CSIsym')
    #print 'Saved', feedfile, 'to', dbPath

    if refreshSea:
        print 'refreshSea is on.. debug mode'
        #for sig
        idx=-2
        prevUpdateDate=pd.read_sql(\
                'select DISTINCT Date from futuresATRhist where Date < %s order by Date ASC' \
                % lastDate.strftime('%Y%m%d') , con=readConn).iloc[-1]
        prevUpdateDate = dt.strptime(str(prevUpdateDate.values.flatten()[-1]),'%Y%m%d')
        futuresDF_old=pd.read_sql( 'select * from futuresDF_all where timestamp=\
                (select max(timestamp) from futuresDF_all where Date=%s)' %prevUpdateDate.strftime('%Y%m%d'),\
                con=readConn,  index_col='CSIsym')
    else:
        idx=-1
        futuresDF_old=pd.read_csv(dataPath2+'futuresATR.csv', index_col=0)
        prevUpdateDate=dt.strptime(futuresDF_old.index.name,"%Y-%m-%d %H:%M:%S")

    #save last seasonal signal for pnl processing
    #update correl charts
    if lastDate >prevUpdateDate:
        #first time run needs to update pivot dates for runsystems.  
        print "First Run.. running seasonalClassifier"
        nextColOrder = ['0.75LastSIG','0.5LastSIG','1LastSIG','prevSEA','prevSRUN','prevvSTART']
        for i,contract in enumerate(marketList):
            print i+1,
            if 'YT' not in contract:
                sym = ''.join([i for i in contract.split('_')[0] if not i.isdigit()])
            else:
                sym=contract
            #seasonality
            seaBias, currRun, date,vStart = seasonalClassifier(sym, dataPath, savePath=pngPath+version+'_'+sym+'_MODE2',\
                                                        debug=showPlots)
            futuresDF.set_value(sym,'vSTART',vStart)
            futuresDF.set_value(sym,'LastSEA',seaBias)
            futuresDF.set_value(sym,'SEA'+str(date),seaBias)
            futuresDF.set_value(sym,'LastSRUN',currRun)
            futuresDF.set_value(sym,'SRUN'+str(date),currRun)
        futuresDF['prevACT']=futuresDF_old['prevACT']
        futuresDF['prevSEA']=futuresDF_old.LastSEA
        futuresDF['prevSRUN']=futuresDF_old.LastSRUN
        futuresDF['prevvSTART']=futuresDF_old.vSTART
        #corrDF.to_csv(savePath+'futuresPCcsv')
        #corrDF.corr().to_csv(savePath+'futuresCorr.csv')
        corrDF=corrDF.corr()
        fig,ax = plt.subplots(figsize=(15,13))
        ax.set_title('Correlation '+str(data.index[0])+' to '+str(data.index[-1]))
        sns.heatmap(ax=ax,data=corrDF)
        plt.yticks(rotation=0) 
        plt.xticks(rotation=90) 
        corrDF.to_html(savePath2+'futures_3.html')

        if pngPath != None:
            print 'Saving '+pngPath+'futures_3.png'
            fig.savefig(pngPath+'futures_3.png', bbox_inches='tight')
            
        if len(sys.argv)==1 and showPlots:
            #print data.index[0],'to',data.index[-1]
            plt.show()
        plt.close()

        for i,col in enumerate(corrDF):
            plt.figure(figsize=(8,10))
            corrDF[col].sort_values().plot.barh(color='r')
            plt.axvline(0, color='k')
            plt.title(col+' '+str(lookback)+' Day Correlation '+str(data.index[0])+' to '+str(data.index[-1]))
            plt.xlim(-1,1)
            plt.xticks(np.arange(-1,1.25,.25))
            plt.grid(True)
            filename=version+'_'+col+'_CORREL'+'.png'
            if pngPath != None:
                print i+1,'Saving '+pngPath+filename
                plt.savefig(pngPath+filename, bbox_inches='tight')
            
            if len(sys.argv)==1 and showPlots:
                #print data.index[0],'to',data.index[-1]
                plt.show()
            plt.close()
    else:
        print "Second Run.. skipping seasonalClassifier"
        #second time load  "old" file for seasonality signals.
        nextColOrder = ['0.75LastSIG','0.5LastSIG','1LastSIG','LastSEA','LastSRUN','vSTART']
        futuresDF['vSTART']=futuresDF_old.vSTART
        futuresDF['LastSEA']=futuresDF_old.LastSEA
        futuresDF['LastSRUN']=futuresDF_old.LastSRUN
        
        for col in futuresDF_old.columns:
            if col.startswith('SEA'):
                futuresDF[col]=futuresDF_old[col]
                
        for col in futuresDF_old.columns:
            if col.startswith('SRUN'):
                futuresDF[col]=futuresDF_old[col]


        
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
            if data.safef.iloc[idx] ==1:
                adjQty = int(round(futuresDF.ix[sym].QTY*(1+safefAdjustment)))
            else:
                adjQty = int(round(futuresDF.ix[sym].QTY*(1-safefAdjustment)))
            
        futuresDF.set_value(sym,'LastSIG',data.signals.iloc[idx])
        futuresDF.set_value(sym,'LastSAFEf',data.safef.iloc[idx])
        futuresDF.set_value(sym,'finalQTY',adjQty)
        futuresDF.set_value(sym,'SIG'+str(data.index[idx]),data.signals.iloc[idx])
       

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
            sigDate = data.index[idx]
        else:
            if data.index[idx]> sigDate:
                sigDate=data.index[idx]
                
        if sym in offline:
            adjQty=0
        else:
            if data.dpsSafef.iloc[idx] ==1:
                adjQty = int(round(futuresDF.ix[sym].QTY*(1+safefAdjustment)))
            else:
                adjQty = int(round(futuresDF.ix[sym].QTY*(1-safefAdjustment)))
                
        futuresDF.set_value(sym,'0.5LastSIG',data.signals.iloc[idx])
        futuresDF.set_value(sym,'0.5LastSAFEf',data.dpsSafef.iloc[idx])
        futuresDF.set_value(sym,'0.5finalQTY',adjQty)
        futuresDF.set_value(sym,'0.5SIG'+str(data.index[idx]),data.signals.iloc[idx])

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
            if data.dpsSafef.iloc[idx] ==1:
                adjQty = int(round(futuresDF.ix[sym].QTY*(1+safefAdjustment)))
            else:
                adjQty = int(round(futuresDF.ix[sym].QTY*(1-safefAdjustment)))
                
        futuresDF.set_value(sym,'0.75LastSIG',data.signals.iloc[idx])
        futuresDF.set_value(sym,'0.75LastSAFEf',data.dpsSafef.iloc[idx])
        futuresDF.set_value(sym,'0.75finalQTY',adjQty)
        futuresDF.set_value(sym,'0.75SIG'+str(data.index[idx]),data.signals.iloc[idx])

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
            if data.dpsSafef.iloc[idx] ==1:
                adjQty = int(round(futuresDF.ix[sym].QTY*(1+safefAdjustment)))
            else:
                adjQty = int(round(futuresDF.ix[sym].QTY*(1-safefAdjustment)))
                
        futuresDF.set_value(sym,'1LastSIG',data.signals.iloc[idx])
        futuresDF.set_value(sym,'1LastSAFEf',data.dpsSafef.iloc[idx])
        futuresDF.set_value(sym,'1finalQTY',adjQty)
        futuresDF.set_value(sym,'1SIG'+str(data.index[idx]),data.signals.iloc[idx])

    futuresDF=futuresDF.sort_index()
    columns = futuresDF.columns.tolist()
    futuresDF.LastPctChg=futuresDF[sorted([x for x in columns if 'PC' in x])[-1]].values
    start_idx =columns.index('ACT')+1
    new_order = columns[:start_idx]+nextColOrder
    new_order =  new_order+[x for x in columns if x not in new_order]
    futuresDF = futuresDF[new_order]
    print futuresDF.iloc[:,:4]



    #system file update
    system = pd.read_csv(systemPath+systemFilename)
    system_mini = pd.read_csv(systemPath+systemFilename2)
    system_micro = pd.read_csv(systemPath+systemFilename3)

    #macro
    for sys in system.System:
        sym=sys.split('_')[1]
        idx=system[system.System==sys].index[0]
        print 'MACRO', sys, sym, system.ix[idx].c2qty,
        system.set_value(idx,'c2qty',int(futuresDF.ix[sym]['finalQTY']))
        print system.ix[idx].c2qty, system.ix[idx].c2sym,
        system.set_value(idx,'c2sym',futuresDF.ix[sym]['Contract'])
        print system.ix[idx].c2sym
    system.Name=systemFilename.split('_')[1][:-4]
    system.c2id=c2id_macro
    system.c2api=c2key_macro
    #mini
    for sys in system_mini.System:
        sym=sys.split('_')[1]
        idx=system_mini[system_mini.System==sys].index[0]
        print 'MINI', sys, sym, system_mini.ix[idx].c2qty,
        if sym in offline_mini:
            system_mini.set_value(idx,'c2qty',0)
        else:
            system_mini.set_value(idx,'c2qty',int(futuresDF.ix[sym]['QTY_MINI']))
        print system_mini.ix[idx].c2qty, system_mini.ix[idx].c2sym,
        system_mini.set_value(idx,'c2sym',futuresDF.ix[sym]['Contract'])
        print system_mini.ix[idx].c2sym
    system_mini.Name=systemFilename2.split('_')[1][:-4]
    system_mini.c2id=c2id_mini
    system_mini.c2api=c2key_mini
    #micro
    for sys in system_micro.System:
        sym=sys.split('_')[1]
        idx=system_micro[system_micro.System==sys].index[0]
        print 'MICRO', sys, sym, system_micro.ix[idx].c2qty,
        if sym in offline_micro:
            system_micro.set_value(idx,'c2qty',0)
        else:
            system_micro.set_value(idx,'c2qty',int(futuresDF.ix[sym]['QTY_MICRO']))
        print system_micro.ix[idx].c2qty, system_micro.ix[idx].c2sym,
        system_micro.set_value(idx,'c2sym',futuresDF.ix[sym]['Contract'])
        print system_micro.ix[idx].c2sym
    system_micro.Name=systemFilename3.split('_')[1][:-4]
    system_micro.c2id=c2id_micro 
    system_micro.c2api=c2key_micro
    #signalDF=signalDF.sort_index()
    #print signalDF
    #signalDF.to_csv(savePath+'futuresSignals.csv')

    #for signal files
    #for system files

    #use LastSEA for seasonality in c2


        

    if lastDate > sigDate:
        AdjSEACols= ['RiskOn','AntiPrevACT','prevSEA']
        votingCols = ['Anti1LastSIG','prevACT','prevSEA']
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
        voting9Cols=['RiskOn','AntiPrevACT','AntiSEA']
        voting10Cols = ['RiskOn','0.5LastSIG','AntiSEA','AntiPrevACT']
        voting11Cols = ['RiskOn','Anti0.75LastSIG','AntiSEA','AntiPrevACT']
        voting12Cols = ['1LastSIG','AntiPrevACT','AdjSEA']
        voting13Cols = ['RiskOn','0.75LastSIG','AntiSEA','prevACT']
        voting14Cols = ['RiskOn','Anti1LastSIG','prevSEA','prevACT']
        voting15Cols = ['RiskOff','Anti0.75LastSIG','AntiSEA']
        
        #1bi. Run v4size(to update vlookback)
        #calc the previous day's results.
        nrows=futuresDF.shape[0]
        totalsDF = pd.DataFrame()
        futuresDF['None']=0
        futuresDF['RiskOff']=np.where(futuresDF.RiskOn<0,1,-1)
        futuresDF['AntiCustom']=np.where(futuresDF.Custom<0,1,-1)
        futuresDF['Anti1LastSIG'] = np.where(futuresDF['1LastSIG']==1,-1,1)
        futuresDF['Anti0.75LastSIG'] = np.where(futuresDF['0.75LastSIG']==1,-1,1)
        futuresDF['Anti0.5LastSIG'] = np.where(futuresDF['0.5LastSIG']==1,-1,1)
        futuresDF['AntiSEA'] = np.where(futuresDF.prevSEA==1,-1,1)
        futuresDF['AntiPrevACT'] = np.where(futuresDF.prevACT==1,-1,1)
        #futuresDF['AdjSEA'] = np.where(futuresDF.prevSRUN <0, futuresDF.prevSEA*-1, futuresDF.prevSEA)
        futuresDF['AdjSEA']=np.where(futuresDF[AdjSEACols].sum(axis=1)<0,-1,1)
        futuresDF['AntiAdjSEA'] = np.where(futuresDF.AdjSEA==1,-1,1)
        futuresDF['Voting']=np.where(futuresDF[votingCols].sum(axis=1)<0,-1,1)
        futuresDF['Voting2']=np.where(futuresDF[voting2Cols].sum(axis=1)<0,-1,1)
        futuresDF['Voting3']=np.where(futuresDF[voting3Cols].sum(axis=1)<0,-1,1)
        
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
        
        pctChgCol = sorted([x for x in columns if 'PC' in x])[-1]
        futuresDF['chgValue'] = futuresDF[pctChgCol]* futuresDF.contractValue*futuresDF.finalQTY
        cv_online = futuresDF['chgValue'].drop(offline,axis=0)
        signals = ['ACT','prevACT','AntiPrevACT','RiskOn','RiskOff','Custom','AntiCustom',\
                'LastSIG', '0.75LastSIG','0.5LastSIG','1LastSIG','Anti1LastSIG','Anti0.75LastSIG','Anti0.5LastSIG',\
                'prevSEA','LastSEA','AntiSEA','AdjSEA','AntiAdjSEA',\
                'Voting','Voting2','Voting3','Voting4','Voting5','Voting6','Voting7','Voting8','Voting9',\
                'Voting10','Voting11','Voting12','Voting13','Voting14','Voting15']
        for sig in signals:
            futuresDF['PNL_'+sig]=futuresDF['chgValue']*futuresDF[sig]
            totalsDF.set_value(lastDate, 'ACC_'+sig, sum(futuresDF[sig]==futuresDF.ACT)/float(nrows))
            totalsDF.set_value(lastDate, 'L%_'+sig, sum(futuresDF[sig]==1)/float(nrows))
        totals =futuresDF[[x for x in futuresDF if 'PNL' in x]].sum()
        for i,value in enumerate(totals):
            totalsDF.set_value(lastDate, totals.index[i], value)
            
        bygroup = pd.concat([abs(futuresDF['chgValue']), futuresDF['group']],axis=1).drop(offline, axis=0).groupby(['group'])
        volByGroupByContract = bygroup.sum()/bygroup.count()
        bygroup2 = pd.concat([futuresDF['chgValue'], futuresDF['group']],axis=1).drop(offline, axis=0).groupby(['group'])
        chgByGroupByContract = bygroup2.sum()/bygroup2.count()
        bygroup3 = pd.concat([futuresDF['ACT']==1, futuresDF['group']],axis=1).drop(offline, axis=0).groupby(['group'])
        longPerByGroup = bygroup3.sum()/bygroup3.count()
        
        if cv_online.count() >0:
            totalsDF.set_value(lastDate, 'Vol_All', abs(cv_online).sum()/cv_online.count())
            totalsDF.set_value(lastDate, 'Chg_All', cv_online.sum()/cv_online.count())
            totalsDF.set_value(lastDate, 'L%_All', sum(futuresDF.ACT.drop(offline, axis=0)==1)/float(cv_online.count()))
        else:
            totalsDF.set_value(lastDate, 'Vol_All', np.nan)
            totalsDF.set_value(lastDate, 'Chg_All', np.nan)
            totalsDF.set_value(lastDate, 'L%_All', np.nan)
        
        for i,value in enumerate(volByGroupByContract['chgValue']):
            totalsDF.set_value(lastDate, 'Vol_'+volByGroupByContract.index[i], value)
        
        
        for i,value in enumerate(chgByGroupByContract['chgValue']):
            totalsDF.set_value(lastDate, 'Chg_'+chgByGroupByContract.index[i], value)
        
       
        for i,value in enumerate(longPerByGroup['ACT']):
            totalsDF.set_value(lastDate, 'L%_'+longPerByGroup.index[i], value)
        
        print totalsDF.sort_index().transpose()
        
        filename='futuresResults_'+lastDate.strftime("%Y%m%d%H%M")+'.csv'
        print 'Saving', savePath2+filename
        totalsDF.sort_index().transpose().to_csv(savePath2+filename)
        
        filename='futuresResults_Last.csv'
        print 'Saving', savePath+filename
        
        totalsDF.sort_index().transpose().to_csv(savePath+filename)
        totalsDF['Date']=int(lastDate.strftime('%Y%m%d'))
        totalsDF['timestamp']=int(calendar.timegm(dt.utcnow().utctimetuple()))
        totalsDF.to_sql(name='totalsDF',con=writeConn, index=False, if_exists='append')
        print 'Saved totalsDF to', dbPath
        
        files = [ f for f in listdir(savePath) if isfile(join(savePath,f)) ]
        filename = 'futuresResultsHistory.csv'
        if filename not in files:
            print 'Saving', savePath+filename
            totalsDF.to_csv(savePath+filename)
        else:
            print 'Saving', savePath+filename
            pd.read_csv(savePath+filename, index_col=0).append(totalsDF).to_csv(savePath+filename)
            
        filename='futuresATR_'+lastDate.strftime("%Y%m%d%H%M")+'.csv'
        print 'Saving', savePath2+filename
        futuresDF.to_csv(savePath2+filename)
        print 'Saving', savePath+'futuresATR_Results.csv'
        futuresDF.to_csv(savePath+'futuresATR_Results.csv')
        futuresDF['Date']=int(lastDate.strftime('%Y%m%d'))
        futuresDF['timestamp']=int(calendar.timegm(dt.utcnow().utctimetuple()))
        futuresDF.drop([col for col in futuresDF.columns if '00:00:00' in col], axis=1).to_sql(name='futuresDF_results',\
                                con=writeConn, index=True, if_exists='append', index_label='CSIsym')
                                
        print 'Saved futuresDF_results to', dbPath
        #futuresDF_live = pd.read_sql('select * from futuresATRhist where timestamp=\
        #        (select max(timestamp) from futuresATRhist where Date=%s)' %prevUpdateDate.strftime('%Y%m%d'),\
        #        con=readConn,  index_col='CSIsym')
        futuresDF_live = pd.read_sql('select * from (select * from futuresATRhist where Date=%s\
                            order by timestamp ASC) group by CSIsym' %prevUpdateDate.strftime('%Y%m%d'),\
                            con=readConn,  index_col='CSIsym')
        
        futuresDF_toexcel=pd.concat([futuresDF_live, futuresDF.drop(futuresDF_live.index,axis=0)],axis=0).sort_index()
        futuresDF_toexcel.columns=[x if x !='Date' else 'signalDate' for x in futuresDF_toexcel.columns]
        
        #update old act and pct change to csi values
        futuresDF_toexcel['ACT_IB']=futuresDF.ACT
        futuresDF_toexcel['LastPctChg_IB']=futuresDF.LastPctChg
        #overwrite CSI's values with ones from IB.
        ACT_IB=pd.read_sql('select CSIsym, ACT from futuresATR', con=readConn, index_col='CSIsym')
        for sym in ACT_IB.index:
            futuresDF_toexcel.set_value(sym, 'ACT_IB', ACT_IB.ix[sym][0])
        LastPctChg_IB=pd.read_sql('select CSIsym, LastPctChg from futuresATR', con=readConn, index_col='CSIsym')
        for sym in LastPctChg_IB.index:
            futuresDF_toexcel.set_value(sym, 'LastPctChg_IB', LastPctChg_IB.ix[sym][0])
        futuresDF_toexcel['pcDate']=pd.read_sql('select CSIsym, Date from futuresATR', con=readConn, index_col='CSIsym')
        futuresDF_toexcel['ACT']=futuresDF.ACT
        futuresDF_toexcel['LastPctChg']=futuresDF.LastPctChg
        pc_cols=[x for x in futuresDF.columns if 'PC' in x]
        futuresDF_toexcel[pc_cols]=futuresDF[pc_cols]
        #seasonality data same as before
        futuresDF_toexcel['prevSEA']=futuresDF.LastSEA
        futuresDF_toexcel['prevSRUN']=futuresDF.LastSRUN
        futuresDF_toexcel['prevvSTART']=futuresDF.prevvSTART
        
        cols =[x for x in futuresDF.columns if x in futuresDF_toexcel.columns]+['LastPctChg_IB','ACT_IB','pcDate','signalDate','timestamp']
        futuresDF_toexcel=futuresDF_toexcel[cols]
        #recreated new price change data with previous signals from csi (offline) ib(online)
        futuresDF_toexcel.to_csv(savePath+'futuresATR_Excel.csv')
        print 'Saved', savePath+'futuresATR_Excel.csv'
        
        filename='futuresL_History.csv'
        #cols=['L%_currency',
        #         'L%_energy',
        #         'L%_grain',
        #         'L%_index',
        #         'L%_meat',
        #         'L%_metal',
        #         'L%_ACT',
        #         'L%_rates',
        #         'L%_soft']
        cols = [x for x in totalsDF if 'L%' in x]
        if filename not in files:
            print 'Saving', savePath+filename
            totalsDF[cols].to_csv(savePath+filename)
        else:
            print 'Saving', savePath+filename
            pd.read_csv(savePath+filename, index_col=0).append(totalsDF[cols]).to_csv(savePath+filename)
            
    else:
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
            
            if debug:
                plt.show()
            plt.close()
        
        print 'Saving signals from', c2system
        #1biv. Run v4size (signals and size) and check system.csv for qty,contracts with futuresATR
        #save signals to v4_ signal files for order processing

        nsig=0
        for ticker in futuresDF.index:
            nsig+=1
            signalFile=pd.read_csv(signalSavePath+ version+'_'+ ticker+ '.csv', index_col=['dates'])
            #addLine = signalFile.iloc[-1]
            #addLine = addLine.append(pd.Series(data=timenow.strftime("%Y%m%d %H:%M:%S %Z"), index=['timestamp']))
            #addLine = addLine.append(pd.Series(data=cycleTime, index=['cycleTime']))
            #addLine.name = sst.iloc[-1].name
            addLine = pd.Series(name=lastDate)
            addLine['signals']=futuresDF.ix[ticker][c2system]
            addLine['safef']=c2safef
            addLine['timestamp']=dt.now().strftime("%Y%m%d %H:%M:%S %Z")
            signalFile = signalFile.append(addLine)
            filename=signalSavePath + version+'_'+ ticker+ '.csv'
            print 'Saving...',  addLine['signals'], addLine['safef'], filename
            signalFile.to_csv(filename, index=True)
        print nsig, 'files updated'
            
        print 'Saving', savePath+'futuresATR_Signals.csv'
        futuresDF.to_csv(savePath+'futuresATR_Signals.csv')
        futuresDF['Date']=int(lastDate.strftime('%Y%m%d'))
        futuresDF['timestamp']=int(calendar.timegm(dt.utcnow().utctimetuple()))
        futuresDF.drop([col for col in futuresDF.columns if '00:00:00' in col], axis=1).to_sql(name='futuresDF_all',\
                                con=writeConn, index=True, if_exists='append', index_label='CSIsym')
        print 'Saved futuresDF_all to', dbPath
        
        for i,sym in enumerate([x.split('_')[1] for x in system.System]):
            system.set_value(i,'signal',futuresDF[c2system].ix[sym])
        system.to_csv(systemPath+systemFilename, index=False)
        print 'Saved', systemPath+systemFilename,c2system
        system['Date']=int(lastDate.strftime('%Y%m%d'))
        system['timestamp']=int(calendar.timegm(dt.utcnow().utctimetuple()))
        tablename = 'v4futures'
        system.to_sql(name=tablename, if_exists='replace', con=writeConn, index=False)
        system.to_sql(name='signals_csi', if_exists='append', con=writeConn, index=False)
        print 'Saved to sql db',  tablename,c2system
        
        for i,sym in enumerate([x.split('_')[1] for x in system_mini.System]):
            system_mini.set_value(i,'signal',futuresDF[c2system_mini].ix[sym])
        system_mini.to_csv(systemPath+systemFilename2, index=False)
        print 'Saved', systemPath+systemFilename2,c2system_mini
        system_mini['Date']=int(lastDate.strftime('%Y%m%d'))
        system_mini['timestamp']=int(calendar.timegm(dt.utcnow().utctimetuple()))
        tablename = 'v4mini'
        system_mini.to_sql(name=tablename, if_exists='replace', con=writeConn, index=False)
        system_mini.to_sql(name='signals_csi', if_exists='append', con=writeConn, index=False)
        print 'Saved to sql db',  tablename,c2system_mini

        for i,sym in enumerate([x.split('_')[1] for x in system_micro.System]):
            system_micro.set_value(i,'signal',futuresDF[c2system_micro].ix[sym])
        system_micro.to_csv(systemPath+systemFilename3, index=False)
        print 'Saved', systemPath+systemFilename3, c2system_micro
        system_micro['Date']=int(lastDate.strftime('%Y%m%d'))
        system_micro['timestamp']=int(calendar.timegm(dt.utcnow().utctimetuple()))
        tablename = 'v4micro'
        system_micro.to_sql(name=tablename, if_exists='replace', con=writeConn, index=False)
        system_micro.to_sql(name='signals_csi', if_exists='append', con=writeConn, index=False)
        print 'Saved to sql db',  tablename,c2system_micro
        
    futuresDF.to_csv(savePath+'futuresATR.csv')
    print 'Saved', savePath+'futuresATR.csv'


    accountInfo['Date']=int(lastDate.strftime('%Y%m%d'))
    accountInfo['timestamp']=int(calendar.timegm(dt.utcnow().utctimetuple()))
    accountInfo.to_sql(name='accountInfo',con=writeConn, index=True, if_exists='append')
    print 'Saved accountInfo to', dbPath
    print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes ', dt.now()
except Exception as e:
    logging.exception("message")
    logging.info( str(dt.now()))