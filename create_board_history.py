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
import re
import datetime
from datetime import datetime as dt
import time
import os
import os.path
import sys
import logging
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from pytz import timezone
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, WeekdayLocator,DayLocator,MO, TU, WE, TH, FR, SA, SU,\
                                            MonthLocator, MONDAY, HourLocator, date2num

start_time = time.time()

    
def fixTypes(original, transformed):
    for x in original.index:
        #print x, type(series[x]),
        transformed[x]=transformed[x].astype(type(original[x]))
    return transformed
    
def is_int(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False
        
def to_signals(df, Anti=False):
    df2=df.copy()
    if Anti:
        df2[df>0]=-1
        df2[df<0]=1
    else:
        df2[df>0]=1
        df2[df<0]=-1
    return df2
    
def checkTableExists(dbconn, tablename):
    dbcur = dbconn.cursor()
    dbcur.execute("""
        SELECT COUNT(*)
        FROM sqlite_master
        WHERE type= 'table' AND name = '{0}'
        """.format(tablename.replace('\'', '\'\'')))
    if dbcur.fetchone()[0] == 1:
        dbcur.close()
        return True

    dbcur.close()
    return False
    

corecomponents =[
                'RiskOn',
                'RiskOff',
                'LastSEA',
                'AntiSEA',
                'prevACT',
                'AntiPrevACT',
                '0.75LastSIG',
                '0.5LastSIG',
                '1LastSIG',
                'Anti1LastSIG',
                'Anti0.75LastSIG',
                'Anti0.5LastSIG',
                'Custom',
                'AntiCustom',
                'None',
                ]
                
reversecomponentsdict ={
                'None':'Off',
                'prevACT':'Previous',
                'AntiPrevACT':'Anti-Previous',
                'RiskOn':'RiskOn',
                'RiskOff':'RiskOff',
                'Custom':'Custom',
                'AntiCustom':'Anti-Custom',
                '0.75LastSIG':'50/50',
                '0.5LastSIG':'LowestEquity',
                '1LastSIG':'HighestEquity',
                'Anti1LastSIG':'AntiHighestEquity',
                'Anti0.75LastSIG':'Anti50/50',
                'Anti0.5LastSIG':'AntiLowestEquity',
                'LastSEA':'Seasonality',
                'AntiSEA':'Anti-Seasonality',
                'none':'none',
                }
                
componentpairs =[
                ['Previous','Anti-Previous'],
                ['RiskOn','RiskOff'],
                ['Custom','Anti-Custom'],
                ['50/50','Anti50/50'],
                ['LowestEquity','AntiLowestEquity'],
                ['HighestEquity','AntiHighestEquity'],
                ['Seasonality','Anti-Seasonality'],
                ]

component_text={'Previous':'Previous trading day\'s signals. For example if gold went up the previous day, the signal would be LONG. ','Anti-Previous':'Opposite of Previous signals. For example if Gold went down the previous day, signal will be LONG.','RiskOn':'Fixed Signals consisting of Short precious metals and bonds, Long all other risky assets','RiskOff':'Opposite of RiskOn signals. (Fixed Signals consisting of Long precious metals and bonds, Short all other risky assets)','Custom':'Custom signals provided by the player.','Anti-Custom':'Opposite of Custom signals provided by the player.','50/50':'Combination of signals from HighestEquity and LowestEquity.','Anti50/50':'Opposite of 50/50 signals.','LowestEquity':'Baysean machine learning system prioritizing signals from worst performing systems.','AntiLowestEquity':'Opposite of LowestEquity signals.','HighestEquity':'Baysean machine learning system prioritizing signals from best performing systems.','AntiHighestEquity':'Opposite of HighestEquity signals.','Seasonality':'Signals computed from 10 to 30+ years of seasonal daily data.','Anti-Seasonality':'Opposite of Seasonality signals.',}
anti_components={'Previous':'Anti-Previous','Anti-Previous':'Previous','RiskOn':'RiskOff','RiskOff':'RiskOn','Custom':'Anti-Custom','Anti-Custom':'Custom','50/50':'Anti50/50','Anti50/50':'50/50','LowestEquity':'AntiLowestEquity','AntiLowestEquity':'LowestEquity','HighestEquity':'AntiHighestEquity','AntiHighestEquity':'HighestEquity','Seasonality':'Anti-Seasonality','Anti-Seasonality':'Seasonality',}

keep_cols = ['Contract', 'ACT','LastPctChg','contractValue','group', 'Date', 'timestamp']
qtydict={'v4futures':'QTY','v4mini':'QTY_MINI','v4micro':'QTY_MICRO',}

#maybe replace these with true account values later
accountvalues={'v4futures':250000,'v4mini':100000,'v4micro':50000,}
web_accountnames={
                    'v4futures':'250K',
                    'v4mini':'100K',
                    'v4micro':'50K',
                    }
lookback_short=1
lookback_mid=2
lookback=20
benchmark_sym='ES'
if len(sys.argv)==1:
    debug=True
else:
    debug=False
    
if debug:
    mode = 'replace'
    savePlots=False
    #marketList=[sys.argv[1]]
    showPlots=False
    dbPath='./data/futures.sqlite3' 
    dbPath2='./data/futures.sqlite3' 
    dbPathWeb = './web/tsdp/db.sqlite3'
    dbPathWebCharts = './web/tsdp/db_charts.sqlite3'
    dataPath='./data/csidata/v4futures2/'
    savePath= './data/results/' 
    jsonPath ='./web/tsdp/'
    pngPath = './data/results/' 
    feedfile='./data/systems/system_ibfeed.csv'
    #test last>old
    #dataPath2=pngPath
    #signalPath = './data/signals/' 
    
    #test last=old
    dataPath2='./data/'
    lastquotePath='./data/csidata/v4futures_last/'
    #signalPath ='D:/ML-TSDP/data/signals2/'
    signalPath ='./signals2/' 
    signalSavePath = './data/signals/' 
    systemPath = './data/systems/' 
    readConn = sqlite3.connect(dbPath2)
    writeConn= sqlite3.connect(dbPath)
    #readWebConn = sqlite3.connect(dbPathWeb)
    #logging.basicConfig(filename='C:/logs/vol_adjsize_live_func_error.log',level=logging.DEBUG)
else:
    mode= 'replace'
    savePlots=True
    #marketList=[sys.argv[1]]
    showPlots=False
    feedfile='./data/systems/system_ibfeed.csv'
    dbPath='./data/futures.sqlite3'
    dbPathWeb ='./web/tsdp/db.sqlite3'
    dbPathWebCharts = './web/tsdp/db_charts.sqlite3'
    jsonPath ='./web/tsdp/'
    dataPath='./data/csidata/v4futures2/'
    #dataPath='./data/csidata/v4futures2/'
    dataPath2='./data/'
    savePath='./data/results/'
    signalPath = './data/signals2/' 
    signalSavePath = './data/signals2/' 
    pngPath = './web/tsdp/betting/static/public/images/'
    systemPath =  './data/systems/'
    lastquotePath='./data/csidata/v4futures_last/'
    readConn = writeConn= sqlite3.connect(dbPath)
    #readWebConn = sqlite3.connect(dbPathWeb)
    #logging.basicConfig(filename='/logs/vol_adjsize_live_func_error.log',level=logging.DEBUG)
    
readWebConn = sqlite3.connect(dbPathWeb)
writeWebChartsConn = sqlite3.connect(dbPathWebCharts)

filename=jsonPath+'accountinfo_data.json'
with open(filename, 'r') as f:
     accountinfo=json.load(f)

active_symbols={}
for account in accountinfo.keys():
    active_symbols[account]=eval(accountinfo[account]['online'])
    
'''
active_symbols={
                        'v4futures':['AD', 'BO', 'BP', 'C', 'CD', 'CL', 'CU', 'EMD', 'ES', 'FC',
                                           'FV', 'GC', 'HG', 'HO', 'JY', 'LC', 'LH', 'MP', 'NE', 'NG',
                                           'NIY', 'NQ', 'PA', 'PL', 'RB', 'S', 'SF', 'SI', 'SM', 'TU',
                                           'TY', 'US', 'W', 'YM'],
                        'v4mini':['C', 'CL', 'CU', 'EMD', 'ES', 'HG', 'JY', 'NG', 'SM', 'TU', 'TY', 'W'],
                        'v4micro':['BO', 'ES', 'HG', 'NG', 'TY'],
                        }
'''
all_syms=active_symbols['v4futures']
futuresDict = pd.read_sql('select * from Dictionary', con=readConn, index_col='CSIsym')
selectionDF=pd.read_sql('select * from betting_userselection where timestamp=\
        (select max(timestamp) from betting_userselection as maxtimestamp)', con=readWebConn, index_col='userID')
#selectionDict=eval(selectionDF.selection.values[0])

#futuresDF_all=pd.read_csv(dataPath2+'futuresATR_Signals.csv', index_col=0)
dates= pd.read_sql('select distinct Date from futuresATRhist', con=readConn).Date.tolist()
dates_csi= pd.read_sql('select distinct Date from futuresDF_all', con=readConn).Date.tolist()

#this is created after every MOC
#datetup=[(dates[i],dates[i+1]) for i,x in enumerate(dates[:-1])][-lookback:]
#datetup_csi=[(dates_csi[i],dates_csi[i+1]) for i,x in enumerate(dates_csi[:-1])][-lookback:]
missing_dates=list(set(dates_csi) -set(dates))
dates+=missing_dates
dates.sort()
date_loc=[(x, 'futuresATRhist') if x not in missing_dates else (x, 'futuresDF_all') for x in dates]
datetup=[(date_loc[i],date_loc[i+1]) for i,x in enumerate(date_loc[:-1])][-lookback:]

def add_missing_rows(df, datetup, all_syms):
    global date_loc
    global readConn
    
    totalnum_sym=len(all_syms)
    if df.shape[0]<totalnum_sym:
        missing_syms=[x for x in all_syms if x not in df.index]
        prev=date_loc[date_loc.index(datetup)-1]
        while len(missing_syms)>0:
            futuresDF_prev2=pd.read_sql('select * from (select * from %s where Date=%s\
                    order by timestamp ASC) group by CSIsym' % (prev[1], prev[0]),\
                    con=readConn,  index_col='CSIsym')
            missing_rows=futuresDF_prev2.ix[[x for x in missing_syms if x in futuresDF_prev2.index]].copy()
            missing_rows.LastPctChg=0
            missing_rows.ACT=0
            missing_rows.Date=int(datetup[0])
            df=pd.concat([df, missing_rows], axis=0)
            print 'Added',missing_syms
            prev=date_loc[date_loc.index(datetup)-1]
            missing_syms=[x for x in missing_syms if x not in df.index]
        return df.ix[all_syms]
    else:
        return df.ix[all_syms]
            
totals_accounts={}
pnl_accounts={}
boards_dict={}
for account in qtydict.keys():
    print '\ncreating history for', account
    componentsdict = eval(selectionDF[account].values[0])
    futuresDF_boards ={}
    signalsDict={}
    signalsDict2={}
    totalsDict = {}
    for prev,current in datetup:
        currentdate=current[0]
        print currentdate,
        futuresDF_prev=add_missing_rows(pd.read_sql('select * from (select * from %s where Date=%s\
                        order by timestamp ASC) group by CSIsym' %(prev[1], prev[0]),\
                        con=readConn,  index_col='CSIsym'), prev, all_syms)
        
        futuresDF_current=add_missing_rows(pd.read_sql('select * from (select * from %s where Date=%s\
                        order by timestamp ASC) group by CSIsym' %(current[1], current[0]),\
                        con=readConn,  index_col='CSIsym'), current, all_syms)



        componentsignals=futuresDF_prev[corecomponents]

        votingSystems = { key: componentsdict[key] for key in [x for x in componentsdict if is_int(x)] }
        #add voting systems
        signalsDict[currentdate]={key: to_signals(futuresDF_prev[componentsdict[key]].sum(axis=1)) for key in votingSystems.keys()}
        #add anti-voting systems
        signalsDict[currentdate].update({'Anti-'+key: to_signals(futuresDF_prev[componentsdict[key]].sum(axis=1), Anti=True)\
                                                for key in votingSystems.keys()})
        #check (signalsDict[key]['1']+signalsDict[key]['Anti-1']).sum()
        signalsDict[currentdate].update({ reversecomponentsdict[key]: componentsignals[key] for key in componentsignals})
        
        #add benchmark
        benchmark_signals=futuresDF_prev['None'].copy()
        benchmark_signals.ix[benchmark_sym]=1
        signalsDict[currentdate]['benchmark']=benchmark_signals
        
        
        '''hotfix for signal display'''
        componentsignals2=futuresDF_current[corecomponents]

        #votingSystems = { key: componentsdict[key] for key in [x for x in componentsdict if is_int(x)] }
        #add voting systems
        signalsDict2[currentdate]={key: to_signals(futuresDF_current[componentsdict[key]].sum(axis=1)) for key in votingSystems.keys()}
        #add anti-voting systems
        signalsDict2[currentdate].update({'Anti-'+key: to_signals(futuresDF_current[componentsdict[key]].sum(axis=1), Anti=True)\
                                                for key in votingSystems.keys()})
        #check (signalsDict[key]['1']+signalsDict[key]['Anti-1']).sum()
        signalsDict2[currentdate].update({ reversecomponentsdict[key]: componentsignals2[key] for key in componentsignals2})

        signalsDict2[currentdate]['benchmark']=benchmark_signals
                
        
        #append signals to each board
        totalsDict[currentdate]=pd.DataFrame()
        futuresDF_boards[currentdate] =  futuresDF_current[keep_cols+[qtydict[account]]].copy()
        nrows=futuresDF_boards[currentdate].shape[0]
        #zero out quantities for offlien symbols
        quantity=futuresDF_boards[currentdate][qtydict[account]].copy()
        quantity.ix[[sym for sym in quantity.index if sym not in active_symbols[account]]]=0
        futuresDF_boards[currentdate]['chgValue'] =  futuresDF_boards[currentdate].LastPctChg*\
                                                                    futuresDF_boards[currentdate].contractValue*\
                                                                    quantity
        futuresDF_boards[currentdate]['abs_chgValue'] =abs(futuresDF_boards[currentdate]['chgValue'])
        for col in signalsDict[currentdate]:
            signalsDict[currentdate][col].name = col
            futuresDF_boards[currentdate]=futuresDF_boards[currentdate].join(signalsDict[currentdate][col])
            futuresDF_boards[currentdate]['PNL_'+col]=futuresDF_boards[currentdate][col]*futuresDF_boards[currentdate]['chgValue']
            #benchmarked to sym 1x leverage of account value
            if col=='benchmark':
                futuresDF_boards[currentdate].set_value(benchmark_sym,'PNL_benchmark',\
                                        futuresDF_boards[currentdate].ix[benchmark_sym].LastPctChg*accountvalues[account])
            totalsDict[currentdate].set_value(currentdate, 'ACC_'+col, sum(futuresDF_boards[currentdate][col]==futuresDF_boards[currentdate].ACT)/float(nrows))
            totalsDict[currentdate].set_value(currentdate, 'L%_'+col, sum(futuresDF_boards[currentdate][col]==1)/float(nrows))

        totals =futuresDF_boards[currentdate][[x for x in futuresDF_boards[currentdate] if 'PNL' in x]].sum()
        for i,value in enumerate(totals):
            totalsDict[currentdate].set_value(currentdate, totals.index[i], value)
        
        #change in value
        chgValuegroup = futuresDF_boards[currentdate].groupby(['group']).chgValue
        avg_chg_by_group = chgValuegroup.sum()/chgValuegroup.count()        
        chg_total = futuresDF_boards[currentdate]['chgValue'].sum()
        avg_chg_total = chg_total/nrows
        for i,value in enumerate(avg_chg_by_group):
            #print currentdate, 'Vol_'+avg_chg_by_group.index[i], value
            totalsDict[currentdate].set_value(currentdate, 'Chg_'+avg_chg_by_group.index[i], value)
        totalsDict[currentdate].set_value(currentdate, 'Chg_Total', chg_total)
        totalsDict[currentdate].set_value(currentdate, 'Chg_Avg', avg_chg_total)
        
        #change in volatility
        abschgValuegroup = futuresDF_boards[currentdate].groupby(['group']).abs_chgValue
        avg_vol_by_group = abschgValuegroup.sum()/abschgValuegroup.count()
        vol_total = futuresDF_boards[currentdate]['abs_chgValue'].sum()
        avg_vol_total = vol_total/nrows
        for i,value in enumerate(avg_vol_by_group):
            #print currentdate, 'Vol_'+avg_vol_by_group.index[i], value
            totalsDict[currentdate].set_value(currentdate, 'Vol_'+avg_vol_by_group.index[i], value)
        totalsDict[currentdate].set_value(currentdate, 'Vol_Total', vol_total)
        totalsDict[currentdate].set_value(currentdate, 'Vol_Avg', avg_vol_total)
        
        #change in long percent
        long_percent_by_group = pd.concat([futuresDF_boards[currentdate]['ACT']==1, futuresDF_boards[currentdate]['group']],axis=1).groupby(['group'])
        longPerByGroup =long_percent_by_group.sum()/long_percent_by_group.count()
        longPerByGroup_all=(futuresDF_boards[currentdate]['ACT']==1).sum()/float(nrows)
        for i in longPerByGroup.index:
            #print currentdate, 'L%_'+i, longPerByGroup.ix[i][0]
            value = longPerByGroup.ix[i][0]
            totalsDict[currentdate].set_value(currentdate, 'L%_'+i, value)
        totalsDict[currentdate].set_value(currentdate, 'L%_Total', longPerByGroup_all)

        #print totalsDict[currentdate].sort_index().transpose()
        #totalsDict[currentdate]['Date']=currentdate
        totalsDict[currentdate]['timestamp']=futuresDF_boards[currentdate].timestamp[0]
        
    
    totalsDF=pd.DataFrame()
    for key in totalsDict.keys():
        totalsDF=totalsDF.append(totalsDict[key])
    #dropna for thanksgiving
    totalsDF=totalsDF.sort_index().dropna()
    totals_accounts[account]=totalsDF
    tablename = 'totalsDF_board_'+account
    totalsDF.to_sql(name=tablename,con=writeConn, index=True, if_exists=mode, index_label='Date')
    print '\nSaved', tablename,'from',datetup[0][1],'to',currentdate,'to', dbPath

    pnlDF=pd.DataFrame()
    for key in futuresDF_boards.keys():
        pnlDF=pnlDF.append(futuresDF_boards[key].set_index('Date'))
    #dropna for thanksgiving
    pnlDF=pnlDF.sort_index().dropna()
    tablename = 'PNL_board_'+account
    pnl_accounts[account]=pnlDF
    pnlDF.to_sql(name= tablename, if_exists=mode, con=writeConn, index=True, index_label='Date')
    filename = savePath+tablename+'_'+str(currentdate)+'.csv'
    pnlDF.to_csv(filename, index=True)
    print 'Saved', tablename,'from',datetup[0][1],'to',currentdate,'to', dbPath,'and', filename
    boards_dict[account]=futuresDF_boards.copy()

#for customize signals
signalsDF=pd.DataFrame(signalsDict[currentdate])
signalsDF['Date']=currentdate
tablename='last_signals'
signalsDF.to_sql(name=tablename, if_exists=mode, con=writeWebChartsConn, index=True, index_label='CSIsym')
print 'Saved', tablename, 'for', currentdate

#last quote df
IB2CSI_multiplier_adj={
    'HG2':100,
    'SI2':100,
    'JY':100,
    }
lastquotes=pd.DataFrame()
futuresDict2=futuresDict.ix[futuresDF_current.index].reset_index().set_index('Filename')
for sym in futuresDict2.index:
    lastquote=pd.read_csv(lastquotePath+sym+'.csv').iloc[-1]
    lastquote.name=futuresDict2.ix[sym].CSIsym
    if sym in IB2CSI_multiplier_adj.keys():
        lastquote2=IB2CSI_multiplier_adj[sym]*lastquote[['Open','High','Low','Close']]
        lastquote2['Date']=lastquote['Date']
        lastquote2['Volume']=lastquote['Volume']
        lastquotes=lastquotes.append(lastquote2)
    else:
        lastquotes=lastquotes.append(lastquote)
        
lastquotes['LastDate']=current[0]
lastquotes['Last']=futuresDF_current.LastClose
lastquotes['Chg']=(lastquotes.Close-lastquotes.Last)/lastquotes.Last
lastquotes['PNL']=lastquotes.Chg*futuresDF_current.contractValue

#for customize chip
market_pnl_by_date=boards_dict['v4futures']
mpbd={}
for key in market_pnl_by_date:
    for sym in market_pnl_by_date[key].index:
        if sym not in mpbd:
            mpbd[sym]=pd.DataFrame({key:market_pnl_by_date[key].ix[sym]})
        else:
            mpbd[sym]=pd.concat([mpbd[sym], pd.DataFrame({key:market_pnl_by_date[key].ix[sym]})], axis=1)
mpbd2={}
for sym in mpbd:
    pnlcols=[x for x in mpbd[sym].index if x.split('_')[0]=='PNL']
    mpbd2[sym]=mpbd[sym].transpose()[pnlcols].cumsum()
    mpbd2[sym].index=[dt.strptime(str(x),'%Y%m%d') for x in mpbd2[sym].index]
    #mpbd2[sym].plot(title=sym)
    
#create charts
def conv_sig(signals):
    sig = signals.copy()
    #sig[sig < 0] = 'SHORT'
    #sig[sig == 1] = 'LONG'
    longs=sig[sig < 0].index
    shorts=sig[sig > 0].index
    off=sig[sig == 0].index
    sig.ix[longs]=['Short '+str(signals.ix[x]) for x in longs]
    sig.ix[shorts]=['Long '+str(signals.ix[x]) for x in shorts]
    sig.ix[off] = 'Off 0'
    return sig.values

performance_dict={}
infodisplay = {key: [reversecomponentsdict[x] for x in componentsdict[key]] for key in componentsdict}

perchgDict={}
#perchgDict_short={}
for account in totals_accounts:
    totalsDF=totals_accounts[account]
    pnl_cols=[x for x in totalsDF.columns if 'PNL' in x]
    pnlsDF=totalsDF[pnl_cols].copy()
    perchgDF=pd.DataFrame()
    for col in pnlsDF:
        pnlarr=pnlsDF[col].copy().values
        pnlarr[0]=pnlarr[0]+accountvalues[account]
        cumper=(pnlarr.cumsum()/accountvalues[account]-1)*100
        perchgDF=perchgDF.append(pd.Series(data=cumper, name=col.split('_')[1], index=pnlsDF.index))
    ranking=perchgDF.transpose().iloc[-1].sort_values(ascending=True)
    ranking.name=str(lookback)+'Day Lookback'

    pnlsDF_mid=pnlsDF.iloc[-lookback_mid:]
    perchgDF_mid=pd.DataFrame()
    for col in pnlsDF_mid:
        pnlarr=pnlsDF_mid[col].copy().values
        pnlarr[0]=pnlarr[0]+accountvalues[account]
        cumper=(pnlarr.cumsum()/accountvalues[account]-1)*100
        perchgDF_mid=perchgDF_mid.append(pd.Series(data=cumper, name=col.split('_')[1], index=pnlsDF_mid.index))
    ranking_mid=perchgDF_mid.transpose().iloc[-1].sort_values(ascending=True)
    ranking_mid.name=str(lookback_mid)+'Day Lookback'
    
    pnlsDF_short=pnlsDF.iloc[-lookback_short:]
    perchgDF_short=pd.DataFrame()
    for col in pnlsDF_short:
        pnlarr=pnlsDF_short[col].copy().values
        pnlarr[0]=pnlarr[0]+accountvalues[account]
        cumper=(pnlarr.cumsum()/accountvalues[account]-1)*100
        perchgDF_short=perchgDF_short.append(pd.Series(data=cumper, name=col.split('_')[1], index=pnlsDF_short.index))
    ranking_short=perchgDF_short.transpose().iloc[-1].sort_values(ascending=True)
    ranking_short.name=str(lookback_short)+'Day Lookback'
    #perchgDict_short[account]=ranking_short.copy()
    #perchgDict_short[account].index=[str(len(ranking_short.index)-idx)+' Rank '+col for idx,col in enumerate(ranking_short.index)]
    
    #sort by long ranking
    #combined_ranking=pd.DataFrame([ranking,ranking_short]).transpose().sort_values(by=[ranking.name], ascending=True)
    #sort by short ranking
    combined_ranking=pd.DataFrame([ranking,ranking_mid,ranking_short]).transpose().sort_values(by=[ranking_short.name], ascending=True)
    
    
    perchgDict[account]=combined_ranking
    #perchgDict[account].plot(kind='barh', figsize=(10,15))
    
    i2=0
    i=0
    rank_num=[]
    for x in ranking_short.index:
        if ranking_short[x]<0:
            i2-=1
            rank_num.append(i2)
            #print i2,x, ranking_short[x]
        else:
            if i==0:
                i=len(ranking_short)+i2
            else:
                i-=1
            rank_num.append(i)
            #print i,x, ranking_short[x]
            
    #perchgDict[account].index=[str(len(combined_ranking.index)-idx)+' Rank '+col for idx,col in enumerate(combined_ranking.index)]
    perchgDict[account].index=[str(rank_num[idx])+' Rank '+col for idx,col in enumerate(combined_ranking.index)]
    
def createRankingChart(ranking, account, line, title, filename, quantity):
    global currentdate
    fig=plt.figure(1, figsize=(10,15))
    ax = fig.add_subplot(111) 
    colors=['b','y','g']
    colors2=['b','y','r']
    if is_int(line):
        anti='Anti-'+line
        #print line, anti
    else:
        if 'Anti' in line and is_int(line.replace('Anti-','')):
            anti=line.replace('Anti-','')
            #print line, anti
        else:
            #component
            anti=anti_components[line]
            #print line, anti
    color_index_ticks=['r' if line==x.split()[2] or anti==x.split()[2] else 'black' for x in ranking.index]
    color_index_ticks=[color_index_ticks[i] if (x.split()[2] not in component_text.keys() or color_index_ticks[i] == 'r')\
                       else 'b' for i,x in enumerate(ranking.index)]
    #color_index=[['r','r'] if line==x.split()[2] or anti==x.split()[2] else ['b','g'] for x in ranking.index]
    pair=sorted([x for x in ranking.index if line==x.split()[2] or anti==x.split()[2]])
    #ranking.plot(kind='barh', figsize=(10,15), width=0.6)
    for i,col in enumerate(list(ranking)):
        #c = colors[col[0]]
        color_index=[colors2[i] if line==x.split()[2] or anti==x.split()[2] else colors[i] for x in ranking.index]
        #color_index2=[color_index[i] if x.split()[2] not in component_text.keys() else 'b' for x in ranking.index]
        ranking[col].plot(kind='barh', width=0.6, ax=ax,color=color_index)
        #ranking
        #pos = positions[i]
        #DFGSum[col].plot(kind='bar', color=c, position=pos, width=0.05)
    [x.set_color(i) for i,x in zip(color_index_ticks,ax.yaxis.get_ticklabels())]
    plt.legend(loc='upper center', bbox_to_anchor=(.5, -0.03),prop={'size':18},
          fancybox=True, shadow=True, ncol=len(colors))
    plt.xlabel('Cumulative % change', size=12)
    title=account+' '+title
    plt.title(title)
    if savePlots:
        plt.savefig(filename, bbox_inches='tight')
        print 'Saved',filename
    if debug and showPlots:
        plt.show()
    plt.close()
    
    #pnl text
    #prevdate=sorted(signalsDict2.keys())[-2]
    #prev_pnl=pd.DataFrame()
    prevdates=sorted(signalsDict2.keys())[:-1]
    currentdates=sorted(signalsDict2.keys())[1:]
    currentsig=signalsDict2[currentdates[-1]][line].astype(int).copy()
    currentsig=(signalsDict2[currentdates[-1]][line]*quantity).astype(int).copy()
    lq=lastquotes[['Date','PNL']].ix[futuresDict.ix[lastquotes.index].sort_values(by=['Group', 'Desc']).index].copy()
    lq['Group']=futuresDict.ix[lq.index].Group
    lq['PNL2']=lq['PNL']*currentsig
    lq['Qty']=currentsig
    sym_order=[x for x in lq.index if x in active_symbols[account]]
    lq=lq.ix[sym_order][['Group','Date','PNL2','Qty']]
    lq.columns=['Group','Timestamp','PNL','Qty']
    #lq.index.name=line
    #lq.index=['<a href="/static/images/v4_'+sym+'_BRANK.png" target="_blank">'+re.sub(r'\(.*?\)', '', futuresDict.ix[sym].Desc)+'</a>' if sym in futuresDict.index else 'Total' for sym in lq.index ]
    lq.set_value('Total', 'PNL', lq.PNL.sum())
    lq.set_value('Total', 'Timestamp', '')
    lq.PNL=lq.PNL.astype(int)
    #text='Current PNL<br>'+pd.DataFrame(lq).to_html(escape=False)
    for prevdate, currentdate in zip(sorted(prevdates, reverse=True), sorted(currentdates, reverse=True)):
        #print prevdate, currentdate
        prevsig=signalsDict2[prevdate][line].astype(int).copy()
        prevsig=(signalsDict2[prevdate][line]*quantity).astype(int).copy()
        #prevsig[prevsig == -1] = 'SHORT'
        #prevsig[prevsig == 1] = 'LONG'
        #prevsig[prevsig == 0] = 'OFF'
        pnl=market_pnl_by_date[currentdate]['PNL_'+line].ix[sym_order].astype(int)
        pnl['Total']=pnl.sum()
        #pnl.name='{} as of MOC {}'.format(pnl.name,currentdate)
        pnl.name='MOC{}'.format(currentdate)
        lq=pd.concat([lq,pd.DataFrame({'Qty':prevsig.ix[pnl.index], pnl.name:pnl})], axis=1)
    lq2=lq.copy()
    lq.index=['<a href="/static/images/v4_'+sym+'_BRANK.png" target="_blank">'+re.sub(r'\(.*?\)', '', futuresDict.ix[sym].Desc)+'</a>' if sym in futuresDict.index else 'Total' for sym in pnl.index ]
    lq.index.name=line
    text='<br>PNL<br>'+pd.DataFrame(lq).to_html(escape=False)
    
    lookback_name=str(lookback)+'Day Lookback'
    text+='<br>'+lookback_name+': '+', '.join([index+' '+str(round(ranking.ix[index].ix[lookback_name],1))+'%' for index in pair])
    lookback_name=str(lookback_mid)+'Day Lookback'
    ranking=ranking.sort_values(by=[lookback_name], ascending=True)
    ranking.index=[x.split()[2] for x in ranking.index]
    ranking.index=[str(len(ranking.index)-idx)+' Rank '+col for idx,col in enumerate(ranking.index)]
    pair=sorted([x for x in ranking.index if line==x.split()[2] or anti==x.split()[2]])
    text+='<br>'+lookback_name+': '+', '.join([index+' '+str(round(ranking.ix[index].ix[lookback_name],1))+'%' for index in pair])
    lookback_name=str(lookback_short)+'Day Lookback'
    ranking=ranking.sort_values(by=[lookback_name], ascending=True)
    ranking.index=[x.split()[2] for x in ranking.index]
    ranking.index=[str(len(ranking.index)-idx)+' Rank '+col for idx,col in enumerate(ranking.index)]
    pair=sorted([x for x in ranking.index if line==x.split()[2] or anti==x.split()[2]])
    text+='<br>'+lookback_name+': '+', '.join([index+' '+str(round(ranking.ix[index].ix[lookback_name],1))+'%' for index in pair])

    
    return text, lq2
lq_dict={}
prev_signals={}
performance_chart_dict={} 
for account in totals_accounts:
    lq_dict[account]={}
    prev_signals[account]={}
    performance_chart_dict[account]=pd.DataFrame()
    performance_dict[account]={}
    quantity=futuresDF_current[qtydict[account]].copy()
    quantity.ix[[sym for sym in quantity.index if sym not in active_symbols[account]]]=0
    totalsDF=totals_accounts[account]
    #pnl_cols=[x for x in totalsDF.columns if 'PNL' in x]
    vskeys=votingSystems.keys()
    vskeys.sort(key=int)
    chart_list=[[key,'Anti-'+key,'benchmark'] for key in vskeys]
    chart_list+=[[x[0],x[1],'benchmark'] for x in componentpairs]
    benchmark=totalsDF['PNL_benchmark'].copy()
    benchmark_xaxis_label=[dt.strptime(str(x),'%Y%m%d').strftime('%Y-%m-%d') for x in benchmark.index]
    nrows=benchmark.shape[0]
    font = {
            'weight' : 'normal',
            'size'   : 22}

    matplotlib.rc('font', **font)
    for cl in chart_list:
        fig = plt.figure(0, figsize=(10,8))
        num_plots = len(cl)
        colormap = plt.cm.gist_ncar
        #plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 1, num_plots)])
        plt.gca().set_color_cycle(['b','g','r'])
        # Plot several different functions...
        for line in cl:
            pnl=totalsDF['PNL_'+line].copy().values
            pnl[0]=pnl[0]+accountvalues[account]
            label = benchmark_sym+' '+line if line=='benchmark' else line
            plotvalues=pnl.cumsum()
            performance_chart_dict[account][line]=[int(x) for x in plotvalues]
            performance_chart_dict[account][line+'_cumper']=[round(x,2) for x in (plotvalues/accountvalues[account]-1)*100]
            performance_chart_dict[account][line+'_pnl']=performance_chart_dict[account][line].pct_change().fillna(performance_chart_dict[account][line+'_cumper'][0]).values
            #plotvalues=(pnl.cumsum()/accountvalues[account]-1)*100
            plt.plot(range(nrows), plotvalues, label=line)
            plt.legend(loc='best', prop={'size':16})
            plt.ylabel('$ Account Value', size=12)
            #plt.ylabel('Cumulative %change', size=12)
            plt.xlabel('MOC Date', size=12)
            plt.xticks(range(nrows), benchmark_xaxis_label)
            fig.autofmt_xdate()
        plt.title(account+' '+str(lookback)+'Day Historical Performance: '+', '.join(cl))

        date=benchmark_xaxis_label[-1]
        for line in cl[:2]:
            plt.figure(0)
            filename=pngPath+date+'_'+account+'_'+line.replace('/','')+'.png'
            filename2=date+'_'+account+'_'+line.replace('/','')+'.png'
            if savePlots:
                plt.savefig(filename, bbox_inches='tight')
                print 'Saved',filename

            if is_int(line):
                text= 'Voting System consisting of '+', '.join(infodisplay[line])+'.'
                print line, text, filename2
            else:
                if 'Anti' in line and is_int(line.replace('Anti-','')):
                    text= 'Opposite signal of Voting '+line.replace('Anti-','')+'.'
                    #print line, text, filename2
                else:
                    #component
                    text=component_text[line]
                    #print line, text, filename2
            #signals dict is one day behind
            currentdate=sorted(signalsDict2.keys())[-1]
            signals=(signalsDict2[currentdate][line]*quantity).astype(int).copy()
            signals.index=[re.sub(r'\(.*?\)', '', futuresDict.ix[sym].Desc) for sym in signals.index]
            signals=pd.Series(conv_sig(signals), index=signals.index).to_dict()
            prevdate=sorted(signalsDict2.keys())[-2]
            prevsig=(signalsDict2[prevdate][line]*quantity).astype(int).copy()
            prevsig.index=[re.sub(r'\(.*?\)', '', futuresDict.ix[sym].Desc) for sym in prevsig.index]
            prevsig=pd.Series(conv_sig(prevsig), index=prevsig.index).to_dict()    
            prev_signals[account][line]={'signals':prevsig}
            text2='Results shown reflect daily close-to-close timesteps, only applicable to MOC orders. All results are hypothetical. Excludes slippage and commission costs.'
            filename=pngPath+date+'_'+account+'_'+line.replace('/','')+'_ranking.png'
            filename3=date+'_'+account+'_'+line.replace('/','')+'_ranking.png'
            title= line+' Ranking from '+benchmark_xaxis_label[0]+' to '+benchmark_xaxis_label[-1]
            text3, lq = createRankingChart(perchgDict[account], account, line, title, filename, quantity)
            lq_dict[account][line]=lq
            performance_dict[account][line]={
                                                            'rank_filename':filename3,
                                                            'rank_text':text3,
                                                            'filename':filename2,
                                                            'infotext':text,
                                                            'infotext2':text2,
                                                            'signals':signals,
                                                            'date':date,
                                                            }
        if debug and showPlots:
            plt.show()
        plt.close()
    performance_chart_dict[account].index=benchmark_xaxis_label

account_values={}
#create account value charts
for account in totals_accounts:
    totalsDF=totals_accounts[account]
    benchmark_values=totalsDF['PNL_benchmark'].copy()
    #print account, benchmark_values
    benchmark_values.index=benchmark_xaxis_label
    #shift 1 because moc results delayed by one day.
    simulated_moc=pd.read_sql('select * from (select * from {}_live where orderType=\'MOC\' order by timestamp)\
                                            group by Date'.format(account), con=readConn, index_col='Date').selection.shift(1).dropna()
    simulated_moc.index=[dt.strptime(str(x),'%Y%m%d') for x in simulated_moc.index]
    #print account,simulated_moc[-5:]
    
    if account=='v4futures':
        broker='ib'
        accountvalue=pd.read_sql('select * from (select * from ib_accountData where Desc=\'NetLiquidation\'\
                                                order by timestamp ASC) group by Date', con=readConn)
        accountvalue.value=[float(x) for x in accountvalue.value.values]
        timestamps=[timezone('UTC').localize(dt.utcfromtimestamp(ts)).astimezone(timezone('US/Eastern')) for ts in accountvalue.timestamp]
        accountvalue.index=timestamps
        monthly_pctchg=accountvalue.value.resample('M').pct_change().dropna()*100
        monthly_pctchg.index=[dt.strftime(date,'%Y-%b') for date in monthly_pctchg.index]
        monthly_pctchg.name='Monthly %Chg'
        av_xaxis_label=[dt.strftime(date,'%Y-%m-%d') for date in timestamps]
        accountvalue.index=av_xaxis_label
        xaxis_labels=[x for x in benchmark_xaxis_label if x in av_xaxis_label]

        accountvalue2=accountvalue.ix[xaxis_labels].copy()
        accountvalue2.index.name='xaxis'
        newidx=accountvalue2.reset_index().xaxis.drop_duplicates(keep='last').index
        accountvalue2=accountvalue2.reset_index().ix[newidx]
        xaxis_labels=accountvalue2.xaxis.values
        yaxis_values=accountvalue2.value.values
        yaxis_pnl=accountvalue2.value.diff().fillna(0).values
        dates=accountvalue2.reset_index().ix[newidx].Date.values
        #slippage=[]
        commissions=[]
        for date in dates:
            slip_df=pd.read_sql('select * from ib_slippage where timestamp=(select max(timestamp) from ib_slippage where Date=\'{}\' and name=\'{}\')'.format(str(date), account), con=readConn)
            #slippage.append(slip_df.dollarslip.sum())
            commissions.append(-slip_df.commissions.sum())
    else:
        broker='c2'
        accountvalue=pd.read_sql('select * from (select * from c2_equity where\
                                system=\'{}\' order by timestamp ASC) group by Date'.format(account), con=readConn)
        accountvalue.index=pd.to_datetime(accountvalue.updatedLastTimeET)
        monthly_pctchg=accountvalue.modelAccountValue.resample('M').pct_change().dropna()*100
        monthly_pctchg.index=[dt.strftime(date,'%Y-%b') for date in monthly_pctchg.index]
        monthly_pctchg.name='Monthly %Chg'
        av_xaxis_label=[dt.strftime(date,'%Y-%m-%d') for date in accountvalue.index]
        xaxis_labels=[x for x in benchmark_xaxis_label if x in av_xaxis_label]
        accountvalue.index=av_xaxis_label
        accountvalue2=accountvalue.ix[xaxis_labels].copy()
        accountvalue2.index.name='xaxis'
        newidx=accountvalue2.reset_index().xaxis.drop_duplicates(keep='last').index
        accountvalue2=accountvalue2.reset_index().ix[newidx]
        xaxis_labels=accountvalue2.xaxis.values
        yaxis_values=accountvalue2.modelAccountValue.values
        yaxis_pnl=accountvalue2.modelAccountValue.diff().fillna(0).values
        dates=accountvalue2.Date.values
        #slippage=[]
        commissions=[]
        for date in dates:
            slip_df=pd.read_sql('select * from slippage where timestamp=(select max(timestamp) from slippage where csiDate=\'{}\' and name=\'{}\')'.format(str(date), account), con=readConn)
            #slippage.append(slip_df.dollarslip.sum())
            commissions.append(-slip_df.commissions.sum())
    
    #intersect index with benchmark axis

    benchmark_pnl=benchmark_values.ix[xaxis_labels].copy().values
    benchmark_values=benchmark_values.ix[xaxis_labels].values
    
    index=[dt.strptime(date, '%Y-%m-%d') for date in xaxis_labels]
    simulated_moc=simulated_moc.ix[index].fillna('Off')
    simulated_moc_values=np.array([totalsDF.ix[int(idx.strftime('%Y%m%d'))]['PNL_'+simulated_moc.ix[idx]] for idx in simulated_moc.index])
    simulated_moc_pnl=simulated_moc_values.copy()
    simulated_moc_values[0]=simulated_moc_values[0]+yaxis_values[0]
    simulated_moc_values=simulated_moc_values.cumsum()
    #simulated_moc_values_percent=np.insert(np.diff(simulated_moc_values).cumsum()/float(simulated_moc_values[0])*100,0,0)
    simulated_moc_values_percent=pd.Series(simulated_moc_values).pct_change().fillna(0).values
    #yaxis_values_percent=np.insert(np.diff(yaxis_values).cumsum()/float(yaxis_values[0])*100,0,0)
    yaxis_values_percent=pd.Series(yaxis_values).pct_change().fillna(0).values
    benchmark_values[0]=benchmark_values[0]+yaxis_values[0]
    benchmark_values=benchmark_values.cumsum()
    #benchmark_values_percent=np.insert(np.diff(benchmark_values).cumsum()/float(benchmark_values[0])*100,0,0)
    benchmark_values_percent= pd.Series(benchmark_values).pct_change().fillna(0).values
    
    #can't get non-trade prices from c2/ib at the time of moc so slip is a plug. 
    slippage=yaxis_pnl-simulated_moc_pnl-commissions
    fig = plt.figure(figsize=(10,8))
    #num_plots = 2
    #colormap = plt.cm.gist_ncar
    #plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, num_plots)])
    ax = fig.add_subplot(111) 
    ax.plot(index, yaxis_values, 'b', alpha=0.5, label=account+' $ account values')
    ax.plot(index, benchmark_values, alpha=0.4, color='r',\
                label=benchmark_sym+' benchmark $ value')
    ax.plot(index, simulated_moc_values, alpha=0.4, color='g',\
                label='Simulated MOC $ value')
                
    ax.set_ylabel('$ Account Values', size=12)
    #ax.legend(loc='upper left', prop={'size':16})
    ax.legend(loc='upper center', bbox_to_anchor=(.1, -0.15),prop={'size':16},
              fancybox=True, shadow=True, ncol=1)
    ax.xaxis.set_major_formatter(DateFormatter('%b %d %Y'))
    #ax.xaxis.set_major_formatter(tick.FuncFormatter(format_date))
    ax.xaxis.set_major_locator(WeekdayLocator(MONDAY))
    ax.xaxis.set_minor_locator(WeekdayLocator(byweekday=(TU,WE,TH,FR)))
    ax.xaxis.set_minor_formatter(DateFormatter('%d'))
    DateFormatter('%b %d %Y')
    ax2 = ax.twinx()
    ax2.plot(index, yaxis_values_percent, 'b', ls=':', alpha=0.5, label=account+' daily % change')
    ax2.plot(index, benchmark_values_percent, alpha=0.4, color='r',ls=':',\
                label=benchmark_sym+' benchmark daily % change')
    ax2.plot(index, simulated_moc_values_percent, alpha=0.4, color='g',ls=':',\
                label='Simulated MOC daily % change')
    ax2.set_ylabel('% Change', size=12)
    ax.set_xlabel('MOC Date', size=12)
    #ax.set_xticklabels(xaxis_labels)
    plt.title(broker+' '+account+' Equity Chart '+str(lookback)+' day lookback', size=16)
    #ax2.legend(loc='lower left', prop={'size':16})
    ax2.legend(loc='upper center', bbox_to_anchor=(.7, -0.15),prop={'size':16},
              fancybox=True, shadow=True, ncol=1)
    #align_yaxis(ax, 0, ax2, 0)
    fig.autofmt_xdate()
    
    date=dt.strftime(index[-1], '%Y-%m-%d')
    filename=pngPath+date+'_'+account+'_'+broker+'_account_value.png'
    filename2=date+'_'+account+'_'+broker+'_account_value.png'
    if savePlots:
        plt.savefig(filename, bbox_inches='tight')
        print 'Saved',filename
    
    account_values[account]=pd.DataFrame(data={'yaxis_values':yaxis_values, 'benchmark_values':benchmark_values,
                                'simulated_moc_values':simulated_moc_values,'yaxis_values_percent':yaxis_values_percent,
                                'benchmark_values_percent':benchmark_values_percent,
                                'simulated_moc_values_percent':simulated_moc_values_percent,
                                'yaxis_pnl':yaxis_pnl, 'benchmark_pnl':benchmark_pnl,
                                'simulated_moc_pnl':simulated_moc_pnl,
                                'selection':simulated_moc, 'slippage':slippage,
                                'commissions':commissions}, index=index)
    account_values[account]['benchmark_sym']=benchmark_sym
    
    if debug and showPlots:
        plt.show()
    plt.close()
    
    text='This chart shows results from all betting activities of the player.<br>'+\
            pd.DataFrame(monthly_pctchg).transpose().to_html()
    print text
    performance_dict[account]['account_value']={
                                                'rank_filename':'',
                                                'rank_text':'',
                                                'filename':filename2,
                                                'infotext':text,
                                                'signals':'',
                                                'date':date,
                                                }

performance_dict_by_box={}

for account in performance_dict:
    keys=performance_dict[account].keys()
    if len(performance_dict_by_box)==0:
        for key in performance_dict[account].keys():
            performance_dict_by_box[key]={}
            
    for key in performance_dict[account].keys():
        performance_dict_by_box[key][account]=performance_dict[account][key]

performance_dict_by_box2={}
for key in performance_dict_by_box:
    newdict={}
    signals_cons=pd.DataFrame()
    signals_cons_prev=pd.DataFrame()
    for account in performance_dict_by_box[key]:
        newdict[account+'_filename']=performance_dict_by_box[key][account]['filename']
        signals_cons=signals_cons.append(pd.Series(performance_dict_by_box[key][account]['signals'], name=account))
        if key != 'account_value':
            signals_cons_prev=signals_cons_prev.append(pd.Series(prev_signals[account][key]['signals'], name=account))
        newdict[account+'_rank_filename']=performance_dict_by_box[key][account]['rank_filename']
        newdict[account+'_rank_text']=performance_dict_by_box[key][account]['rank_text']
        if key=='account_value':
            newdict[account+'_infotext']=performance_dict_by_box[key][account]['infotext']
        else:
            print key, performance_dict_by_box[key][account]['infotext']
            newdict['infotext']=performance_dict_by_box[key][account]['infotext']
    if 'infotext2' in performance_dict_by_box[key][account]:
        newdict['infotext2']=performance_dict_by_box[key][account]['infotext2']        
    newdict['date']=performance_dict_by_box[key][account]['date']

    if key != 'account_value':
        signals_cons=signals_cons.transpose()
        signals_cons.columns=[web_accountnames[x] for x in signals_cons.columns]
        signals_cons.index=['<a href="/static/images/v4_'+[futuresDict.index[i] for i,desc in enumerate(futuresDict.Desc)\
                                    if re.sub(r'-[^-]*$','',x) in desc][0]+'_BRANK.png" target="_blank">'+x+'</a>' for x in signals_cons.index]
        signals_cons.index.name=key
        signals_cons_prev=signals_cons_prev.transpose()
        signals_cons_prev.columns=[web_accountnames[x] for x in signals_cons_prev.columns]
        signals_cons_prev.index=['<a href="/static/images/v4_'+[futuresDict.index[i] for i,desc in enumerate(futuresDict.Desc)\
                                    if re.sub(r'-[^-]*$','',x) in desc][0]+'_BRANK.png" target="_blank">'+x+'</a>' for x in signals_cons_prev.index]
        signals_cons_prev.index.name=key
        
        newdict['signals']= signals_cons[['50K', '100K', '250K']].to_html(escape=False)
        newdict['signals']+= '<br>Signals as of {} <br>'.format(prevdate)+signals_cons_prev[['50K', '100K', '250K']].to_html(escape=False)
    else:
        newdict['signals']=''
    performance_dict_by_box2[key]=newdict
    
filename=jsonPath+'performance_data.json'
with open(filename, 'w') as f:
     json.dump(performance_dict_by_box2, f)
print 'Saved',filename

for account in totals_accounts:
    tablename=account+'_totals'
    totals_accounts[account].to_sql(name=tablename,con=writeWebChartsConn, index=True, if_exists='replace',\
                    index_label='Date')
    print 'saved',tablename, 'to',dbPathWebCharts

for account in performance_chart_dict:
    tablename=account+'_performance'
    performance_chart_dict[account].to_sql(name=tablename,con=writeWebChartsConn, index=True, if_exists='replace',\
                    index_label='Date')
    print 'saved',tablename, 'to',dbPathWebCharts
    
for account in perchgDict:
    tablename=account+'_ranking'
    perchgDict[account].to_sql(name=tablename,con=writeWebChartsConn, index=True, if_exists='replace',\
                    index_label='Ranking'+str(lookback_short)+'D')
    print 'saved',tablename, 'to',dbPathWebCharts

for account in account_values:
    tablename=account+'_accountvalues'
    account_values[account].to_sql(name=tablename,con=writeWebChartsConn, index=True, if_exists='replace',\
                    index_label='Date')
    print 'saved',tablename, 'to',dbPathWebCharts
    

#signals list
for d in sorted(signalsDict2.keys())[-1:]:
    print 'signals charts for', d
    d2=str(d)[:4]+'-'+str(d)[4:6]+'-'+str(d)[6:]
    keys=signalsDict2[d].keys()
    voting_keys=sorted([x.split('Anti-')[1] for x in keys if 'Anti-' in x and is_int(x.split('Anti-')[1])], key=int)
    anti_voting_keys=['Anti-'+x for x in voting_keys]
    component_keys=[x for x in keys if x not in voting_keys and x not in anti_voting_keys]
    component_keys.remove('Off')
    component_keys.remove('benchmark')
    #component_keys=sorted(component_keys)
    component_keys=['Custom',
     'RiskOn',
     'HighestEquity',
     '50/50',
     'LowestEquity',
     'RiskOff',
     'Seasonality',
     'Previous',
     #'Anti-Custom',
     #'Anti-Previous',
     #'Anti-Seasonality',
     #'Anti50/50',
     #'AntiHighestEquity',
     #'AntiLowestEquity'
     ]
    cmap = sns.diverging_palette(350, 120, sep=2, as_cmap=True)
    
    
    for l,name in [(component_keys,'Components'), (voting_keys,'Voting'), (anti_voting_keys,'Antivoting')]:
        df=pd.DataFrame()
        for k in l:
            #print k
            signalsDict2[d][k].name=k
            df=pd.concat([df, signalsDict2[d][k]],axis=1)
        df=df.ix[futuresDict.ix[df.index].sort_values(by=['Group','Desc']).index]
        desc_list=futuresDict.ix[df.index].Desc.values
        idx2=futuresDF_current.ix[df.index].LastPctChg.values*100
        idx1=[re.sub(r'\(.*?\)', '', desc) for desc in desc_list]
        df.index=[x+' '+str(round(idx2[i],2))+'%' for i,x in enumerate(idx1)]
        fig,ax = plt.subplots(figsize=(15,15))
        #title = 'Lookback '+str(lookback)+' '+data.index[-lookback-1].strftime('%Y-%m-%d')+' to '+data.index[-1].strftime('%Y-%m-%d')
        title='{} {} Signals Heatmap'.format(d, name)
        ax.set_title(title)
        sns.heatmap(ax=ax, data=df,cmap=cmap)
        plt.yticks(rotation=0) 
        plt.xticks(rotation=90) 
        filename=pngPath+d2+'_'+name+'_heatmap.png'
        plt.savefig(filename, bbox_inches='tight')
        print 'Saved',filename
        if debug and showPlots:
            plt.show()
        plt.close()


#last signal accuracy by market by system
prev_signals=pd.DataFrame()
prev_acc=pd.DataFrame()
for sys in signalsDict2[prev[0]]:
    series=signalsDict2[prev[0]][sys]
    series.name=sys
    prev_signals=pd.concat([prev_signals,series], axis=1)
        
        

for col in prev_signals:
    #print col
    acc=pd.DataFrame(index=prev_signals[col].index)
    nonzero=prev_signals[col][prev_signals[col] !=0].copy()
    correct=nonzero==futuresDF_current.ACT.ix[nonzero.index]
    
    for sym in acc.index:
        if sym in correct.index:
            signal=prev_signals[col].ix[sym]
            if correct.ix[sym]:
                #correct long 2
                if signal>0:
                    acc.set_value(sym,col,2)
                #correct short -2
                elif signal<0:
                    acc.set_value(sym,col,-2)
                
            else:
                #incorrect long 1
                if signal>0:
                    acc.set_value(sym,col,1)
                #incorrect short -1
                if signal<0:
                    acc.set_value(sym,col,-1)
        else:
            #off 0
            acc.set_value(sym,col,0)
    prev_acc=pd.concat([prev_acc,acc],axis=1)
    
prev_acc=prev_acc.ix[futuresDict.ix[prev_acc.index].sort_values(by=['Group','Desc']).index]


cmap = sns.diverging_palette(0, 255, sep=2, as_cmap=True)
for l,name in [(component_keys,'Components'), (voting_keys,'Voting'), (anti_voting_keys,'Antivoting')]:
    df=prev_acc[l]

    for account in performance_dict:
        df2=df.ix[active_symbols[account]].copy()
        colnames=[]
        for col in df2.columns:
            nonzero=df2[col][df2[col]!=0]
            if len(nonzero)>0:
                acc= str(round(float(len(nonzero[abs(nonzero)==2]))/len(nonzero)*100,1))+'%'
            else:
                acc='0%'
            colnames.append(col+' '+acc)
        df2.columns=colnames
        
        desc_list=futuresDict.ix[df2.index].Desc.values
        idx2=futuresDF_current.ix[df2.index].LastPctChg.values*100
        idx1=[re.sub(r'\(.*?\)', '', desc) for desc in desc_list]
        df2.index=[x+' '+str(round(idx2[i],2))+'%' for i,x in enumerate(idx1)]
        
        fig,ax = plt.subplots(figsize=(15,15))
        #title = 'Lookback '+str(lookback)+' '+data.index[-lookback-1].strftime('%Y-%m-%d')+' to '+data.index[-1].strftime('%Y-%m-%d')
        title='{} {} {} Signals Accuracy Heatmap (light-incorrect, dark-correct, blue-long, red-short)'.format(account, prev[0], name)
        ax.set_title(title)
        sns.heatmap(ax=ax, data=df2,cmap=cmap)
        plt.yticks(rotation=0) 
        plt.xticks(rotation=90) 
        filename=pngPath+d2+'_'+account+'_'+name+'_accuracy_heatmap.png'
        plt.savefig(filename, bbox_inches='tight')
        print 'Saved',filename
        if debug and showPlots:
            plt.show()
        plt.close()

lq_dict2={}
for account in totals_accounts:
    lq_df=pd.DataFrame()
    for line in lq_dict[account].keys():
        total=lq_dict[account][line]['PNL'].Total
        lq_df.set_value(line, 'TotalPNL',total)
        #print account, line, total
    lq_dict2[account]=lq_df.sort_values(by='TotalPNL')
    
    fig=plt.figure(1, figsize=(10,15))
    ax = fig.add_subplot(111) 
    color_index=['b' if x in anti_components.keys() else 'black' for x in lq_dict2[account].index]
    
    #[x.set_color(i) for i,x in zip(color_index,ax.yaxis.get_ticklabels())]
    
    lq_dict2[account].plot(kind='barh', width=0.6, ax=ax, color=color_index)

    plt.xlabel('Total PNL', size=12)
    title=account+' Total Current PNL as of '+lq.Timestamp.values.max()
    plt.title(title)
    filename=pngPath+d2+'_'+account+'_current_total_pnl.png'
    if savePlots:
        plt.savefig(filename, bbox_inches='tight')
        print 'Saved',filename
    if debug and showPlots:
        plt.show()
    plt.close()

print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes ', dt.now()

