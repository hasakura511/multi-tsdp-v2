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
if len(sys.argv)==1:
    debug=True
else:
    debug=False
    
if debug:
    mode = 'replace'
    #marketList=[sys.argv[1]]
    showPlots=True
    dbPath='./data/futures.sqlite3' 
    dbPath2='./data/futures.sqlite3' 
    dbPath3='./web/tsdp/db_charts.sqlite3'
    dbPathWeb = './web/tsdp/db.sqlite3'
    dataPath='./data/csidata/v4futures2/'
    savePath= './data/results/' 
    pngPath = './data/results/' 
    feedfile='./data/systems/system_ibfeed.csv'
    #test last>old
    #dataPath2=pngPath
    #signalPath = './data/signals/' 
    
    #test last=old
    dataPath2='./data/'
    
    #signalPath ='./data/signals2/'
    signalPath ='./data/signals2/' 
    signalSavePath = './data/signals/' 
    systemPath = './data/systems/' 
    stylePath =jsonPath=  './web/tsdp/'
    readConn = sqlite3.connect(dbPath2)
    readChartConn= sqlite3.connect(dbPath3)
    writeConn= sqlite3.connect(dbPath)
    #readWebConn = sqlite3.connect(dbPathWeb)
    #logging.basicConfig(filename='C:/logs/vol_adjsize_live_func_error.log',level=logging.DEBUG)
else:
    mode= 'replace'
    #marketList=[sys.argv[1]]
    showPlots=False
    feedfile='./data/systems/system_ibfeed.csv'
    dbPath='./data/futures.sqlite3'
    dbPathWeb ='./web/tsdp/db.sqlite3'
    jsonPath ='./web/tsdp/'
    dataPath='./data/csidata/v4futures2/'
    #dataPath='./data/csidata/v4futures2/'
    dataPath2='./data/'
    savePath='./data/results/'
    signalPath = './data/signals2/' 
    signalSavePath = './data/signals2/' 
    pngPath =  './web/tsdp/betting/static/images/'
    systemPath =  './data/systems/'
    stylePath =  './web/tsdp/'
    readConn = writeConn= sqlite3.connect(dbPath)
    dbPath3='./web/tsdp/db_charts.sqlite3'
    readChartConn= sqlite3.connect(dbPath3)
    #readWebConn = sqlite3.connect(dbPathWeb)
    #logging.basicConfig(filename='/logs/vol_adjsize_live_func_error.log',level=logging.DEBUG)
    
readWebConn = sqlite3.connect(dbPathWeb)

filename=jsonPath+'accountinfo_data.json'
with open(filename, 'r') as f:
     accountinfo=json.load(f)

active_symbols={}
for account in accountinfo.keys():
    active_symbols[account]=eval(accountinfo[account]['online'])
    
#active_symbols={
#                        'v4futures':['AD', 'BO', 'BP', 'C', 'CD', 'CL', 'CU', 'EMD', 'ES', 'FC',
#                                           'FV', 'GC', 'HG', 'HO', 'JY', 'LC', 'LH', 'MP', 'NE', 'NG',
#                                           'NIY', 'NQ', 'PA', 'PL', 'RB', 'S', 'SF', 'SI', 'SM', 'TU',
#                                           'TY', 'US', 'W', 'YM'],
#                        'v4mini':['C', 'CL', 'CU', 'EMD', 'ES', 'HG', 'JY', 'NG', 'SM', 'TU', 'TY', 'W'],
#                        'v4micro':['BO', 'ES', 'HG', 'NG', 'TY'],
#                        }
                        
def saveCharts(df, **kwargs):
    ylabel=kwargs.get('ylabel','')
    xlabel=kwargs.get('xlabel','')
    title=kwargs.get('title','')
    filename=kwargs.get('filename','chart.png')
    width=kwargs.get('width',0.8)
    legend_outside=kwargs.get('legend_outside',False)
    figsize=kwargs.get('figsize',(15,13))
    kind=kwargs.get('kind','bar')
    df2=kwargs.get('df2',None)
    lookback=kwargs.get('lookback',5)
    twiny=kwargs.get('twiny',False)
    
    font = {
            'weight' : 'normal',
            'size'   : 22}
    matplotlib.rc('font', **font)

    fig, ax = plt.subplots(figsize=figsize)
    colormap = plt.cm.gist_ncar
    plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, lookback)])

    
    if df2 is not None:
        ax.set_xlabel(xlabel, size=24)
        ax.set_ylabel(ylabel, size=24)
        ax.grid(False)
        ax.yaxis.grid(False)
        ax2 = ax.twinx()
        ax2.set_ylabel('Avg $', size=24)
        ax2.grid(False)
        df.plot(kind=kind, ax=ax,  color='blue', width=width)
        df2.plot(kind='line',  colors=['g','r'], ax=ax2)
        fig.autofmt_xdate()
        plt.title(title, size=30)
        for label in ax.xaxis.get_majorticklabels():
            label.set_fontsize(18)
            #label.set_fontname('courier')
        for label in ax.yaxis.get_majorticklabels():
            label.set_fontsize(18)
            #label.set_fontname('verdana')
        for label in ax2.yaxis.get_majorticklabels():
            label.set_fontsize(18)
            #label.set_fontname('verdana')

    else:
        df.plot(kind=kind, ax=ax, width=width)
        plt.ylabel(ylabel, size=24)
        #plt.ylabel('Cumulative %change', size=12)
        plt.xlabel(xlabel, size=24)
        plt.xticks(fontsize = 24)
        plt.yticks(fontsize = 24)
        
        if twiny:
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())
            ax2.set_xticks(ax.get_xticks())
            for label in ax2.xaxis.get_majorticklabels():
                label.set_fontsize(18)
                
            plt.title(title, size=30,y=1.01)
        else:
            plt.title(title, size=30)
                
    if legend_outside:
        plt.legend(loc='upper center', bbox_to_anchor=(.5, 1.15),prop={'size':24},
          fancybox=True, shadow=True, ncol=3)
    else:
        if twiny:
            ax.legend(loc='upper center', bbox_to_anchor=(.5, 1.04),prop={'size':24},
              fancybox=True, shadow=True, ncol=3)
        else:
            plt.legend(loc='best', prop={'size':24})
        

    #plt.xticks(range(nrows), benchmark_xaxis_label)
    #fig.autofmt_xdate()
    

    #filename2=date+'_'+account+'_'+line.replace('/','')+'.png'
    plt.savefig(filename, bbox_inches='tight')
    print 'Saved',filename
    if debug and showPlots:
        plt.show()
    plt.close()

#for account in active_symbols:

account='v4futures'
totalsDF=pd.read_sql('select * from {}'.format('totalsDF_board_'+account), con=readConn,  index_col='Date')
date=str(totalsDF.index[-1])
totalsDF.index=totalsDF.index.astype(str).to_datetime().strftime('%A, %b %d %Y')

#volatility by group
lookback=5
volatility_cols=[x for x in totalsDF.columns if 'Vol' in x and 'Total' not in x]
df=totalsDF[volatility_cols].iloc[-lookback:].transpose()
ylabel='$ Volatiliy'
xlabel='Group'
title='Day Volatility by Group'
filename=pngPath+date+'_'+title.replace(' ','')+'.png'
title=str(lookback)+' Day Volatility by Group'
width=.8
saveCharts(df, ylabel=ylabel, xlabel=xlabel, title=title, filename=filename)

#change by group
lookback=5
change_cols=[x for x in totalsDF.columns if 'Chg' in x and 'Total' not in x]
df=totalsDF[change_cols].iloc[-lookback:].transpose()
ylabel='$ Change'
xlabel='Group'
title='Day Change by Group'
filename=pngPath+date+'_'+title.replace(' ','')+'.png'
title=str(lookback)+' Day Change by Group'
width=.8
saveCharts(df, ylabel=ylabel, xlabel=xlabel, title=title, filename=filename)

#long% columns
lookback=5
lper_cols=['L%_currency','L%_energy','L%_grain','L%_index','L%_meat','L%_metal','L%_rates','L%_Total',]
df=totalsDF[lper_cols].iloc[-lookback:].transpose()*100
ylabel='% Long'
xlabel='Group'
title='Long Percent by Group'
filename=pngPath+date+'_'+title.replace(' ','')+'.png'
title=str(lookback)+' Day Long Percent by Group'
width=.8
saveCharts(df, ylabel=ylabel, xlabel=xlabel, title=title, filename=filename, legend_outside= True)

#average totals
lookback=5
avg_cols=[x for x in totalsDF.columns if 'Avg' in x]
df=totalsDF[avg_cols].iloc[-lookback:]
df2=totalsDF['L%_Total'].iloc[-lookback:]*100
ylabel='Long %'
#xlabel='Dates'
title='Day Averages by Group'
filename=pngPath+date+'_'+title.replace(' ','')+'.png'
title=str(lookback)+' Day Averages by Group'
width=.6
saveCharts(df2, df2=df, ylabel=ylabel, title=title, filename=filename, kind='bar',\
            lookback=2, width=width, legend_outside= True)
            
for account in active_symbols:
    totalsDF=pd.read_sql('select * from {}'.format('totalsDF_board_'+account), con=readConn,  index_col='Date')
    date=str(totalsDF.index[-1])
    totalsDF.index=totalsDF.index.astype(str).to_datetime().strftime('%A, %b %d %Y')
    #accuracy
    lookback=3
    acc_cols=[x for x in totalsDF.columns if 'ACC' in x and 'benchmark' not in x and 'Off' not in x]
    df=totalsDF[acc_cols].iloc[-lookback:].transpose()
    df=df.sort_values(by=df.columns[-1])*100
    ylabel='Systems'
    xlabel='% Accuracy'
    title=' Day Accuracy by System'
    filename=pngPath+date+'_'+account+'_'+title.replace(' ','')+'.png'
    title=account+' '+str(lookback)+' '+title
    width=.6
    saveCharts(df, ylabel=ylabel, xlabel=xlabel, title=title, filename=filename, figsize=(15,50),\
                        lookback=lookback, kind='barh',width=width, twiny=True)

    #pnl
    lookback=3
    pnl_cols=[x for x in totalsDF.columns if 'PNL' in x and 'benchmark' not in x and 'Off' not in x]
    df=totalsDF[pnl_cols].iloc[-lookback:].transpose()
    df=df.sort_values(by=df.columns[-1])
    ylabel='Systems'
    xlabel='$ PNL'
    title=' Day $ PNL by System'
    filename=pngPath+date+'_'+account+'_'+title.replace(' ','')+'.png'
    title=account+' '+str(lookback)+' '+title
    width=.6
    saveCharts(df, ylabel=ylabel, xlabel=xlabel, title=title, filename=filename, figsize=(15,50),\
                        lookback=lookback, kind='barh',width=width, twiny=True)


av_cumper_df=pd.DataFrame()
for account in active_symbols.keys():
    #totalsDF=pd.read_sql('select * from {}'.format('totalsDF_board_'+account), con=readConn,  index_col='Date')
    #benchmark_values=totalsDF['PNL_benchmark'].copy()
    #print account, benchmark_values
    #benchmark_values.index=benchmark_xaxis_label=[dt.strptime(str(x),'%Y%m%d').strftime('%Y-%m-%d') for x in benchmark_values.index]
    
    if account=='v4futures':
        #broker='ib'
        title='MOM\'s '+account
        accountvalue=pd.read_sql('select * from (select * from ib_accountData where Desc=\'NetLiquidation\'\
                                                order by timestamp ASC) group by Date', con=readConn)
        accountvalue.value=[float(x) for x in accountvalue.value.values]
        timestamps=[timezone('UTC').localize(dt.utcfromtimestamp(ts)).astimezone(timezone('US/Eastern')) for ts in accountvalue.timestamp]
        accountvalue.index=[dt.strftime(date,'%Y-%m-%d') for date in timestamps]
        accountvalue.index.name='xaxis'
        newidx=accountvalue.reset_index().xaxis.drop_duplicates(keep='last').index
        xaxis_values=accountvalue.reset_index().ix[newidx].xaxis.values
        yaxis_values=accountvalue.reset_index().ix[newidx].value.values
    else:
        #broker='c2'
        title='DAD\'s '+account
        accountvalue=pd.read_sql('select * from (select * from c2_equity where\
                                system=\'{}\' order by timestamp ASC) group by Date'.format(account), con=readConn)
        accountvalue.index=[dt.strftime(date,'%Y-%m-%d') for date in pd.to_datetime(accountvalue.updatedLastTimeET)]
        accountvalue.index.name='xaxis'
        newidx=accountvalue.reset_index().xaxis.drop_duplicates(keep='last').index
        xaxis_values=accountvalue.reset_index().ix[newidx].xaxis.values
        yaxis_values=accountvalue.reset_index().ix[newidx].modelAccountValue.values
    
    #yaxis_values_percent=np.insert(np.diff(yaxis_values).cumsum()/float(yaxis_values[0])*100,0,0)
    av_cumper_df=pd.concat([av_cumper_df,pd.Series(data=yaxis_values, index=xaxis_values, name=title)], axis=1, join='outer')
    
##update chip
filename = stylePath+'boxstyles_data.json'
if isfile(filename):
    with open(filename, 'r') as f:
        boxstyles_data = json.load(f)

accounts = {x.split()[1]:x for x in av_cumper_df.columns.tolist()}
for dic in boxstyles_data:
    key=dic.keys()[0]
    if len(key.split('_'))>1:
        key2 =key.split('_')[1]
    else:
        key2=''
    if key2 in accounts:
        index=boxstyles_data.index(dic)
        #print av_cumper_df
        #print accounts[key2], av_cumper_df[accounts[key2]][-1]
        av=int(av_cumper_df[accounts[key2]].dropna()[-1])
        chip_value=str(av/1000)+'K'
        print key, key2, index,dic[key]['text'],av, chip_value
        dic[key]['text']=chip_value
with open(filename, 'w') as f:
     json.dump(boxstyles_data, f)
print 'Saved',filename

fig = plt.figure()
ax = fig.add_subplot(111)
av_cumper_df=av_cumper_df.dropna().copy()
for col in av_cumper_df:
    yaxis_values=av_cumper_df[col].values
    av_cumper_df[col]=np.insert(np.diff(yaxis_values).cumsum()/float(yaxis_values[0])*100,0,0)
av_cumper_df.index=pd.to_datetime(av_cumper_df.index)
av_cumper_df.plot(ax=ax, figsize=(15,13))
ax.xaxis.set_major_formatter(DateFormatter('%b %d %Y'))
#ax.xaxis.set_major_formatter(tick.FuncFormatter(format_date))
ax.xaxis.set_major_locator(WeekdayLocator(MONDAY))
ax.xaxis.set_minor_locator(WeekdayLocator(byweekday=(TU,WE,TH,FR)))
ax.xaxis.set_minor_formatter(DateFormatter('%d'))
DateFormatter('%b %d %Y')
plt.ylabel('Cumulative % Change', size=24)
plt.xlabel('MOC Date', size=24)
#ax.set_xticklabels(xaxis_labels)
plt.title('Account Comparison', size=24)
#ax2.legend(loc='lower left', prop={'size':16})
plt.legend(loc='upper center', prop={'size':24},
          fancybox=True, shadow=True, ncol=3)
fig.autofmt_xdate()
    
date=dt.strftime(av_cumper_df.index[-1], '%Y-%m-%d')
filename=pngPath+date+'_comparison_account_value.png'
plt.savefig(filename, bbox_inches='tight')
print 'Saved',filename
if debug and showPlots:
    plt.show()
plt.close()

date2=str(totalsDF.index[-1])

for account in active_symbols.keys():
    fig = plt.figure()
    ax = fig.add_subplot(111)
    performance=pd.read_sql('select * from {}'.format(account+'_performance'), con=readChartConn)
    performance=performance[[x for x in performance.columns if '_pnl' in x]]
    
    listofsystems=[]
    for col in performance:
        lastthree=performance[col].values[-3:]
        if (lastthree[0]<0 and lastthree[1]>0 and lastthree[2]>0) or\
                (lastthree[0]>0 and lastthree[1]<0 and lastthree[2]<0):
            listofsystems+=[col]
            
    listofsystems=[x.split('_pnl')[0] for x in listofsystems]
        
    ranking=pd.read_sql('select * from {}'.format(account+'_ranking'), con=readChartConn)
    indexname=[x for x in ranking.columns if 'Ranking' in x][0]
    index=[x.split('Rank')[1].strip() for i,x in enumerate(ranking[indexname])]
    combinedranking=pd.DataFrame(index=index)
    
    for system in index:
        if system != 'Off':
            returns = performance[system+'_pnl']
            combinedranking.set_value(system,'Sharpe',np.sqrt(20) * returns.mean() / returns.std())
        else:
            combinedranking.set_value(system,'Sharpe',0)
    combinedranking['Sharpe']=(combinedranking['Sharpe']-combinedranking['Sharpe'].mean())/combinedranking['Sharpe'].std()
    
    for col in [ranking.drop([indexname],axis=1).columns[-1]]:
        combinedranking[col]=((ranking[col]-ranking[col].mean())/ranking[col].std()).values
        
        
    combinedranking['Score']=combinedranking.sum(axis=1)
    combinedranking=combinedranking.ix[listofsystems].sort_values(by='Score', ascending=True)
    combinedranking.index=[x+' ({})'.format(str(len(combinedranking.index)-i)) for i,x in enumerate(combinedranking.index)]
    combinedranking.plot(ax=ax, kind='barh', figsize=(12,24))
    plt.ylabel('Ranking', size=24)
    plt.xlabel('Z Score', size=24)
    #ax.set_xticklabels(xaxis_labels)
    plt.title(account+' Scores as of '+date2, size=24)
    #ax2.legend(loc='lower left', prop={'size':16})
    plt.legend(loc='upper center', prop={'size':18}, bbox_to_anchor=(.4, -0.03),
              fancybox=True, shadow=True, ncol=4)
    filename=pngPath+date+'_'+account+'_'+'ranking_zscores.png'
    plt.savefig(filename, bbox_inches='tight')
    print 'Saved',filename
    if debug and showPlots:
        plt.show()
    plt.close()
    

print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes ', dt.now()