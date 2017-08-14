import copy
import pandas as pd
from os import listdir
from os.path import isfile, join
import sys
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime as dt
import time
import sqlite3
import re


start_time = time.time()
#with open('./data/futures.txt') as f:
#    futures = f.read().splitlines()
def getBackendDB():
    dbPath = './data/futures.sqlite3'
    readConn = sqlite3.connect(dbPath)
    return readConn

readConn = getBackendDB()

ib_contracts = pd.read_csv('./data/systems/ib_contracts.csv', index_col='CSIsym2')

debug=False

if debug:
    dataPath ='D:/ML-TSDP/data/csidata/v4futures2/'
    savePath =  'C:/Users/Hidemi/Desktop/Python/TSDP/ml/data/results/' 
    #pairPath='D:/ML-TSDP/data/csidata/v4futures/'
else:
    savePath = './web/tsdp/betting/static/images/'
    dataPath = './data/csidata/v4futures2/'
    #pairPath='./data/'
    
lookback=1
futuresMatrix=pd.DataFrame()
files = [ f for f in listdir(dataPath) if isfile(join(dataPath,f)) ]
#auxFutures = [x.split('_')[0] for x in files]
auxFutures=ib_contracts.index

for contract in auxFutures:    
    #if 'F_'+contract+'.txt' in files and (ticker[0:3] in contract or ticker[3:6] in contract):
    filename = contract+'_B.CSV'
    data = pd.read_csv(dataPath+filename, index_col=0, header=None)[-(lookback+10):]
    
    #data = data.drop([' P',' R', ' RINFO'],axis=1)
    #data = ratioAdjust(data)
    data.index = pd.to_datetime(data.index,format='%Y%m%d')
    data.columns = ['Open','High','Low','Close','Volume','OI','R','S']
    data.index.name = 'Dates'
    #print contract, data,'\n'
    #contract = ''.join([i for i in contract if not i.isdigit()])
    if 'YT' not in contract:
        contract = ''.join([i for i in contract.split('_')[0] if not i.isdigit()])
    else:
        contract=contract.split('_')[0]

    for contract2 in auxFutures:
        filename = contract2+'_B.CSV'
        data2 = pd.read_csv(dataPath+filename, index_col=0, header=None)[-(lookback+10):]    
        data2.index = pd.to_datetime(data2.index,format='%Y%m%d')
        data2.columns = ['Open','High','Low','Close','Volume','OI','R','S']
        data2.index.name = 'Dates'
        #contract = ''.join([i for i in contract if not i.isdigit()])
        if 'YT' not in contract2:
            contract2 = ''.join([i for i in contract2.split('_')[0] if not i.isdigit()])
        else:
            contract2=contract2.split('_')[0]
        
        union=data.index.union(data2.index)
        #print 'Missing data:',contract,contract2, union
        
        #forward fill missing data
        data=data.ix[union].fillna(method='ffill')
        data2=data2.ix[union].fillna(method='ffill')
            
        change=(data.Close/data2.Close).pct_change(periods=lookback)[-1]*100
        #print contract, data, '\n', contract2, data2
        #print change, '\n\n'
        futuresMatrix.set_value(contract,contract2,change)
    #print contract, data,'\n'

for contract in auxFutures:
    if 'YT' not in contract:
        contract = ''.join([i for i in contract.split('_')[0] if not i.isdigit()])
    else:
        contract=contract.split('_')[0]
    futuresMatrix.set_value(contract,'Avg',futuresMatrix.ix[contract].dropna().mean())
    
futuresMatrix=futuresMatrix.sort_values(by='Avg', ascending=False).drop('Avg', axis=1)
futuresDict = pd.read_sql('select * from Dictionary', con=readConn,\
                          index_col='CSIsym')

desc_list =  futuresDict.ix[futuresMatrix.columns].Desc
futuresMatrix.columns = [re.sub(r'\(.*?\)', '', desc) for desc in desc_list]
desc_list =  futuresDict.ix[futuresMatrix.index].Desc
futuresMatrix.index = [re.sub(r'\(.*?\)', '', desc) for desc in desc_list]

#rankByMean=futuresMatrix['Avg'].sort_values(ascending=False)

#with open(savePath+'futures_1.html','w') as f:
#    f.write(str(data.index[0])+' to '+str(data.index[-1]))
    
#futuresMatrix.to_html(savePath+'futures_3.html')

#print futuresMatrix
fig,ax = plt.subplots(figsize=(15,13))
ax.set_title(str(data.index[-lookback-1])+' to '+str(data.index[-1]))
sns.heatmap(ax=ax,data=futuresMatrix)
plt.yticks(rotation=0) 
plt.xticks(rotation=90) 
#plt.pcolor(futuresMatrix)
#plt.yticks(np.arange(0.5, len(futuresMatrix.index), 1), futuresMatrix.index)
#plt.xticks(np.arange(0.5, len(futuresMatrix.columns), 1), futuresMatrix.columns)
if savePath != None:
    print 'Saving '+savePath+'futures_2.png'
    fig.savefig(savePath+'futures_2.png', bbox_inches='tight')
    
if len(sys.argv)==1:
    print data.index[0],'to',data.index[-1]
    plt.show()
#print rankByMean

'''
ranking = rankByMean.index
buyHold=[]
sellHold=[]
cplist = copy.deepcopy(futures)
for contract in ranking:
    for i,pair in enumerate(cplist):
        #print pair
        if pair not in buyHold and pair not in sellHold:
            if contract in pair[0:3]:
                #print i,'bh',pair
                buyHold.append(pair)
                #cplist.remove(pair)
            elif contract in pair[3:6]:
                #print i,'sh',pair
                sellHold.append(pair)
                #cplist.remove(pair)
            #else:
                #print i,contract,pair
print 'buyHold',len(buyHold),buyHold
print 'sellHold',len(sellHold),sellHold
'''
print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes ', dt.now()
