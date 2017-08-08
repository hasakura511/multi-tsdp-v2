import numpy as np
import pandas as pd
import time
from os import listdir
from os.path import isfile, join
from swigibpy import EPosixClientSocket, ExecutionFilter, CommissionReport, Execution, Contract
import sys
import random
import copy
import pytz
from pytz import timezone
from datetime import datetime as dt
from tzlocal import get_localzone
import os
from dateutil.parser import parse
import logging
import re

rtbar={}
rtdict={}
rthist={}
rtfile={}
rtreqid={}
lastDate={}
tickerId=1
dataPath='./data/from_IB/'

if len(sys.argv)==1:
    debug=True
    showPlots=True
    dataPath='D:/ML-TSDP/data/'
    portfolioPath = 'D:/ML-TSDP/data/portfolio/'
    #savePath= 'C:/Users/Hidemi/Desktop/Python/TSDP/ml/data/results/' 
    savePath = savePath2 = pngPath='C:/Users/Hidemi/Desktop/Python/TSDP/ml/data/results/' 
    systemPath =  'D:/ML-TSDP/data/systems/'
    systemfile='D:/ML-TSDP/data/systems/system_ibfeed.csv'
else:
    debug=False
    showPlots=False
    dataPath='./data/'
    portfolioPath = './data/portfolio/'
    savePath='./data/'
    pngPath = './data/results/'
    savePath2 = './data/portfolio/'
    systemPath =  './data/systems/'
    systemfile='./data/systems/system_consolidated.csv'
    
def get_cash_contracts():
    symList=dict()
    global systemfile
    systemdata=pd.read_csv(systemfile)
    #print systemdata.columns
    systemdata=systemdata.reset_index()
    for i in systemdata.index:
        #print 'Read: ',i
        system=systemdata.ix[i]
        #print system
        contract = Contract()
        
        if system['ibtype'] == 'CASH':
            symbol=system['ibsym']+system['ibcur']
        else:
            symbol=system['iblocalsym']

        #contract.symbol = system['ibsym']
        contract.secType = system['ibtype']
        contract.exchange = system['ibexch']
        contract.currency = system['ibcur']
        contract.symbol=contract.localSymbol= system['iblocalsym']
        symList[symbol]=contract
    print len(symList.keys()), symList.keys()
    return symList.keys(), symList.values()  
    
def update_bars(contracts, interval='30m'):
    global tickerId
    global lastDate
    global dataPath
    while 1:
        try:
            for contract in contracts:
                filename=dataPath+interval+'_'+contract+'.csv'
                minFile='./data/bars/'+contract+'.csv'
                symbol = contract
                if os.path.isfile(minFile):
                    data=pd.read_csv(minFile)
                     
                    eastern=timezone('US/Eastern')
                    
                    date=data.iloc[-1]['Date']
                    date=parse(date).replace(tzinfo=eastern)
                    timestamp = time.mktime(date.timetuple())
                    
                    if not lastDate.has_key(symbol):
                        lastDate[symbol]=timestamp
                        dataFile=dataPath+'1 min_'+contract+'.csv'
                        if os.path.isfile(dataFile):
                            data=pd.read_csv(dataFile)
                            regentime=60
                            if interval == '30m':
                                regentime=60*6
                            elif interval == '1h':
                                regentime = 60 * 6
                            elif interval == '10m':
                                regentime == 60 * 6
                            
                            quote=data
                            if quote.shape[0] > regentime:
                                quote=quote.tail(regentime)
                            for i in quote.index:
                                data=quote.ix[i]
                                compress_min_bar(symbol, data, filename, interval)
                                       
                    if lastDate[symbol] < timestamp:
                        lastDate[symbol]=timestamp
                        quote=data.iloc[-1]
                        compress_min_bar(symbol, quote, filename, interval) 
                        
            time.sleep(20)
        except Exception as e:
            logging.error("update_bars", exc_info=True)
