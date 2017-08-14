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
    return symList.values()  