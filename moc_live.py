#import ibapi.futures_bars_1d as bars
import os
from os import listdir
from os.path import isfile, join
import random
import sys
from shutil import copyfile
from subprocess import Popen, PIPE, check_output
import pandas as pd
import numpy as np
import threading
import time
import logging
import copy
import calendar
#import get_feed2 as feed
from pytz import timezone
from dateutil.parser import parse
import datetime
import traceback
from ibapi.wrapper_v5 import IBWrapper, IBclient
from ibapi.place_order2 import place_orders as place_iborders
from swigibpy import Contract 
import pandas as pd
from time import gmtime, strftime, localtime, sleep
import json
import datetime
from pandas.io.json import json_normalize
from pytz import timezone
from datetime import datetime as dt
from tzlocal import get_localzone
import logging
from swigibpy import EPosixClientSocket, ExecutionFilter, CommissionReport, Execution, Contract
from dateutil.parser import parse
import sqlite3
#from suztoolz.vol_adjsize_live_func import vol_adjsize_live
from suztoolz.check_systems_live_func import check_systems_live
from suztoolz.proc_signal_v4_live_func import proc_signal_v4_live
from suztoolz.vol_adjsize_moclive_func import vol_adjsize_moc
from suztoolz.vol_adjsize_immediate_func import vol_adjsize_immediate

import slackweb
fulltimestamp=datetime.datetime.now().strftime('%Y%m%d_%H_%M_%S')
slackhook='https://hooks.slack.com/services/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'
slack = slackweb.Slack(url=slackhook)
slack_channel="#logs"


#currencyPairsDict=dict()
#prepData=dict()
start_time = time.time()
callback = IBWrapper()
client=IBclient(callback, port=7496, clientid=0)


#systems = ['v4micro','v4mini','v4macro']

durationStr ='2 D'
barSizeSetting='30 mins'
whatToShow='TRADES'

filename=None
eastern=timezone('US/Eastern')
endDateTime=dt.now(get_localzone())
endDateTime=endDateTime.astimezone(eastern)
endDateTime=endDateTime.strftime("%Y%m%d %H:%M:%S EST")    
data = pd.DataFrame({}, columns=['Date','Open','High','Low','Close','Volume']).set_index('Date')
#tickerId=random.randint(100,9999)

interval='1d'
minDataPoints = 5

#python moc_live.py 1 1 1
#python moc_live.py live submitc2 submitIB
if len(sys.argv)==1:
    #systems = ['v4mini']
    systems = ['v4micro','v4mini','v4futures']
    debug=True
    run_moc_immediate=True
    immediate=False
    showPlots=True
    submitIB=False
    submitC2=False
    #if trigger time set to value then trigger time in feedfile is ignored.
    triggertimes = None
    #triggertime = 30 #mins
    dbPath='./data/futures.sqlite3' 
    dbPath2='D:/ML-TSDP/data//futures.sqlite3' 
    runPath='D:/ML-TSDP/run_futures_live.py'
    runPath2= ['python','D:/ML-TSDP/vol_adjsize_live.py']
    runPath3=  ['python','D:/ML-TSDP/proc_signal_v4_live.py','0']
    runPath4=['python','D:/ML-TSDP/check_systems_live.py','0']
    logPath='C:/logs/'
    dataPath='D:/ML-TSDP/data/'
    #portfolioPath = 'D:/ML-TSDP/data/portfolio/'
    #savePath= './data/results/' 
    savePath = savePath2 = pngPath=portfolioPath='./data/results/' 
    #systemPath =  'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/systems/'
    systemPath = 'D:/ML-TSDP/data/systems/'
    feedfile='D:/ML-TSDP/data/systems/system_ibfeed.csv'
    #systemfile='D:/ML-TSDP/data/systems/system_v4micro.csv'
    timetablePath=   'D:/ML-TSDP/data/systems/timetables_debug/'
    #feedfile='D:/ML-TSDP/data/systems/system_ibfeed_fx.csv'
    csiDataPath=  'D:/ML-TSDP/data/csidata/v4futures2/'
    csiDataPath2=  'D:/ML-TSDP/data/csidata/v4futures3_debug/'
    csiDataPath3=  'D:/ML-TSDP/data/csidata/v4futures4_debug/'
    csiDataPath4=  'D:/ML-TSDP/data/csidata/v4futures_last_debug/'
    signalPathDaily =  'D:/ML-TSDP/data/signals/'
    signalPathMOC =  'D:/ML-TSDP/data/signals2/'
    logging.basicConfig(filename='C:/logs/broker_live_'+dt.now().strftime('%Y%m%d-%H%M%S')+'.log',level=logging.DEBUG)
else:
    systems = ['v4micro','v4mini','v4futures']
    debug=False
    showPlots=False

    if sys.argv[1]=='1':
        #run MOC/immediate process
        run_moc_immediate=True
    else:
        run_moc_immediate=False
        
    if sys.argv[2]=='1':
        submitC2=True
    else:
        submitC2=False
    
    if sys.argv[3]=='1':
        submitIB=True
    else:
        submitIB=False
    
    if sys.argv[4]=='1':
        immediate=True
    else:
        immediate=False
        
    if sys.argv[1]=='0' and sys.argv[2]=='0' and sys.argv[3]=='0' and sys.argv[4]=='0':
        refresh_timetable=True
    else:
        refresh_timetable=False

    if sys.argv[1]=='0' and sys.argv[2]=='0' and sys.argv[3]=='1' and sys.argv[4]=='1':
        post_processing=True
    else:
        post_processing=False


    if sys.argv[1]=='0' and sys.argv[2]=='0' and sys.argv[3]=='1' and sys.argv[4]=='0':
        get_lastquotes=True
    else:
        get_lastquotes=False

    triggertimes = None
    #triggertime = 30 #mins
    dbPath=dbPath2='./data/futures.sqlite3'
    runPath='./run_futures_live.py'
    runPath2=['python','./vol_adjsize_live.py','1']
    runPath3=['python','./proc_signal_v4_live.py','1']
    runPath4=['python','./check_systems_live.py','1']
    logPath='/logs/'
    dataPath='./data/'
    portfolioPath = './data/portfolio/'
    savePath='./data/'
    pngPath = './web/tsdp/betting/static/images/'
    savePath2 = './data/portfolio/'
    systemPath =  './data/systems/'
    feedfile='./data/systems/system_ibfeed.csv'
    #systemfile='./data/systems/system_v4micro.csv'
    timetablePath=   './data/systems/timetables/'
    #feedfile='D:/ML-TSDP/data/systems/system_ibfeed_fx.csv'
    csiDataPath=  './data/csidata/v4futures2/'
    csiDataPath2=  './data/csidata/v4futures3/'
    csiDataPath3=  './data/csidata/v4futures4/'
    csiDataPath4=  './data/csidata/v4futures_last/'
    signalPathDaily =  './data/signals/'
    signalPathMOC =  './data/signals2/'
    logging.basicConfig(filename='/logs/broker_live_'+dt.now().strftime('%Y%m%d-%H%M%S')+'.log',level=logging.DEBUG)

writeConn = sqlite3.connect(dbPath)
readConn =  sqlite3.connect(dbPath2)

timeZoneId ={'AUD': 'CST',
     'CAD': 'CST',
     'CHF': 'CST',
     'CL': 'EST5EDT',
     'EMD': 'CST',
     'ES': 'CST',
     'EUR': 'CST',
     'GBP': 'CST',
     'GC': 'EST5EDT',
     'GF': 'CST',
     'HE': 'CST',
     'HG': 'EST5EDT',
     'HO': 'EST5EDT',
     'JPY': 'CST',
     'LE': 'CST',
     'MXP': 'CST',
     'NG': 'EST5EDT',
     'NIY': 'CST',
     'NQ': 'CST',
     'NZD': 'CST',
     'PA': 'EST5EDT',
     'PL': 'EST5EDT',
     'RB': 'EST5EDT',
     'SI': 'EST5EDT',
     'YM': 'CST',
     'ZB': 'CST',
     'ZC': 'CST',
     'ZF': 'CST',
     'ZL': 'CST',
     'ZM': 'CST',
     'ZN': 'CST',
     'ZS': 'CST',
     'ZT': 'CST',
     'ZW': 'CST'}
     
tzDict = {
    'CST':'CST6CDT',
    'EST':'EST5EDT',
    }
days = {
                0:'Mon',
                1:'Tues',
                2:'Wed',
                3:'Thurs',
                4:'Fri',
                5:'Sat',
                6:'Sun',
                }
months = {
                'F':1,
                'G':2,
                'H':3,
                'J':4,
                'K':5,
                'M':6,
                'N':7,
                'Q':8,
                'U':9,
                'V':10,
                'X':11,
                'Z':12
                }
IB2CSI_multiplier_adj={
    'HG':100,
    'SI':100,
    'JPY':100,
    }
    
def is_int(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False
    
def getTableColumns(dbconn, tablename):
    dbcur = dbconn.cursor()
    dbcur.execute("""
        PRAGMA table_info('{0}')
        """.format(tablename.replace('\'', '\'\'')))
    columns=dbcur.fetchall()
    dbcur.close()
    if len(columns)>0:
        return [x[1] for x in columns]
    else:
        return []

def get_write_mode(dbconn, tablename, dataframe):
    dbcols=getTableColumns(dbconn, tablename)
    check=[col in dbcols for col in dataframe.columns]
    if False in check:
        return 'replace'
    else:
        return 'append'
    
def lastCsiDownloadDate():
    global csiDataPath
    datafiles = os.listdir(csiDataPath)
    dates = []
    for f in datafiles:
        lastdate = pd.read_csv(csiDataPath+f, index_col=0).index[-1]
        if lastdate not in dates:
            dates.append(lastdate)
            
    return max(dates)
    
csidate=lastCsiDownloadDate()

    
def guessServerdate():
    eastern = timezone('US/Eastern')
    now = dt.now(get_localzone())
    now = now.astimezone(eastern)
    reset_time = datetime.time(1, 0, 0, 0)
    

    if now.time()>reset_time:
        #update the date
        ib_server_reset_date = now.strftime("%Y%m%d")
    else:
        #date is not yet updated, keep it previous date.
        ib_server_reset_date =  (now - datetime.timedelta(days=1)).strftime("%Y%m%d")
        
    if now.weekday() == 5:
        # Saturday so set to monday
        next = now + datetime.timedelta(days=2)
        ib_server_reset_date = next.strftime("%Y%m%d")

    if now.weekday() == 6:
        # Sunday so set to monday
        next = now + datetime.timedelta(days=1)
        ib_server_reset_date = next.strftime("%Y%m%d")
    return ib_server_reset_date

ib_server_reset_date =guessServerdate()

def lastTimeTableDate():
    #load timetable
    ttfiles = os.listdir(timetablePath)
    ttdates = []
    for f in ttfiles:
        if '.csv' in f and is_int(f.split('.')[0]):
            ttdates.append(int(f.split('.')[0]))
        
    return max(ttdates)

ttdate= lastTimeTableDate()

def getContractDate(c2sym, systemdata):
    currentcontract = [x for x in systemdata.c2sym if x[:-2] == c2sym]
    if len(currentcontract)==1:
        #Z6
        ccontract = currentcontract[0][-2:]
        ccontract = 201000+int(ccontract[-1])*100+months[ccontract[0]]
        return str(ccontract)
    else:
        return ''

def runInThread(sym, popenArgs):
    with open(logPath+sym+'.txt', 'w') as f:
        with open(logPath+sym+'_error.txt', 'w') as e:
            proc = Popen(popenArgs, stdout=f, stderr=e)
            proc.wait()
            #check_output(popenArgs)
            proc2= Popen(popenArgs2, stdout=f, stderr=e)
            proc2.wait()
            proc_orders(sym)
        return
    thread = threading.Thread(target=runInThread)
    #, args=(onExit, popenArgs))
    thread.start()
    # returns immediately after the thread starts
    return thread

def runThreads(threadlist):

    def runInThread(sym, popenArgs):
        print 'starting thread for', sym
        
        with open(logPath+sym+'.txt', 'w') as f:
            with open(logPath+sym+'_error.txt', 'w') as e:
                proc = Popen(popenArgs, stdout=f, stderr=e)
                proc.wait()
                f.flush()
                e.flush()
                print sym,'Done!'
                #check_output(popenArgs)
                #proc2= Popen(popenArgs2, stdout=f, stderr=e)
                #proc2.wait()
                #proc_orders(sym)
            return
            
    threads=[]
    for arg in threadlist:
        #print arg
        t = threading.Thread(target=runInThread, args=arg)
        threads.append(t)
        
     # Start all threads
    for x in threads:
        x.start()

     # Wait for all of them to finish
    for x in threads:
        x.join()

    
def get_ibfutpositions(portfolioPath):
    global client
    global csidate
    (account_value, portfolio_data)=client.get_IB_account_data()
    
    if len(account_value) != 0:
        accountSet=pd.DataFrame(account_value,columns=['desc','value','currency','account_id'])
        accountSet=accountSet.set_index(['desc'])
        filename=portfolioPath+'ib_account_value.csv'
        accountSet.to_csv(filename, index=True)
        print 'saved', filename
        if 'NetLiquidation' in accountSet.index:
            accountValue = accountSet.ix['NetLiquidation'].value        
            print 'Account value:', accountValue
            try:
                accountSet['Date']=csidate
                accountSet['timestamp']=int(calendar.timegm(dt.utcnow().utctimetuple()))
                accountSet.to_sql(name='ib_accountData', con=writeConn, index=True, if_exists='append', index_label='Desc')
                print 'saved ib_accountData to', dbPath
            except Exception as e:
                #print e
                traceback.print_exc()
    else:
        print 'Account value returned nothing'
        
    if len(portfolio_data) !=0:
        data=pd.DataFrame(portfolio_data,columns=['sym','exp','qty','price','value','avg_cost','unr_pnl','real_pnl','accountid','currency'])
        #return contracts only
        dataSet=data[data.exp != ''].copy(deep=True)
        dataSet['contracts']=[dataSet.ix[i].sym+dataSet.ix[i].exp for i in dataSet.index]
        dataSet=dataSet.set_index(['contracts'])
        filename=portfolioPath+'ib_portfolio.csv'
        dataSet.to_csv(filename)
        print 'saved', filename
        try:
            dataSet['Date']=csidate
            dataSet['timestamp']=int(calendar.timegm(dt.utcnow().utctimetuple()))
            dataSet.to_sql(name='ib_portfolioData', con=writeConn, index=True, if_exists='append', index_label='contracts')
            print 'saved ib_portfolioData to', dbPath
        except Exception as e:
            #print e
            traceback.print_exc()
        
        return dataSet.drop(['Date','timestamp'],axis=1)
    else:
        return 0
  
        
def create_execDict(feeddata, systemdata):
    global debug
    global client
    global csidate
    global ttdate
    
    #after 1AM EST server reset
    downloadtt = not ttdate==int(ib_server_reset_date)
    print 'ttdate',ttdate, 'ib_server_reset_date',ib_server_reset_date, 'downloadtt', downloadtt,'refresh_timetable',refresh_timetable
    execDict=dict()
    #need systemdata for the contract expiry
    #systemdata=pd.read_csv(systemfile)

    #openPositions=get_ibfutpositions(portfolioPath)
    #print feeddata.columns
    #if True:
    if downloadtt or refresh_timetable:
        downloadtt=True
        print 'new timetable due, geting new contract details from IB'
        contractsDF=pd.DataFrame()
    else:
        print 'loading contract details from file'
        contractsDF=pd.read_csv(systemPath+'ib_contracts.csv', index_col='ibsym')
        
    feeddata=feeddata.reset_index()
    for i in feeddata.index:
        
        #print 'Read: ',i
        system=feeddata.ix[i]
        c2sym=system.c2sym
        ibsym=system.ibsym   
        index = systemdata[systemdata.c2sym2==c2sym].index[0]
        #find the current contract

        #print system
        contract = Contract()
        
        if system['ibtype'] == 'CASH':
            #fx
            symbol=system['ibsym']+system['ibcur']
            contract.symbol=system['ibsym']
            ccontract = ''
        else:
            #futures
            contract.expiry = getContractDate(system.c2sym, systemdata)
            symbol=contract.symbol= system['ibsym']
            contract.multiplier = str(system['multiplier'])

        #contract.symbol = system['ibsym']
        contract.secType = system['ibtype']
        contract.exchange = system['ibexch']
        contract.currency = system['ibcur']
        
        print i+1, contract.symbol, contract.expiry
        if downloadtt:
            contractInfo=client.get_contract_details(contract)
            #print contractInfo
            contractsDF=contractsDF.append(contractInfo)
            execDict[symbol+contractInfo.expiry[0]]=['PASS', 0, contract]
            systemdata.set_value(index, 'ibcontract', symbol+contractInfo.expiry[0])
        else:
            execDict[contractsDF.ix[symbol].contracts]=['PASS', 0, contract]
            systemdata.set_value(index, 'ibcontract', contractsDF.ix[symbol].contracts)
            
        #update system file with correct ibsym and contract expiry
        #print index, ibsym, contract.expiry, systemdata.columns
        systemdata.set_value(index, 'ibsym', ibsym)
        systemdata.set_value(index, 'ibcontractmonth', contract.expiry)
        

        #print c2sym, ibsym, systemdata.ix[index].ibsym.values, systemdata.ix[index].c2sym.values, contract.expiry
    
    #systemdata.to_csv(systemfile, index=False)
    systemdata.to_sql(name='v4futures_moc_live', if_exists='replace', con=writeConn, index=False)
    print '\nsaved v4futures_moc_live to', dbPath
    
    if downloadtt:
        feeddata=feeddata.set_index('ibsym')
        contractsDF=contractsDF.set_index('symbol')
        contractsDF.index.name = 'ibsym'
        contractsDF['contracts']=[x+contractsDF.ix[x].expiry for x in contractsDF.index]
        contractsDF['Date']=csidate
        contractsDF['timestamp']=int(calendar.timegm(dt.utcnow().utctimetuple()))
        #print contractsDF.index
        #print feeddata.ix[contractsDF.index].drop(['ibexch','ibtype','ibcur'],axis=1).head()
        contractsDF = pd.concat([ feeddata.ix[contractsDF.index].drop(['ibexch','ibtype','ibcur','Date','timestamp'],axis=1),contractsDF], axis=1)
        try:
            contractsDF.to_sql(name='ib_contracts', con=writeConn, index=True, if_exists='replace', index_label='ibsym')
            print '\nsaved ib_contracts to',dbPath
        except Exception as e:
            #print e
            traceback.print_exc()
        if not debug:
            contractsDF.to_csv(systemPath+'ib_contracts.csv', index=True)
            print 'saved', systemPath+'ib_contracts.csv'
            
    print '\nCreated exec dict with', len(execDict.keys()), 'contracts:'
    print execDict.keys()
    return execDict,contractsDF,systemdata
   
def update_orders(feeddata, systemdata2, execDict, threadlist):
    global client
    global portfolioPath
    #systemdata=pd.read_csv(systemfile)
    triggered_syms=[feeddata[feeddata.CSIsym==x[0]].index.values[0] for x in threadlist]
    systemdata= systemdata2.copy(deep=True)
    systemdata['c2sym2']=[x[:-2] for x in systemdata.c2sym]
    #systemdata['c2sym2']=[x[:-2] for x in systemdata.c2sym]
    #systemdata['CSIsym']=[x.split('_')[1] for x in systemdata.System]
    #openPositions=get_ibfutpositions(portfolioPath)
    #print feeddata.columns
    
    portfolio=get_ibfutpositions(portfolioPath)

    if isinstance(portfolio, type(pd.DataFrame())):
        #get rid of 0 qty
        clean_portfolio=portfolio[portfolio.qty != 0]
        print clean_portfolio.shape[0],'futures positions found'
        #openPositions = portfolio.reset_index().groupby(['sym'])[['qty']].sum()
        #openPositions = portfolio.reset_index().groupby(['sym','exp'])[['qty']].sum()
        openPositions = clean_portfolio.reset_index().groupby(clean_portfolio.index)[['qty']].sum()
        currentOpenPos = pd.DataFrame()
        execDictExpired={}
        for contract in openPositions.index:
            #print sym,exp
            sym=contract[:-8]
            exp=contract[-8:]
            exp2=contract[-8:-2]
            qty=openPositions.ix[contract].qty
            #execDict should only contain one match at this point.
            match = [x for x in execDict.keys() if contract[:-8] in x][0]
            #currentExpiry=execDict[match][2].expiry
            currentContract=execDict[match][2]
            if exp==match[-8:]:
                currentOpenPos.set_value(contract, 'qty',openPositions.ix[contract][0])
                currentOpenPos.set_value(contract, 'exp',exp)
            else:
                #add expired contract to execdict if in threadlist to avoid early exits
                if sym in triggered_syms:
                    if qty > 0:
                        action='SLD'
                    else:
                        action='BOT'
                    
                    #create new contract. no deep copy
                    IBcontract = Contract()
                    IBcontract.symbol= currentContract.symbol
                    #missing from ibswigpy
                    #IBcontract.LastTradeDateOrContractMonth=exp
                    IBcontract.multiplier = currentContract.multiplier
                    IBcontract.secType = currentContract.secType
                    IBcontract.exchange = currentContract.exchange
                    IBcontract.currency =currentContract.currency
                    contractInfo=client.get_contract_details(IBcontract)
                    #print contractInfo
                    contractMonth=contractInfo[contractInfo.expiry==exp].contractMonth
                    if len(contractMonth) >0:
                        IBcontract.expiry=contractMonth.values[0]
                        execDictExpired[contract]=[action,int(abs(qty)),IBcontract]
                        print 'Expired contract:', contract, qty, 'added to execDict:',sym, action, abs(qty), IBcontract.expiry
                    else:
                        print 'Expired contract:', contract, qty, 'could not be found. Exit Manually:',sym, action, abs(qty)
                else:
                    print 'Expired contract:', contract, qty, 'not yet triggered'
                    
        feeddata=feeddata.reset_index()
        for i in feeddata.index:
            system=feeddata.ix[i]
            c2sym=system.c2sym
            ibsym=system.ibsym
            if c2sym not in systemdata.c2sym2.values:
                continue
                
            index = systemdata[systemdata.c2sym2==c2sym].index[0]
            currentcontract = [x for x in systemdata.c2sym if x[:-2] == system.c2sym]
            if len(currentcontract)==1:
                #Z6
                ccontract = currentcontract[0][-2:]
                ccontract = 201000+int(ccontract[-1])*100+months[ccontract[0]]
                
            match = [x for x in currentOpenPos.index if ibsym in x]
            
            if len(match) !=0:
                contract=match[0]
                if contract in currentOpenPos.index:
                    ib_pos_qty=currentOpenPos.ix[contract].qty
                else:
                    ib_pos_qty=0
            else:
                ib_pos_qty=0
            #ib_pos_qty=0
            #print ib_pos_qty
            ibquant = systemdata.ix[index].c2qty
            system_ibpos_qty=systemdata.ix[index].signal * ibquant
            #print 'ibq', type(ibquant), 'sysibq', type(system_ibpos_qty)
            #print( "system_ib_pos: " + str(system_ibpos_qty) ),
            #print( "ib_pos: " + str(ib_pos_qty) ),
            
            action='PASS'
            if system_ibpos_qty > ib_pos_qty:
                action = 'BOT'
                ibquant=int(system_ibpos_qty - ib_pos_qty)
                #print( 'BUY: ' + str(ibquant) )
                #place_iborder('BUY', ibquant, ibsym, ibtype, ibcurrency, ibexch, ibsubmit, iblocalsym);
            if system_ibpos_qty < ib_pos_qty:
                action='SLD'
                ibquant=int(ib_pos_qty - system_ibpos_qty)
                #print( 'SELL: ' + str(ibquant) )
                #place_iborder('SELL', ibquant, ibsym, ibtype, ibcurrency, ibexch, ibsubmit, iblocalsym);         
            #print( action+': ' + str(ibquant) )
            #expiry = getContractDate(c2sym, systemdata)
            contract=[x for x in execDict.keys() if ibsym in x][0]
            execDict[contract][0]=action
            execDict[contract][1]=ibquant
            if action != 'PASS':
                print 'Position Change: ', contract, 'IB', ib_pos_qty, 'SYS', system_ibpos_qty, 'ADJ', action, ibquant
        #systemdata.to_csv(systemfile, index=False)
        #print 'saved', systemfile
          
        #print len(execDict.keys()), execDict.keys()
        execDictMerged = execDict.copy()
        execDictMerged.update(execDictExpired)
        return execDictMerged
    else:
        print 'IB returned no positions. Updating execDict from system file.'
        #resetting ibsym because wasn't updated in system csv at the time i wrote this.
        systemdata.index=[feeddata.reset_index().set_index('CSIsym').ix[x.split('_')[1]].ibsym for x in systemdata.System]
        for sym in systemdata.index:
            system=systemdata.ix[sym]
            #print sym, [x for x in execDict.keys() if sym in x]
            execdictkey=[x for x in execDict.keys() if sym in x]
            if len(execdictkey)>0:
                key=execdictkey[0]
                #needs to be type int or ib wrapper will throw an error
                ibquant = int(systemdata.ix[sym].c2qty)
                if systemdata.ix[sym].signal >0:
                    action='BOT'
                else:
                    action='SLD'
                execDict[key][0]=action
                execDict[key][1]=ibquant
                print 'updated', key, action, ibquant
            else:
                print 'could not find', sym, 'in execDict. skipping.'
        return execDict

def get_contractdf(execDict, systemPath):
    global client
    contracts = [x[2] for x in execDict.values()]
    contractsDF=pd.DataFrame()
    for i,contract in enumerate(contracts):
        print i, contract.symbol,
        contractsDF=contractsDF.append(json_normalize(client.get_contract_details(contract)))
    contractsDF.to_csv(systemPath+'ib_contracts.csv', index=False)
    return contractsDF


def refresh_all_histories(execDict):
    global client
    global feeddata
    global endDateTime
    for sym in execDict:
        data = pd.DataFrame({}, columns=['Date','Open','High','Low','Close','Volume']).set_index('Date')
        tickerId=random.randint(100,9999)
        contract = execDict[sym][2]
        print sym, 'getting data from IB',
        data = client.get_history(endDateTime, contract, whatToShow, data ,None,tickerId, minDataPoints, durationStr, barSizeSetting, formatDate=1)
        filename=csiDataPath4+feeddata.ix[sym[:-8]].CSIsym2+'.csv'
        data.to_csv(filename, index=True)
        print 'saved', filename
        
def refresh_history(sym, execDict):
    global client
    global feeddata
    global endDateTime
    data = pd.DataFrame({}, columns=['Date','Open','High','Low','Close','Volume']).set_index('Date')
    tickerId=random.randint(100,9999)
    sym2=[x for x in execDict.keys() if sym in x][0]
    contract = execDict[sym2][2]
    print 'getting data from IB...',
    data = client.get_history(endDateTime, contract, whatToShow, data ,filename,tickerId, minDataPoints, durationStr, barSizeSetting, formatDate=1)
    data.to_csv(csiDataPath2+feeddata.ix[sym].CSIsym2+'.csv', index=True)
    return data
    
def get_tradingHours(sym, contractsDF):
    global triggertimes
    meats= ['GF','LE','HE']
    fmt = '%Y-%m-%d %H:%M'
    dates = contractsDF.ix[sym].tradingHours.split(";")
    
    #IB didn't return timezoneid one day..
    if contractsDF.ix[sym].timeZoneId[:3] =='':
        tz=timezone(tzDict[timeZoneId[sym][:3]])
    else:
        tz = timezone(tzDict[contractsDF.ix[sym].timeZoneId[:3]])
    
    if triggertimes == None:
        triggertime = int(contractsDF.ix[sym].triggertime)
        #print sym, triggertime
    else:
        triggertime = triggertimes
        
    thDict = {}
    print sym, dates
    for th in dates:
        print th
        thlist = th.split(':')
        date = thlist[0]
        if thlist[1] =='CLOSED':
            continue
        else:
            #print thlist
            openclosetimes = thlist[1].split('-')
            opentime = openclosetimes[0]
            if int(date[6:8])-1 ==0:
                #print 'first day of month', date
                #first day of the month
                date2=dates[0].split(':')[0]
                if date != date2:
                    #print 'last day of previous month in timetable. opendate', date2
                    opendate=tz.localize(dt(year=int(date2[0:4]), month=int(date2[4:6]), day=int(date2[6:8]),\
                        hour=int(opentime[0:2]), minute=int(opentime[2:4]))).astimezone(timezone(tzDict['EST']))
                else:
                    date3 = (dt.strptime(date2,'%Y%m%d')-datetime.timedelta(days=1)).strftime('%Y%m%d')
                    #print 'last day of previous month not in timetable. opendate', date3
                    opendate=tz.localize(dt(year=int(date3[0:4]), month=int(date3[4:6]), day=int(date3[6:8]),\
                        hour=int(opentime[0:2]), minute=int(opentime[2:4]))).astimezone(timezone(tzDict['EST']))
            else:
                #print 'not a first day of month', date
                opendate=tz.localize(dt(year=int(date[0:4]), month=int(date[4:6]), day=int(date[6:8])-1,\
                    hour=int(opentime[0:2]), minute=int(opentime[2:4]))).astimezone(timezone(tzDict['EST']))
            closetime = openclosetimes[-1]
            closedate=tz.localize(dt(year=int(date[0:4]), month=int(date[4:6]), day=int(date[6:8]),\
                hour=int(closetime[0:2]), minute=int(closetime[2:4]))).astimezone(timezone(tzDict['EST']))
            triggerdate=closedate-datetime.timedelta(minutes=triggertime)
            
            #overwrite meats
            if sym in meats:
                opendate=tz.localize(dt(year=int(date[0:4]), month=int(date[4:6]), day=int(date[6:8]),\
                    hour=int(opentime[0:2]), minute=int(opentime[2:4]))).astimezone(timezone(tzDict['EST']))
            thDict[date]=[opendate.strftime(fmt),closedate.strftime(fmt), triggerdate.strftime(fmt)]

    return thDict
        

    
def filterIBexec():
    global client
    global feeddata
    global csiDataPath
    global csiDataPath3
    global csidate
    global portfolioPath
    
    executions_raw=pd.DataFrame(client.get_executions())
    if len(executions_raw) ==0:
        print 'IB returned no executions, returning to main thread'
        return None
    filename = portfolioPath+'ib_executions_raw.csv'
    executions_raw.to_csv(filename)
    print 'Saved', filename
    executions_raw=executions_raw.set_index('symbol')
    #get rid of symbols not in feeddata
    executions=executions_raw.ix[[x for x in executions_raw.index if x in feeddata.index]].copy()
    executions['CSIsym2']=[feeddata.ix[sym].CSIsym2 for sym in executions.index]
    index = executions.reset_index().groupby(['CSIsym2'])['times'].transform(max)==executions.times
    executions= executions.reset_index().ix[index].set_index('CSIsym2')

    #append csi's lastdate first if ib's does not exist yet (expired contracts)
    datafiles_csi = os.listdir(csiDataPath)
    for f in [f for f in datafiles_csi if f.split('_')[0] in executions.index]:
        sym = f.split('_')[0]
        lastdate = pd.read_csv(csiDataPath+f, index_col=0).index[-1]
        #print lastdate
        executions.set_value(sym, 'lastAppend', dt.strptime(str(lastdate),'%Y%m%d'))
        
    datafiles_ib = os.listdir(csiDataPath3)
    for f in [f for f in datafiles_ib if f.split('_')[0] in executions.index]:
        sym = f.split('_')[0]
        lastdate = pd.read_csv(csiDataPath3+f, index_col=0).index[-1]
        executions.set_value(sym, 'lastAppend', dt.strptime(str(lastdate),'%Y%m%d'))
        
    executions.times = pd.to_datetime(executions.times)
    executions = executions[executions.times >= executions.lastAppend].reset_index().set_index('symbol')
    
    executions['contract'] =[x+executions.iloc[i].expiry for i,x in enumerate(executions.index)]
    executions.to_csv(portfolioPath+'ib_exec_last.csv', index=True)
    print 'saved', portfolioPath+'ib_exec_last.csv'
    executions['Date']=csidate
    executions['timestamp']=int(calendar.timegm(dt.utcnow().utctimetuple()))
    try:
        tablename='ib_executions'
        mode = get_write_mode(writeConn, tablename, executions)
        executions.to_sql(name=tablename, con=writeConn, index=True, if_exists=mode, index_label='ibsym')
        print 'saved', tablename,'to',dbPath,'writemode',mode
    except Exception as e:
        #print e
        traceback.print_exc()
    return executions.drop(['Date','timestamp'],axis=1)
    
def get_timetable(contractsDF):
    global client
    global csidate
    

    for i,sym in enumerate(contractsDF.index):
        thDict = get_tradingHours(sym, contractsDF)
        if i == 0:
            timetable=pd.DataFrame(thDict,index=[sym+' open',sym+' close',sym+' trigger'])
        else:
            timetable=pd.concat([timetable, pd.DataFrame(thDict,index=[sym+' open',sym+' close',sym+' trigger'])], axis=0)
    print 'csidate', csidate
    filedates=[str(x) for x in [d for d in timetable.columns.astype(int) if d>csidate]]
    filedate=filedates[0]
    print filedates, filedate
    #print timetable.to_csv()
    #fill nans with the next filedate
    #if len(filedates)>1:
    #    next_filedate=filedates[1]
    #    timetable[filedate]=[timetable[next_filedate].ix[idx] if timetable[filedate].ix[idx] is np.nan \
    #                                    else timetable[filedate].ix[idx] for idx in timetable[filedate].index]
    #    timetable=timetable.drop([next_filedate],axis=1)
    print timetable.to_csv()                                  
    filename=timetablePath+str(filedate)+'.csv'
    timetable.to_csv(filename, index=True)
    print 'saved', filename
    timetable['Date']=csidate
    timetable['timestamp']=int(calendar.timegm(dt.utcnow().utctimetuple()))
    try:
        timetable.to_sql(name='timetable', con=writeConn, index=True, if_exists='replace', index_label='Desc')
    except Exception as e:
        #print e
        traceback.print_exc()
    return timetable.drop(['Date','timestamp'],axis=1)
        
def find_triggers(feeddata, contractsDF):
    global csidate
    global ttdate
    #make sure servertime is in EST
    nowDateTime=dt.now()
    #eastern=timezone(tzDict['EST'])
    #nowDateTime=dt.now(get_localzone())
    #nowDateTime=dt.now(get_localzone())+datetime.timedelta(days=5)
    #nowDateTime=nowDateTime.astimezone(eastern)
    loaddate=str(lastTimeTableDate())
    #if ttdate>csidate:
    #    #timetable file date is greater than the csi download date. 
    #    loaddate=str(ttdate)
    #else:
    #    #get a new timetable
    #    print 'csidate',csidate, '>=', 'ttdate', ttdate, 'getting new timetable'
    #    timetable = get_timetable(contractsDF)
    #    loaddate=str([d for d in timetable.columns.astype(int) if d>csidate][0])
        
    filename=timetablePath+loaddate+'.csv'
    timetable=pd.read_csv(filename, index_col=0)
    triggers=timetable.ix[[i for i in timetable.index if 'trigger' in i]][loaddate]
    threadlist=[]
    signalFilesMOC = [ f for f in listdir(signalPathMOC) if isfile(join(signalPathMOC,f)) ]
    signalFilesDaily =[ f for f in listdir(signalPathDaily) if isfile(join(signalPathDaily,f)) ]
    for t in triggers.index:
        ibsym=t.split()[0]
        csiFileSym=feeddata.ix[ibsym].CSIsym2
        csiRunSym=feeddata.ix[ibsym].CSIsym
        if not isinstance(triggers.ix[t], str):
            print t, triggers.ix[t], 'datetime not found skipping..'
            continue
            
        fmt = '%Y-%m-%d %H:%M'
        #tdate=dt.strptime(triggers.ix[t],fmt).replace(tzinfo=eastern)
        tdate=dt.strptime(triggers.ix[t],fmt)
        if nowDateTime>tdate:
            #print 'checking trigger:',
            filename = csiDataPath3+csiFileSym+'_B.CSV'
            if not os.path.isfile(filename) or os.path.getsize(filename)==0:
                #create new file
                print csiRunSym, 'file not found appending data',
                dataNotAppended = True
            else:
                #check csiDataPath3 for last date
                data = pd.read_csv(filename, index_col=0, header=None)
                lastdate=data.index[-1]
                
                symSignalFilesDaily=[x for x in signalFilesDaily if '_'+csiRunSym+'_' in x]
                symSignalFilesMOC=[x for x in signalFilesMOC if '_'+csiRunSym+'_' in x]
                
                for f in [x for x in symSignalFilesDaily if x not in symSignalFilesMOC]:
                    #is signal file dosen't exist copy a portion of the old one.
                    pd.read_csv(signalPathDaily+f).iloc[-2:].to_csv(signalPathMOC+f, index=False)
                
                #if int(loaddate) > lastdate and int(loaddate) > int(lastsignaldate):
                if int(loaddate) > lastdate:
                    print csiRunSym,'appending.. data has not yet been appended',
                    print 'loaddate', loaddate, '>', 'lastdate', lastdate,
                    #print 'loaddate', loaddate, '>', 'lastdate',lastdate,'lastsignaldate', lastsignaldate
                    dataNotAppended=True
                else:
                    #if int(loaddate) <= lastdate:
                    print csiRunSym,'skipping append.. data has already been appended',
                    #if int(loaddate) <= int(lastsignaldate):
                    #    print csiRunSym,'skipping append.. signal has been generated',
                    print 'loaddate', loaddate, '<', 'lastdate',lastdate
                    dataNotAppended=False
            #append data if M-F, not a holiday and if the data hasn't been appended yet. US MARKETS EST.
            dayofweek = nowDateTime.date().weekday()
            
            if dayofweek<5 and dataNotAppended:
            #if dataNotAppended:
                #append new bar
                runsystem = append_data(ibsym, timetable, loaddate)
                if runsystem:
                    print 'running system',
                    if debug==True:
                        print 'debug mode'
                        popenArgs = ['python', runPath,csiRunSym]
                        #popenArgs2 = ['python', runPath2, csiFileSym,'0']
                        threadlist.append((csiRunSym,popenArgs))
                    else:
                        print 'live mode'
                        popenArgs = ['python', runPath,csiRunSym,'1']
                        #popenArgs2 = ['python', runPath2, csiFileSym,'1']
                        threadlist.append((csiRunSym,popenArgs))
                else:
                    print 'skipping runsystem append_data returned 0, using last saved signals.'
                    popenArgs = ['python', runPath,csiRunSym,'0']
                    #popenArgs2 = ['python', runPath2, csiFileSym,'1']
                    threadlist.append((csiRunSym,popenArgs))
            else:
                if dayofweek>=5:
                    print 'skipping append.. day of week', days[dayofweek]

                
        else:
            print csiRunSym,'not triggered: next trigger',tdate,'now', nowDateTime
            
    return threadlist
                
def append_data(sym, timetable, loaddate):
    global feeddata
    global execDict
    global client
    global csiDataPath
    global csiDataPath3
    global IB2CSI_multiplier_adj
    
    if sym in IB2CSI_multiplier_adj.keys():
        mult=IB2CSI_multiplier_adj[sym]
    else:
        mult=1
    #datafiles = os.listdir(csiDataPath)
    fmt = '%Y-%m-%d %H:%M'
    #create new bar
    data = refresh_history(sym, execDict)
    data.index = data.index.to_datetime()
    opentime = dt.strptime(timetable[loaddate][sym+' open'],fmt)
    closetime = dt.strptime(timetable[loaddate][sym+' close'],fmt)
    mask = (data.index >= opentime) & (data.index <= closetime)
    data2=data.ix[mask]
    if data2.shape[0]>0:
        newbar = pd.DataFrame({}, columns=['Date', 'Open','High','Low','Close','Volume','OI','R','S']).set_index('Date')
        newbar.loc[loaddate] = [data.Open[0]*mult,max(data2.High)*mult, min(data2.Low)*mult,data2.Close[-1]*mult,data2.Volume.sum(),np.nan,np.nan,np.nan]
        #load old bar
        csisym=feeddata.ix[sym].CSIsym2
        filename = csiDataPath+csisym+'_B.CSV'
        if os.path.isfile(filename):
            csidata=pd.read_csv(filename, index_col=0, header=None)
            csidata.index.name = 'Date'
            csidata.columns = ['Open','High','Low','Close','Volume','OI','R','S']
            filename = csiDataPath3+csisym+'_B.CSV'
            csidata.append(newbar).fillna(method='ffill').to_csv(filename, header=False, index=True)
            print 'saved', filename,
            print 'appended data between', opentime, closetime,
            return True
        else:
            print filename, 'not found..',
            return False
    else:
        print 'no data found between', opentime, closetime,
        return False

def getIBopen():
    global csidate
    openOrders = pd.DataFrame(client.get_open_orders()).transpose()
    if len(openOrders) !=0:
        #openOrders=openOrders.set_index('orderid')
        openOrders['contract']=[openOrders.ix[i].symbol+openOrders.ix[i].expiry for i in openOrders.index]
        openOrders=openOrders.set_index('contract')
        openOrders['Date']=csidate
        openOrders['timestamp']=int(calendar.timegm(dt.utcnow().utctimetuple()))
        try:
            tablename='ib_openorders'
            mode = get_write_mode(writeConn, tablename, openOrders)
            openOrders.to_sql(name=tablename, con=writeConn, index=True, if_exists=mode, index_label='contract')
        except Exception as e:
            #print e
            traceback.print_exc()
        return openOrders.drop(['Date','timestamp'],axis=1)
    else:
        return 0

def updateWithOpen(iborders, cid):
    print 'checking IB open orders..'
    openOrders=getIBopen()
    #filter out openorders from prior clients

    #check open orders
    iborders_lessOpen=[]
    if isinstance(openOrders, type(pd.DataFrame())) and len(openOrders>0):
        openOrders=openOrders[openOrders.clientid==cid].copy()
        for (contract,[order,qty]) in iborders:
            #print sym, order, qty
            if order == 'SLD':
                order2='SELL'
            else:
                order2='BUY'
                
            if contract in openOrders.index:
                print 'open order found..',contract, order, qty, openOrders.ix[contract].side, openOrders.ix[contract].qty,
                if openOrders.ix[contract].side==order2 and openOrders.ix[contract].qty == qty:
                    print 'OK: removing order from iborders'
                    openOrders = openOrders.drop(contract, axis=0)
                else:
                    print 'Error: mismatch!'
                    iborders_lessOpen+=[(contract,[order,qty])]
            else:
                print 'Error: Open Order for',contract, order, qty,'Not found!! Check IB..'
                iborders_lessOpen+=[(contract,[order,qty])]
                
        for contract in openOrders.index:
            if contract not in [x[0] for x in iborders]:
                print 'Found open order not in execDict!\n', openOrders.ix[contract]
                
        #if len(openOrders)>0:
        #    print 'Open orders that were not found in execDict:'
        #    print openOrders
        #    print 'Cancel these orders if immediate == True?'
    else:
        print 'no open orders found'
        iborders_lessOpen=iborders
        
    #if execDict == None:
    return iborders_lessOpen
    #else:
    #return iborders_lessOpen, { key: execDict[key] for (key,lst) in iborders_lessOpen }
    
def checkIBpositions(account='v4futures'):
    ordersDF=pd.read_sql('select * from (select * from %s\
            order by timestamp ASC) group by CSIsym' % (account+'_live'),\
            con=readConn,  index_col='ibsym')
    ordersDF['ibqty']=ordersDF.signal * ordersDF.c2qty
    #errors=check_systems_live(debug, ordersDict, csidate)
    #print 'total c2 errors found', errors
    
    print 'checking ib positions..'
    raw_portfolio = get_ibfutpositions(portfolioPath)
    if not isinstance(raw_portfolio, type(pd.DataFrame())):
        print 'IB returned no positions. creating zero portfolio.'
        columns=['contracts','exp', 'qty','price','value','avg_cost','unr_pnl','real_pnl','accountid','currency']
        portfolio = pd.DataFrame(data={}, columns=columns,index=ordersDF.index)
        portfolio.index.name='sym'
        portfolio.qty=0
    else:
        raw_portfolio = raw_portfolio.reset_index().set_index('sym')
        portfolio=raw_portfolio[raw_portfolio.qty != 0].copy()
        print portfolio.shape[0],'futures positions found'
        #print portfolio
        port2=portfolio.reset_index().groupby(['sym'])['exp'].max()
        index=[sym+str(port2.ix[sym]) for sym in port2.index]
        portfolio=portfolio.reset_index().set_index('contracts').ix[index].set_index('sym').copy()
        #portfolio.to_csv('portfolio.csv')
        ##if all executions went through properly all contract should be current. 
        #portfolio = portfolio.reset_index().groupby(portfolio.index)[['qty']].sum()
        
    errors=0
    adj_syms=[]
    for sym in ordersDF.index:
        sysqty = ordersDF.ix[sym].ibqty
        text=''
        if sym in portfolio.index:
            ibqty = portfolio.ix[sym].qty
            if sysqty == ibqty:
                text+= 'OK'
            else:
                text+= 'ERROR '+str(sysqty-ibqty)+ ' adjustment needed'
                adj_syms.append((sym, sysqty-ibqty))
                errors+=1
        else:
            if sysqty==0:
                text+= 'OK'
            else:
                text+= 'ERROR '+str(sysqty)+' adjustment needed'
                adj_syms.append((sym, sysqty))
                errors+=1
        print sym,text
        portfolio.set_value(sym,'status',text)
    
    if errors>0:
        print 'total ib order errors', errors, 'checking open orders.'
        openorders=getIBopen()
        if openorders is not 0:
            openorders = openorders.reset_index().set_index('symbol')
            openorders['orderqty']=np.where(openorders.side=='SELL',openorders.qty*-1, openorders.qty)
            for sym, adjqty in adj_syms:
                if sym in openorders.index and adjqty==openorders.ix[sym].orderqty:
                    #print sym, adjqty, openorders.ix[sym].orderqty
                    errors-=1
                    text='OK: open order found'
                    print sym, text
                    portfolio.set_value(sym,'status',text)
        else:
            print 'no open orders found. check IB.'
            
    portfolio['bet']=ordersDF.selection[0]
    portfolio['ordertype']=ordersDF.ordertype[0]
    portfolio['Date']=ordersDF.Date[0]
    portfolio['timestamp']=int(calendar.timegm(dt.utcnow().utctimetuple()))
    tablename='checkSystems_ib_'+account
    mode = get_write_mode(writeConn, tablename, portfolio)
    portfolio.to_sql(name=tablename,con=writeConn, index=True, if_exists=mode, index_label='ibsym')
    print 'saved', tablename,'to',dbPath,'writemode',mode
    return errors, portfolio
    
if __name__ == "__main__":
    print 'IB get history seetings:', durationStr, barSizeSetting, whatToShow
    #feedfile='D:/ML-TSDP/data/systems/system_ibfeed.csv'
    feeddata=pd.read_csv(feedfile,index_col='ibsym')
    #update margin function here
    feeddata['Date']=csidate
    feeddata['timestamp']=int(calendar.timegm(dt.utcnow().utctimetuple()))
    feeddata.to_sql(name='feeddata', con=writeConn, index=True, if_exists='replace', index_label='IBsym')

    #threadlist = [(feeddata.ix[x].CSIsym,x) for x in feeddata.index]
    #systemfile=systemPath+'system_v4futures_live.csv'
    #load last systemfile from vol_adjsize csi
    #systemdata=pd.read_sql('select * from v4futures where timestamp=\
    #                        (select max(timestamp) from v4futures as maxtimestamp)', con=readConn)
    #systemdata['c2sym2']=[x[:-2] for x in systemdata.c2sym]
    #systemdata['CSIsym']=[x.split('_')[1] for x in systemdata.System]
    #systemdata = systemdata.set_index('CSIsym')
    #systemdata = systemdata.ix[feeddata.CSIsym.tolist()]
    #systemdata = systemdata.reset_index()

    futuresDF_results=pd.read_sql('select * from futuresDF_results where timestamp=\
        (select max(timestamp) from futuresDF_results)', con=readConn,  index_col='CSIsym')
    futuresDF = futuresDF_results.ix[feeddata.CSIsym.tolist()].copy()
    futuresDF = futuresDF.reset_index()
    futuresDF.rename(columns=lambda x: x.replace('Contract', 'c2sym'), inplace=True)
    futuresDF['c2sym2']=[x[:-2] for x in futuresDF.c2sym]
    
    execDict={}
    contractsDF=pd.DataFrame()
    tries = 0
    while (len(execDict)  == 0 or len(contractsDF) == 0) and tries<5:
        try:
            execDict, contractsDF, futuresDF=create_execDict(feeddata, futuresDF)
        except Exception as e:
            #print e
            slack.notify(text='create_execDict: '+str(e), channel=slack_channel, username="ibot", icon_emoji=":robot_face:")
            traceback.print_exc()
            tries+=1
            if tries==5:
                sys.exit('failed 5 times to get contract info')
    
    if get_lastquotes:
        try:
            refresh_all_histories(execDict)
        except Exception as e:
            #print e
            slack.notify(text='get_timetable: '+str(e), channel=slack_channel, username="ibot", icon_emoji=":robot_face:")
            traceback.print_exc()
        print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes ', dt.now()
        sys.exit('get_lastquotes')

    if refresh_timetable:
        try:
            timetable = get_timetable(contractsDF)
        except Exception as e:
            #print e
            slack.notify(text='get_timetable: '+str(e), channel=slack_channel, username="ibot", icon_emoji=":robot_face:")
            traceback.print_exc()
        
        #print timetable.to_csv()
        delta=dt.strptime(str(ttdate),'%Y%m%d')-dt.strptime(str(csidate),'%Y%m%d')
        txt = 'Warning! Delta: {}, Last timetable date: {}, Last csidate: {}'.format(delta.days, ttdate, csidate)
        print txt
        if delta.days >1 or (delta.days ==1 and dt.today().day in [2,3,4,5]):
            slack.notify(text=txt, channel=slack_channel, username="ibot", icon_emoji=":robot_face:")

        currentdate=timetable.columns[0]
        print timetable[currentdate]
        if timetable[currentdate].isnull().any():
            #load previous file.
            dates=sorted([int(x.split('.csv')[0]) for x in listdir(timetablePath) if is_int(x.split('.csv')[0])])
            if int(currentdate) in dates:
                dates.remove(int(currentdate))
            lastdate=str(dates[-1])
            src=timetablePath+lastdate+'.csv'
            dest=timetablePath+currentdate+'.csv'
            #copyfile(src,dest)
            timetable2=pd.read_csv(src, index_col=0)
            dates2=[x for x in timetable2.columns if int(x)>=int(currentdate)]
            timetable2=timetable2[dates2]
            if timetable2[currentdate].isnull().any():
                #copy over the next trigger times from the next date.
                print 'copying data from {} to {} for nans'.format(dates2[1], currentdate)
                timetable2[currentdate][timetable[currentdate].isnull()]=timetable[dates2[1]][timetable[currentdate].isnull()]
                if timetable2[currentdate].isnull().any():
                    slack.notify(text='there\'s still nans...', channel=slack_channel, username="ibot", icon_emoji=":robot_face:")
            
            timetable2.to_csv(dest, index=True)
            txt='nans in timetable for {}. Copied {} to {}.'.format(currentdate, src, dest)
            print txt
            slack.notify(text=txt, channel=slack_channel, username="ibot", icon_emoji=":robot_face:")
            print 'saved', dest
            timetable2['Date']=csidate
            timetable2['timestamp']=int(calendar.timegm(dt.utcnow().utctimetuple()))
            try:
                timetable2.to_sql(name='timetable', con=writeConn, index=True, if_exists='replace', index_label='Desc')
            except Exception as e:
                #print e
                slack.notify(text='get_timetable: '+str(e), channel=slack_channel, username="ibot", icon_emoji=":robot_face:")
                traceback.print_exc()


            '''
        #dont do this because it will mess up account values timing.
        ib_portfolio=get_ibfutpositions(portfolioPath)
        print ib_portfolio

        from c2api.get_exec import getSystemDetails
        accountInfo=pd.read_sql('select * from accountInfo where timestamp=\
            (select max(timestamp) from accountInfo as maxtimestamp)', con=readConn,  index_col='index')
        systems = [x for x in accountInfo.columns if x not in ['Date','timestamp']]
        for account in systems:
            c2id=accountInfo[account].c2id
            c2key=accountInfo[account].c2key
            jsondata = json.loads(getSystemDetails(c2id, c2key))
            #logging.info('\n Length jsondata'+str(len(jsondata['equity_data'])))
            equity=json_normalize(jsondata['response']['marginEquityData'])
            if len(equity)>0 and 'modelAccountValue' in equity.columns:
                print account, equity['modelAccountValue'][0]
                equity['system']=account
                equity['Date']=csidate
                equity['timestamp']=int(calendar.timegm(dt.utcnow().utctimetuple()))
                equity.to_sql(name='c2_equity',con=writeConn, index=False, if_exists='append')
                print  'Saved',account,'c2_equity to sql db',dbPath
            else:
                print 'Could not get last equity from c2'
        '''
        print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes ', dt.now()
        sys.exit('refresh_timetable')

    if post_processing:
        skipcheck=True
        print('requesting last executions from IB..')
        executions = filterIBexec()
        if executions is not None:
            print(executions.to_csv())
        else:
            print 'No executions found..'
        errors, portfolio=checkIBpositions()
        print errors, 'errors found'
        print portfolio.to_csv()
        sys.exit('post_processing')
    else:
        skipcheck=False

    if immediate:
        print 'Running Immediate Process...'
        #include all symbols in threadlist to refresh all orders from selection
        threadlist = [(feeddata.ix[x].CSIsym,x) for x in feeddata.index]
    else:
        print 'Running MOC Process...'
        threadlist=find_triggers(feeddata, contractsDF)
        print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes ', dt.now()
        runThreads(threadlist)
        #threadlist = [(feeddata.ix[x].CSIsym,x) for x in feeddata.index]
        
    print 'returned to main thread with', len(threadlist), 'threads'
    print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes ', dt.now()
    

    
    if len(threadlist)>0:
        if immediate:
            print 'running vol_adjsize_immediate to process immediate orders'
            ordersDict = vol_adjsize_immediate(debug, threadlist, skipcheck=skipcheck)
        else:
            print 'running vol_adjsize_moc to update system files'
            ordersDict = vol_adjsize_moc(debug, threadlist)
        #ordersDict={}
        #ordersDict['v4futures']=pd.read_csv(systemfile)[-4:]
        
        if isinstance(ordersDict, type({})):
            #ordersDict['v4futures']=pd.read_csv(systemfile)[-4:]
            if not post_processing:
                totalc2orders, _ =check_systems_live(debug, ordersDict, csidate)
            else:
                totalc2orders=0
            #check ib positions
            try:
                if 'v4futures' in ordersDict.keys():
                    execDict=update_orders(feeddata, ordersDict['v4futures'], execDict, threadlist)
                iborders = [(sym, execDict[sym][:2]) for sym in execDict.keys() if execDict[sym][0] != 'PASS']
                
                #get rid of orders if they are already working orders.
                #dont need this anymore because we reqGlobalCancel
                #iborders, execDict= updateWithOpen(iborders, execDict=execDict)
                num_iborders=len(iborders)
            except Exception as e:
                #print e
                traceback.print_exc()
        else:
            print 'No orders in orderDict'
            totalc2orders =0
            num_iborders=0
    else:
        print 'threadlist returning nothing. skipping vol_adjsize_live, check_systems'
        totalc2orders =0
        num_iborders=0
        
    
    if totalc2orders ==0 and num_iborders==0:
        print 'Found nothing to update!'
    else:
        print 'Found', totalc2orders, 'c2 position adjustments.'
        print 'Found', num_iborders,'ib position adjustments.'
        #send orders if live mode
        if debug==False:
            if submitC2 and totalc2orders !=0:
                print 'Live mode: running c2 orders'
                proc_signal_v4_live(debug, ordersDict)
                print 'returned to main thread, running check systems again..'
                totalc2orders, ordersDictWithErrors=check_systems_live(debug, ordersDict, csidate)
                
                #run orders again if it didn't go through.
                tries = 0
                while totalc2orders>0 and tries<5:
                    tries+=1
                    print '\n\nFound', totalc2orders, 'c2 position adjustments for',ordersDictWithErrors.keys()
                    print 'Running orders.. Attempt: ',tries
                    proc_signal_v4_live(debug, ordersDictWithErrors)
                    totalc2orders, ordersDictWithErrors=check_systems_live(debug, ordersDictWithErrors, csidate)
            else:
                if submitC2==False:
                    print 'skipped c2 orders: submitC2 set to False'
                if totalc2orders ==0:
                    print 'skipped c2 orders: No c2 orders to be processed.'
            
            if submitIB and num_iborders!=0:
                print 'returned to main thread, processing ib orders' 
                try:
                    #clear all orders first
                    print 'requesting global cancel IB orders'
                    client.reqGlobalCancel()
                    sleep(1)
                    #set the client date to timestamp to filter out openorders from prior runs.
                    cid=int(calendar.timegm(dt.utcnow().utctimetuple()))
                    print 'placing IB orders', iborders
                    place_iborders(execDict, cid)
                    #wait for orders to be filled
                    sleep(10)
                    errors, portfolio=checkIBpositions()
                    print errors, 'errors found'
                    print portfolio.to_csv()

                except Exception as e:
                    #print e
                    traceback.print_exc()

            else:
                if submitIB==False:
                    print 'submitIB set to False'
                if num_iborders==0:
                    print 'skipped IB orders: No IB orders to be processed.'
                    
            totalc2orders=int(pd.read_sql('select * from checkSystems', con=readConn).iloc[-1])
            print 'Found', totalc2orders, 'c2 position adjustments.'
        else:
            print 'Debug mode: skipping orders'
    
    print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes ', dt.now()


    
#deprecated
'''
print 'checking executions..'
executions=filterIBexec()
iborders_lessExec=[]
if executions is not None:
    #check executions
    executions2 = executions.reset_index().groupby(['contract','side'])[['qty']].max()
    
    #check if expired contracts have been exited.
    #executions2 = executions.reset_index().groupby(['symbol','side'])[['qty']].max()
    
    for (sym,[order,qty]) in iborders:
        if (sym,order) in executions2.index and executions2.ix[sym].qty[0] ==qty:
            print 'execution found..',sym, order, qty, executions2.ix[sym].index[0],  executions2.ix[sym].qty[0]
            #execDict[sym][0] = 'PASS'
            #execDict[sym][1] = 0
        else:
            print 'There was an error:',sym,'order',  execDict[sym][:2], 
            if sym in executions2.index:
                print 'ib returned', executions2.ix[sym].index[0], executions2.ix[sym].qty[0]
            else:
                print 'ib execution not found'
            iborders_lessExec+=[(sym,[order,qty])]
            
    #num_iborders=len([execDict[sym][0] for sym in execDict.keys() if execDict[sym][0] != 'PASS'])
    print 'Found', len(iborders_lessExec),'ib position adjustments after placing orders.'
    print iborders_lessExec
else:
    print 'executions returned None. IB orders could not be verified:'
    print iborders

if len(iborders_lessExec)>0:
    print 'checking open orders to see if orders not executed are pending orders..'
    iborders_lessOpen=updateWithOpen(iborders_lessExec, cid)
'''
'''
#debug order executions
systemfile='D:/ML-TSDP/data/systems/system_v4futures_live.csv'
feeddata=pd.read_csv(feedfile,index_col='ibsym')
#systemfile=systemPath+'system_'+sys+'_live.csv'
execDict=create_execDict(feeddata, systemfile)
try:
    execDict=update_orders(feeddata, systemfile, execDict)
except Exception as e:
    #print e
    traceback.print_exc()
iborders = [(sym, execDict[sym][:2]) for sym in execDict.keys() if execDict[sym][0] != 'PASS']
num_iborders=len([execDict[sym][0] for sym in execDict.keys() if execDict[sym][0] != 'PASS'])
executions=filterIBexec()

executions=filterIBexec()
executions2 = executions.reset_index().groupby(['symbol','side'])[['qty']].max()

'''

#place_iborders(execDict)
#contractsDF = get_contractdf(execDict, systemPath)
#timetable= get_timetable(execDict, systemPath)

#client.get_realtimebar(contracts[0], tickerId, whatToShow, data, filename)
#client.get_history(endDateTime, execDict['LE'][2], whatToShow, data ,filename,tickerId, minDataPoints, durationStr, barSizeSetting, formatDate=1)
