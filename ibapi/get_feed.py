from wrapper_v4 import IBWrapper, IBclient
from swigibpy import Contract 
import time
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
import seitoolz.bars as bars
from dateutil.parser import parse
import os
import random

currencyPairsDict=dict()
prepData=dict()
tickerId=random.randint(100,9999)

callback = IBWrapper()
client=IBclient(callback)

def reconnect_ib():
    global callback
    global client
    client.tws.eDisconnect()
    callback = IBWrapper()
    client=IBclient(callback)

def get_bar(symbol):
    return client.get_bar(str(symbol))
    
def get_ask(symbol):
    global client
    ask=client.get_IBAsk(str(symbol))
    return ask
    
def get_bid(symbol):
    global client
    return client.get_IBBid(str(symbol))
   
def get_feed(contract, tickerId):
    global tickerid, client, callback
    client.get_IB_market_data(contract, tickerId)

def get_realtimebar(ibcontract, tickerid, whatToShow, data, filename):
    global client, callback
    client.get_realtimebar(ibcontract, tickerid, whatToShow, data, filename)

def proc_history(tickerId, contract, data, barSizeSetting):
    global client, callback
    return client.proc_history(tickerId, contract, data, barSizeSetting)

def get_history(date, contract, whatToShow, data,filename, tickerId, minDataPoints, durationStr, barSizeSetting):
        symbol= contract.symbol
        currency=contract.currency
        ticker=symbol        
        if contract.secType == 'CASH':
            ticker=symbol+currency
            print "Requesting history for: " + ticker + " ending: " + date
            mydata=data
            #for date in getHistLoop:
            if data.shape[0] < minDataPoints:
                count=0
                finished=False
                while not finished:    
                    data = client.get_history(date, contract, whatToShow, data,filename,tickerId, minDataPoints, durationStr, barSizeSetting)
                    data = get_bar(ticker)
                    if data.shape[0] > 0:
                        logging.info("Received Date: " + str(data.index[0]) + " " + str(data.shape[0]) + " records out of " + str(minDataPoints) )
                        date = data.sort_index().index.to_datetime()[0]
                        #eastern=timezone('US/Eastern')
                        #date=date.astimezone(eastern)
                        date=date.strftime("%Y%m%d %H:%M:%S EST")
                        time.sleep(30)
                        if int(data.shape[0]) > int(minDataPoints):
                            finished=True
                    else:
                        count=count + 1
                        if count > 10:
                            finished=True
                            return mydata
            else:
    
                data = client.get_history(date, contract, whatToShow, data,filename,tickerId, minDataPoints, durationStr, barSizeSetting)
                data = get_bar(ticker)
                if data.shape[0] > 0:
                    logging.info("Received Date: " + str(data.index[0]) )
                    #update date
                    date = data.sort_index().index.to_datetime()[0]
                    #eastern=timezone('US/Eastern')
                    #date=date.astimezone(eastern)
                    date=date.strftime("%Y%m%d %H:%M:%S EST")
                    
                    time.sleep(30)
                    return data
                else:
                    return mydata
                
            #set date as last index for next request
    
        return data
        
def get_TickerId(symbol):
    global tickerId
    global currencyPairsDict
    if not currencyPairsDict.has_key(symbol):
            tickerId=tickerId+1
            currencyPairsDict[symbol] = tickerId
    return currencyPairsDict[symbol]

def get_TickerDict():
    global currencyPairsDict
    return currencyPairsDict
    
def cache_bar_csv(dataPath, barSizeSetting, symfilter=''):
    global prepData
    symbols=bars.get_contracts()
    for contract in symbols:
        symbol=contract.symbol
        if contract.secType == 'CASH':
            symbol = contract.symbol + contract.currency
            pair=symbol
            print pair
            if len(symfilter) == 0 or pair == symfilter:
                logging.info( 'Reading Existing Data For ' + symbol )
                interval=duration_to_interval(barSizeSetting)
                filename=dataPath+interval+'_'+pair+'.csv'
                pair=symbol
                tickerId=get_TickerId(pair)
                data = pd.DataFrame({}, columns=['Date','Open','High','Low','Close','Volume']).set_index('Date')
                   
                if os.path.isfile(filename):           
                    data=pd.read_csv(filename, index_col='Date')
                    data=proc_history(tickerId, contract, data, barSizeSetting)
                else:
                    data.to_csv(filename)
                
                prepData[pair]=data
                logging.info( 'Done Reading Existing Data For ' + pair )
    return prepData

def get_bar_bidask(symfilter = ''):
    global tickerId
    symbols=bars.get_contracts()
    for contract in symbols:
        pair=contract.symbol
        if contract.secType == 'CASH':
            pair=contract.symbol + contract.currency 
            if len(symfilter) == 0 or pair == symfilter:
                logging.info(  'Subscribing Bid/Ask to ' + pair  )
                eastern=timezone('US/Eastern')
                endDateTime=dt.now(get_localzone())
                date=endDateTime.astimezone(eastern)
                date=date.strftime("%Y%m%d %H:%M:%S EST")
                tickerId=get_TickerId(pair)
                get_feed(contract, tickerId)
                logging.info( 'Done Subscribing to ' + pair  )
        
def get_bar_feed(dataPath, whatToShow, barSizeSetting, symfilter=''):
    global prepData
    symbols=bars.get_contracts()
    for contract in symbols:
        pair=contract.symbol
        if contract.secType == 'CASH':
            pair = contract.symbol + contract.currency
            if len(symfilter) == 0 or pair == symfilter:
                logging.info(  'Subscribing Realtime Bar to ' + pair  )
                interval=duration_to_interval(barSizeSetting)
                filename=dataPath+interval+'_'+pair+'.csv'
                tickerId=get_TickerId(pair)          
                get_realtimebar(contract, tickerId, whatToShow, prepData[pair], filename)
                logging.info( 'Done Subscribing to ' + pair  )
            
def duration_to_interval(duration):
    if duration == '1 min':
        return '1 min'
    elif duration == '30 mins':
        return '30m'
    elif duration == '10 mins':
        return '10m'
    elif duration == '1 hour':
        return '1h'
    elif duration == '1 day':
        return '1d'
        
def interval_to_ibhist_duration(interval):
    durationStr='1 D'
    barSizeSetting='1 min'
    if interval == '1m':
        durationStr='1 D'
        barSizeSetting='1 min'
    elif interval == '30m':
        durationStr='30 D'
        barSizeSetting='30 mins'
    elif interval == '10m':
        durationStr='10 D'
        barSizeSetting='10 mins'
    elif interval == '1h':
        durationStr='30 D'
        barSizeSetting='1 hour'
    elif interval == '1d':
        durationStr='30 D'
        barSizeSetting='1 hour'
    whatToShow='MIDPOINT'
    return (durationStr, barSizeSetting, whatToShow)


        
def get_bar_date(barSizeSetting, date):
        interval=duration_to_interval(barSizeSetting)
        timestamp = time.mktime(date.timetuple())
        if interval == '30m':
            mins=int(datetime.datetime.fromtimestamp(
                        int(timestamp)
                    ).strftime('%M'))
            if mins < 30:
                #time
                date=datetime.datetime.fromtimestamp(
                            int(timestamp)
                        ).strftime('%Y%m%d  %H:00:00') 
            else:
                 date=datetime.datetime.fromtimestamp(
                            int(timestamp)
                        ).strftime('%Y%m%d  %H:30:00') 
        elif interval == '10m':
            mins=int(datetime.datetime.fromtimestamp(
                        int(timestamp)
                    ).strftime('%M'))
            if mins < 10:
                #time
                date=datetime.datetime.fromtimestamp(
                            int(timestamp)
                        ).strftime('%Y%m%d  %H:00:00') 
            elif mins < 20:
                #time
                date=datetime.datetime.fromtimestamp(
                            int(timestamp)
                        ).strftime('%Y%m%d  %H:10:00') 
            elif mins < 30:
                #time
                date=datetime.datetime.fromtimestamp(
                            int(timestamp)
                        ).strftime('%Y%m%d  %H:20:00') 
            elif mins < 40:
                #time
                date=datetime.datetime.fromtimestamp(
                            int(timestamp)
                        ).strftime('%Y%m%d  %H:30:00') 
            elif mins < 50:
                #time
                date=datetime.datetime.fromtimestamp(
                            int(timestamp)
                        ).strftime('%Y%m%d  %H:40:00') 
            else:
                 date=datetime.datetime.fromtimestamp(
                            int(timestamp)
                        ).strftime('%Y%m%d  %H:50:00') 
        elif interval == '1h':
            date=datetime.datetime.fromtimestamp(
                    int(timestamp)
                ).strftime('%Y%m%d  %H:00:00') 
        return date
        
def get_bar_hist(dataPath, whatToShow, minDataPoints, durationStr, barSizeSetting, symfilter=''):
    global tickerId
    global currencyPairsDict
    symbols=bars.get_contracts()
    for contract in symbols:
        pair=contract.symbol
        if contract.secType == 'CASH':
            pair = contract.symbol + contract.currency
            if len(symfilter) == 0 or pair == symfilter:
                logging.info(  'Getting History for ' + pair  )
                interval=duration_to_interval(barSizeSetting)
                filename=dataPath+interval+'_'+pair+'.csv'
                eastern=timezone('US/Eastern')
                endDateTime=dt.now(get_localzone())
                date=endDateTime.astimezone(eastern)
                #date=date.strftime("%Y%m%d %H:%M:%S EST")
                date=get_bar_date(barSizeSetting, date) + ' EST'
                tickerId=get_TickerId(pair)
                data=get_history(date, contract, whatToShow, prepData[pair], filename, tickerId, minDataPoints, durationStr, barSizeSetting)
        
                logging.info( 'Done Getting History for ' + pair  )
                if len(symfilter) > 0:
                    return data
                
def get_bar_hist_date(date, dataPath, whatToShow, minDataPoints, durationStr, barSizeSetting, symfilter=''):
    global tickerId
    global currencyPairsDict
    symbols=bars.get_contracts()
    for contract in symbols:
        pair=contract.symbol
        if contract.secType == 'CASH':
            pair = contract.symbol + contract.currency
            if len(symfilter) == 0 or pair == symfilter:
                logging.info(  'Getting History for ' + pair  )
                interval=duration_to_interval(barSizeSetting)
                filename=dataPath+interval+'_'+pair+'.csv'
                date=get_bar_date(barSizeSetting, date) + ' EST'
                tickerId=get_TickerId(pair)
                data=get_history(date, contract, whatToShow, prepData[pair], filename, tickerId, minDataPoints, durationStr, barSizeSetting)
                logging.info( 'Done Getting History for ' + pair  )
                if len(symfilter) > 0:
                    return data
            
def check_bar(barSizeSetting, symfilter=''):
    dataPath = './data/from_IB/'
    barPath='./data/bars/'
    interval=duration_to_interval(barSizeSetting)
    
    try:
        count=0
        symbols=bars.get_contracts()
        for contract in symbols:
            pair=contract.symbol
            if contract.secType == 'CASH':
                pair = contract.symbol + contract.currency
                if len(symfilter) == 0 or pair == symfilter:
                    dataFile=dataPath + interval + '_' + pair + '.csv'
                    barFile=barPath + pair + '.csv'
                    
                    if os.path.isfile(dataFile) and os.path.isfile(barFile):
                        #data=pd.read_csv(dataFile, index_col='Date')
                        bar=pd.read_csv(barFile, index_col='Date')
                        eastern=timezone('US/Eastern')
                        
                        #timestamp
                        #dataDate=parse(data.index[-1]).replace(tzinfo=eastern)
                        nowDate=datetime.datetime.now(get_localzone()).astimezone(eastern)
                        if bar.shape[0] > 0:
                            barDate=parse(bar.index[-1]).replace(tzinfo=eastern)         
                        else:
                            barDate=datetime.date(2000,1,1)
                        #dtimestamp = time.mktime(dataDate.timetuple())
                        btimestamp = time.mktime(barDate.timetuple())
                        timestamp=time.mktime(nowDate.timetuple()) + 3600
                        checktime = 3
                        
                        checktime = checktime * 60
                        logging.info(pair + ' Feed Last Received ' + str(round((timestamp - btimestamp)/60, 2)) + ' mins ago')
                            
                        if timestamp - btimestamp > checktime:
                            logging.error(pair + ' Feed not being received for ' + str(round((timestamp - btimestamp)/60, 2))) + ' mins'
                            if len(symfilter) > 0:
                                return False
                            count = count + 1
        if count > 5:
                return False
        else:
                return True
    except Exception as e:
            logging.error("check_bar", exc_info=True)