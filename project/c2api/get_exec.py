import requests
from time import gmtime, strftime, time, localtime, sleep
import json
from pandas.io.json import json_normalize
import numpy as np
import pandas as pd
import time
import logging
import os
import requests
import sqlite3



        
def place_order(dbPath, action, quant, sym, type, systemid, submit,apikey, parentsig=None):
    #conn = sqlite3.connect(dbPath)
    #sigid=int(pd.read_sql('select * from c2sigid', con=conn).iloc[-1])
    if submit == False:
        return 0;
    url = 'https://collective2.com/world/apiv2/submitSignal'
    
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
    #sigid=int(sigid)+1
    parentsig= "" if parentsig==None else parentsig
    data = { 
    		"apikey":   apikey, # "tXFaL4E6apdfmLtGasIovtGnUDXH_CQso7uBpOCUDYGVcm1w0w", 
    		"systemid": str(systemid), 
    		"signal":{ 
    	   		"action": action, 
    	   		"quant": quant, 
    	   		"symbol": sym, 
    	   		"typeofsymbol": type, 
    	   		"market": 1, 	#"limit": 31.05, 
    	   		"duration": "DAY", 
               "signalid": "",
               "conditionalupon": parentsig
    		} 
    	}
    #logging.info( 'sigid is: ' + str( sigid ))
    #dataf=pd.DataFrame([[sigid]], columns=['sigid'])
    #dataf.to_sql(name='c2sigid',con=conn,if_exists='replace', index=False)
    params={}
    
    r=requests.post(url, params=params, json=data);
    #sleep(2)
    print r.text
    logging.info( str(r.text)  )
    return r.json()['signalid']

def place_order2(dbPath, action, quant, sym, type, systemid, submit,apikey, parentsig=None):
    conn = sqlite3.connect(dbPath)
    sigid=int(pd.read_sql('select * from c2sigid', con=conn).iloc[-1])
    if submit == False:
        return 0;
    url = 'https://collective2.com/world/apiv2/submitSignal'
    
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
    sigid=int(sigid)+1
    
    data = { 
    		"apikey":   apikey, # "tXFaL4E6apdfmLtGasIovtGnUDXH_CQso7uBpOCUDYGVcm1w0w", 
    		"systemid": str(systemid), 
    		"signal":{ 
    	   		"action": action, 
    	   		"quant": quant, 
    	   		"symbol": sym, 
    	   		"typeofsymbol": type, 
    	   		"market": 1, 	#"limit": 31.05, 
    	   		"duration": "DAY", 
               "signalid": sigid,
               "conditionalupon": parentsig
    		} 
    	}
    logging.info( 'sigid is: ' + str( sigid ))
    dataf=pd.DataFrame([[sigid]], columns=['sigid'])
    dataf.to_sql(name='c2sigid',con=conn,if_exists='replace', index=False)
    params={}
    
    r=requests.post(url, params=params, json=data);
    #sleep(2)
    #print r.text
    logging.info( str(r.text)  )
    return str(r.text)
    
def set_position(positions, systemid, submit, apikey):
    if submit == False:
        return 0;
    url = 'https://api.collective2.com/world/apiv3/setDesiredPositions'
    
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
    
    data = { 
          "apikey":   apikey, # "tXFaL4E6apdfmLtGasIovtGnUDXH_CQso7uBpOCUDYGVcm1w0w", 
          "systemid": str(systemid), 
          "positions": positions
          #{
          #   "symbol" : "MSFT",
          #   "typeofsymbol" : "stock",
          #   "quant" : -30
          #},
          # {
          #    "symbol" : "@ESH6",
          #    "typeofsymbol" : "future",
          #   "quant" : 1
          # }
          #
          #
    }
    
    params={}
    
    r=requests.post(url, params=params, json=data);
    sleep(2)
    #print r.text
    logging.info( str(r.text)  )
    return str(r.text)
    
def flat_position(systemid, submit, apikey):
    if submit == False:
        return 0;
    url = 'https://api.collective2.com/world/apiv3/setDesiredPositions'
    
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
    
    data = { 
          "apikey":   apikey, # "tXFaL4E6apdfmLtGasIovtGnUDXH_CQso7uBpOCUDYGVcm1w0w", 
          "systemid": str(systemid), 
          "positions": [
					{ 
						"symbol"  : "flat",
						"typeofsymbol"  : "flat",
						"quant"  : "0"
					}

				]
          #{
          #   "symbol" : "MSFT",
          #   "typeofsymbol" : "stock",
          #   "quant" : -30
          #},
          # {
          #    "symbol" : "@ESH6",
          #    "typeofsymbol" : "future",
          #   "quant" : 1
          # }
          #
          #
    }
    
    params={}
    
    r=requests.post(url, params=params, json=data);
    sleep(2)
    #print r.text
    logging.info( str(r.text)  )
    return str(r.text)
    
def stop_position(systemid, submit, apikey):
    if submit == False:
        return 0;
    url = 'https://api.collective2.com/world/apiv3/setDesiredPositions'
    
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
    
    data = { 
          "apikey":   apikey, # "tXFaL4E6apdfmLtGasIovtGnUDXH_CQso7uBpOCUDYGVcm1w0w", 
          "systemid": str(systemid), 
          "positions":	[
					{ 
						"symbol"			: "stop",
						"typeofsymbol"	: "stop",
						"quant"			: "0"
					}

				]
          #{
          #   "symbol" : "MSFT",
          #   "typeofsymbol" : "stock",
          #   "quant" : -30
          #},
          # {
          #    "symbol" : "@ESH6",
          #    "typeofsymbol" : "future",
          #   "quant" : 1
          # }
          #
          #
    }
    
    params={}
    
    r=requests.post(url, params=params, json=data);
    sleep(2)
    #print r.text
    logging.info( str(r.text)  )
    return str(r.text)
    
def retrieveSignalsWorking(systemid, apikey):
    url = 'https://collective2.com/world/apiv3/retrieveSignalsWorking'
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
    
    data={
        "systemid" : str(systemid),
        "apikey" : str(apikey)
        }
    
    params={}
    
    r=requests.post(url, params=params, json=data);
    #print r.text
    logging.info(r.text)
    return r.text
    
def retrieveSystemEquity(systemid, apikey, commission_plan='default'):
    url = 'https://api.collective2.com/world/apiv3/retrieveSystemEquity'
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
    
    data={
       "commission_plan" : commission_plan,
       "systemid" : str(systemid),
       "apikey" : str(apikey)
        }
    
    params={}
    
    r=requests.post(url, params=params, json=data);
    #print r.text
    logging.info(r.text)
    return r.text
    
def getSystemDetails(systemid, apikey):
    url = 'https://api.collective2.com/world/apiv3/getSystemDetails'
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
    
    data={
       #"commission_plan" : commission_plan,
       "systemid" : str(systemid),
       "apikey" : str(apikey)
        }
    
    params={}
    
    r=requests.post(url, params=params, json=data);
    #print r.text
    logging.info(r.text)
    return r.text
    
def get_exec(systemid, apikey):
    url = 'https://collective2.com/world/apiv3/requestTrades'
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
    data = { 
    		"apikey":   str(apikey),#"tXFaL4E6apdfmLtGasIovtGnUDXH_CQso7uBpOCUDYGVcm1w0w", 
    		"systemid": str(systemid)
          }
    params={}
    r=requests.post(url, params=params, json=data);
    #print r.text
    logging.info(r.text)
    return r.text


def get_exec_open(systemid, apikey):
    url = 'https://collective2.com/world/apiv3/requestTradesOpen'
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
    data = { 
    		"apikey":   str(apikey),    #"tXFaL4E6apdfmLtGasIovtGnUDXH_CQso7uBpOCUDYGVcm1w0w", 
    		"systemid": str(systemid)
          }
    
    params={}
    
    r=requests.post(url, params=params, json=data);
    #print r.text
    logging.info(r.text)
    return r.text

def get_c2equity(systemdata):
    logging.info('GETTING C2 EQUITY...')
    c2list=get_c2_list(systemdata)
    systems=c2list.keys()
    for systemname in systems:
        (systemid, apikey)=c2list[systemname]
        #c2list[systemname]=get_c2livepos(systemid, apikey, systemname)
        equityData=retrieveSystemEquity(systemid, apikey)
        jsondata = json.loads(equityData)
        logging.info('\n Length jsondata'+str(len(jsondata['equity_data'])))
        dataSet=json_normalize(jsondata['equity_data'])
        dataSet.index = pd.to_datetime(dataSet['unix_timestamp'].astype(int),unit='s')
        dataSet.index.name = 'timestamp'
        filename = './data/portfolio/c2_' + systemname + '_equity.csv'
        dataSet.to_csv(filename)
        print 'saved', filename
    return dataSet

def get_c2lastEquity(systemdata):
    #logging.info('GETTING C2 EQUITY...')
    c2list=get_c2_list(systemdata)
    systems=c2list.keys()
    for systemname in systems:
        (systemid, apikey)=c2list[systemname]
        #c2list[systemname]=get_c2livepos(systemid, apikey, systemname)
        equityData=getSystemDetails(systemid, apikey)
        jsondata = json.loads(equityData)
        #logging.info('\n Length jsondata'+str(len(jsondata['equity_data'])))
        dataSet=json_normalize(jsondata['response']['marginEquityData'])
        #dataSet.index = pd.to_datetime(dataSet['unix_timestamp'].astype(int),unit='s')
        #dataSet.index.name = 'timestamp'
        filename = './data/portfolio/c2_' + systemname + '_lastEquity.csv'
        dataSet.to_csv(filename, index=False)
        print 'saved', filename
    return dataSet
    
def get_c2pos(systemdata):
    logging.info('GETTING C2 POSITIONS...')
    c2list=get_c2_list(systemdata)
    systems=c2list.keys()
    for systemname in systems:
        (systemid, apikey)=c2list[systemname]
        c2list[systemname]=get_c2livepos(systemid, apikey, systemname)
        #equityData=retrieveSystemEquity(systemid, apikey)
        #jsondata = json.loads(equityData)
        #logging.info('\n Length jsondata'+str(len(jsondata['equity_data'])))
        #dataSet=json_normalize(jsondata['equity_data'])
        #dataSet.index = pd.to_datetime(dataSet['unix_timestamp'].astype(int),unit='s')
        #dataSet.index.name = 'timestamp'
        #dataSet.to_csv('./data/portfolio/c2_' + systemname + '_equity.csv')
    return c2list
    


def get_c2livepos(systemid, apikey, systemname):
    data=get_exec_open(systemid,apikey);
    #print systemid, apikey, data
    jsondata = json.loads(data)
    #logging.info('\n Length jsondata'+str(len(jsondata['response'])))
    
    if len(jsondata['response']) > 0:
    #if 'response' in jsondata:
        dataSet=json_normalize(jsondata['response'])
        dataSet=dataSet.set_index('symbol')
        filename='./data/portfolio/c2_' + systemname + '_portfolio.csv'
        dataSet.to_csv(filename)
        print 'saved', filename
        return dataSet
    else:
        return pd.DataFrame()
        
        
        
def get_c2pos_from_csv(systemname):
    dataSet = pd.read_csv('./data/portfolio/c2_' + systemname + '_portfolio.csv', index_col='symbol')
    return dataSet
    


def get_c2_portfolio(systemname):
    filename='./data/portfolio/c2_' + systemname + '_portfolio.csv'
     
    if os.path.isfile(filename):
        dataSet = pd.read_csv(filename, index_col='symbol')
        if 'PurePL' not in dataSet:
            dataSet['PurePL']=0
        return dataSet
        
    else:
        dataSet=pd.DataFrame({},columns=['symbol','open_or_closed','long_or_short','quant_opened', 'quant_closed', \
                     'opening_price_VWAP', 'closing_price_VWAP', 'PL', 'PurePL', 'commission', \
                     'trade_id','closeVWAP_timestamp','closedWhen','closedWhenUnixTimeStamp', \
                     'expir','instrument',\
                     'markToMarket_time','openVWAP_timestamp','openedWhen','qty'])
        dataSet = dataSet.set_index('symbol')
        dataSet.to_csv(filename)
        print 'saved', filename
        return dataSet

    
def get_c2_pos(systemname, c2sym):
    
    portfolio_data=get_c2_portfolio(systemname)
    portfolio_data=portfolio_data.reset_index()
    sym_cur=c2sym
    portfolio_data=portfolio_data.set_index('symbol')
    if sym_cur not in portfolio_data.index.values:
       return 0
    else:
        c2_pos=portfolio_data.loc[sym_cur]
        c2_pos_qty=float(c2_pos['quant_opened']) - float(c2_pos['quant_closed'])
        c2_pos_side=str(c2_pos['long_or_short'])
        if c2_pos_side == 'short':
            c2_pos_qty=-abs(c2_pos_qty)
        return c2_pos_qty
    
#place_order('BTO','1','EURUSD','forex')
def get_c2_list(systemdata):
    c2list=dict()
    
    #systemdata=pd.read_csv('./data/systems/system.csv')
    #systemdata=systemdata.reset_index()
    for i in systemdata.index:
        system=systemdata.ix[i]
        if system['c2submit']:
            c2list[system['Name']]=[system['c2id'], system['c2api']]
        
    return c2list
    
def get_executions(data):        
    #data=pd.read_csv('./data/systems/system.csv')
    #data=data.reset_index()

    c2dict={}
    for i in data.index:
        system=data.ix[i]
        #print system['Name'] + ' ' + str(system['c2submit'])
        if system['c2submit']:
                c2dict[system['c2id']]=(system['Name'],system['c2api'])



    for c2id in c2dict:
        (stratName,c2api)=c2dict[c2id]
        get_c2trades(c2id, stratName, c2api)
        
def get_c2trades(systemid, name, c2api):

    filename='./data/portfolio/c2_' + name + '_trades.csv'
    
    datestr=strftime("%Y%m%d", localtime())
    data=get_exec(systemid,c2api);
    
    jsondata = json.loads(data)
    if len(jsondata['response']) > 1:
        dataSet=json_normalize(jsondata['response'])
        dataSet=dataSet.set_index('trade_id')
        '''
        if os.path.isfile(filename):
            existData = pd.read_csv(filename, index_col='trade_id')
            existData = existData.reset_index()
            dataSet   =   dataSet.reset_index()
            dataSet=existData.append(dataSet)
            dataSet['trade_id'] = dataSet['trade_id'].astype('int')
            dataSet=dataSet.drop_duplicates(subset=['trade_id'],keep='last')
            dataSet=dataSet.set_index('trade_id') 
        '''
        dataSet=dataSet.sort_values(by='closedWhenUnixTimeStamp')
        
        dataSet.to_csv(filename)
        print 'saved', filename