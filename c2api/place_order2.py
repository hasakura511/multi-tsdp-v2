import requests
from time import gmtime, strftime, time, localtime, sleep
import logging
import pandas as pd
import sqlite3



    
        
def place_order(dbPath, action, quant, sym, type, systemid, submit,apikey, parentsig=None):
    conn = sqlite3.connect(dbPath)
    sigid=int(pd.read_sql('select * from c2sigid', con=conn).iloc[-1])
    if submit == False:
        return 0;
    url = 'https://collective2.com/world/apiv2/submitSignal'
    
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
    sigid=int(sigid)+1
    
    data = { 
    		"apikey":   apikey, # "tXFaL4E6apdfmLtGasIovtGnUDXH_CQso7uBpOCUDYGVcm1w0w", 
    		"systemid": systemid, 
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
    return sigid

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
    
#place_order('BTO','1','EURUSD','forex')
