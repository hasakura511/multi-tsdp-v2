import requests
from time import gmtime, strftime, time, localtime, sleep
import logging

def get_working_signals(systemid,apikey):
    logging.info(  "get_working_signals: " + systemid)
    url = 'https://api.collective2.com/world/apiv3/retrieveSignalsWorking'
    
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
    
    data = { 
    		"apikey":   apikey, #"tXFaL4E6apdfmLtGasIovtGnUDXH_CQso7uBpOCUDYGVcm1w0w", 
    		"systemid": systemid
    	}
    
    params={}
    
    r=requests.post(url, params=params, json=data);
    sleep(2)
    logging.info( str(r.text)  )
    return r.text


def cancel_signal(signalid, systemid, apikey):
    logging.info(  "cancel_signal: systemid:" + systemid + ', signalid' + signalid)
    
    url = 'https://api.collective2.com/world/apiv3/cancelSignal'
    
    headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
    
    data = { 
    		"apikey":   apikey,#"tXFaL4E6apdfmLtGasIovtGnUDXH_CQso7uBpOCUDYGVcm1w0w", 
    		"systemid": systemid, 
    		"signalid": signalid
    	}
    
    params={}
    
    r=requests.post(url, params=params, json=data);
    logging.info( str(r.text)  )