import sys
import os
#print os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from pandas.io.json import json_normalize
from c2api.sig_adj import get_working_signals, cancel_signal

import numpy as np
import pandas as pd
import subprocess
import time
#from ibapi.place_order import place_order as place_iborder
#from c2api.place_order2 import place_order as place_c2order
import json
from pandas.io.json import json_normalize
from c2api.get_exec import get_c2pos, get_exec_open, get_c2equity, get_c2_pos, get_exec, place_order
#from seitoolz.get_exec import get_executions as get_c2trades
#from ibapi.place_order2 import place_orders as place_iborders
from time import gmtime, strftime, localtime, sleep
import logging
import sys
import threading
import sqlite3
import traceback

#logging.basicConfig(filename='/logs/proc_signal_v4_live.log',level=logging.DEBUG)


def checkTableExists(dbcon, tablename):
    dbcur = dbcon.cursor()
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
    
def get_c2trades(systemid, name, c2api, portfolioPath):
    #global portfolioPath
    filename=portfolioPath+ name + '_trades.csv'
    
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
        print 'Saved c2 trades to ', filename

def get_c2executions(data, portfolioPath):        
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
        get_c2trades(c2id, stratName, c2api, portfolioPath)


def start_trade(systems, dbPath): 
    systems.index=systems.c2sym
    #global debug
    #if debug:
    #print "Starting " + str(systems.iloc[0]['Name'])
       #logging.info("Starting " + str(systems.iloc[0]['Name']))
    try:
        #model=get_models(systems)
        model = pd.concat([systems.System, systems.signal, systems.c2qty], axis=1)
        model.columns = ['system','action','qty']
        model=model.set_index(['system'])
        
        symbols=systems['c2sym'].values
        for symbol in symbols:
          system=systems.ix[symbol].copy()
          symbol=system['ibsym']
          if system['ibtype'] == 'CASH':
                symbol = str(system['ibsym']) + str(system['ibcur'])
          
          #feed_dict=bars.get_bidask_list()
          if system['ibtype'] != 'BITCOIN':
            #and get_timestamp() - int(system['last_trade']) > int(system['trade_freq']):
            if system['c2submit'] or system['ibsubmit']:
                adj_size(dbPath, model, system['System'], system['Name'], 
                         str(int(system['c2id'])),system['c2api'],
                         system['c2qty'],system['c2sym'],system['c2type'],system['c2submit'], 
                         system['ibqty'],system['ibsym'],system['ibcur'],
                         system['ibexch'],system['ibtype'],system['ibsubmit'],
                         system['iblocalsym'])
            
          #time.sleep(30)
    except Exception as e:
        #logging.error("something bad happened", exc_info=True)
        print e

def process_systems(systemdata):
      threads = []        
      systemList=dict()
      #get c2 positions
      get_c2pos(systemdata)
      for i in systemdata.index:
          system=systemdata.ix[i]
          #print system, sys.argv[2]
          if len(sys.argv) < 2 or (len(sys.argv[2]) > 0 and sys.argv[2] == system['Name']):
              if systemList.has_key(system['Name']):
                  systemList[system['Name']]=systemList[system['Name']].append(system)
              else:
                  systemList[system['Name']]=pd.DataFrame()
                  systemList[system['Name']]=systemList[system['Name']].append(system)
              
      for systemname in systemList.keys():
           systems=systemList[systemname]
           systems['last_trade']=0
           systems['key']=systems['c2sym']
           systems=systems.set_index('key')
           sig_thread = threading.Thread(target=start_trade, args=[systems])
           sig_thread.daemon=True
           threads.append(sig_thread)
           sig_thread.start()
      [t.join() for t in threads]



def adj_size(dbPath, model_pos, system, systemname, systemid, c2apikey, c2quant,\
                    c2sym, c2type, c2submit, ibquant, ibsym, ibcurrency, ibexch, ibtype,\
                    ibsubmit, iblocalsym=''):
    system_pos=model_pos.loc[system]
   
    if c2submit == 'TRUE':
        c2submit=True
    elif c2submit == 'FALSE':
        c2submit=False
        
    if ibsubmit == 'TRUE':
        ibsubmit=True
    elif ibsubmit == 'FALSE':
        ibsubmit=False
    #print str(system_pos['action'])
    #print "c2: " 
    #print c2_pos
    if c2submit:
        c2_pos_qty=get_c2_pos(systemname, c2sym)           
        system_c2pos_qty=round(system_pos['action']) * c2quant
        #print('==============')
        print systemname, 
        print system, 
        #print('Strategy:' + systemname),
        #print('system_pos:' +str(system_pos))
        #print("  Signal Name: " + system)
        #print("  C2ID: " + systemid + "  C2Key: " + c2apikey)
        #print("  C2Sym: " + c2sym + " IBSym: " + ibsym)
        print( "system_pos: " + str(system_c2pos_qty) ),
        print( "c2_pos: " + str(c2_pos_qty) ),
        if system_c2pos_qty != c2_pos_qty:
            print 'Placing Order',
        else:
            print 'No orders to be placed.'
            
        if system_c2pos_qty > c2_pos_qty:
            c2quant=system_c2pos_qty - c2_pos_qty
            isrev=False
            psigid=0
            if c2_pos_qty < 0:        
                qty=min(abs(c2_pos_qty), abs(c2_pos_qty - system_c2pos_qty))
                print( 'BTC: ' + str(qty) ),
                if system_c2pos_qty ==0:
                    print ''
                psigid=place_order(dbPath,'BTC', qty, c2sym, c2type, systemid, c2submit, c2apikey)
                isrev=True                
                c2quant = c2quant - qty
                
            if c2quant > 0:
                print( 'BTO: ' + str(c2quant) )
                if isrev:
                    place_order(dbPath,'BTO', c2quant, c2sym, c2type, systemid, c2submit, c2apikey, psigid)
                else:
                    place_order(dbPath,'BTO', c2quant, c2sym, c2type, systemid, c2submit, c2apikey)
        
        if system_c2pos_qty < c2_pos_qty:
            c2quant=c2_pos_qty - system_c2pos_qty   
            isrev=False
            psigid=0
            if c2_pos_qty > 0:        
                qty=min(abs(c2_pos_qty), abs(c2_pos_qty - system_c2pos_qty))
                print( 'STC: ' + str(qty) ),
                if system_c2pos_qty ==0:
                    print ''
                psigid=place_order(dbPath,'STC', qty, c2sym, c2type, systemid, c2submit, c2apikey)
                isrev=True 
                c2quant = c2quant - qty

            if c2quant > 0:
                print( 'STO: ' + str(c2quant) )
                if isrev:
                    place_order(dbPath,'STO', c2quant, c2sym, c2type, systemid, c2submit, c2apikey, psigid)
                else:
                    place_order(dbPath,'STO', c2quant, c2sym, c2type, systemid, c2submit, c2apikey)
'''
    if ibsubmit:
        ib_pos_qty=get_ib_pos(ibsym, ibcurrency)
        system_ibpos_qty=round(system_pos['action']) * ibquant
        
        logging.info( "system_ib_pos: " + str(system_ibpos_qty) )
        logging.info( "ib_pos: " + str(ib_pos_qty) )
        if system_ibpos_qty > ib_pos_qty:
            ibquant=int(system_ibpos_qty - ib_pos_qty)
            logging.info( 'BUY: ' + str(ibquant) )
            place_iborder('BUY', ibquant, ibsym, ibtype, ibcurrency, ibexch, ibsubmit, iblocalsym);
        if system_ibpos_qty < ib_pos_qty:
            ibquant=int(ib_pos_qty - system_ibpos_qty)
            logging.info( 'SELL: ' + str(ibquant) )
            place_iborder('SELL', ibquant, ibsym, ibtype, ibcurrency, ibexch, ibsubmit, iblocalsym);
    #
    #place_iborder(ibaction, ibquant, ibsym, ibtype, ibcurrency, ibexch, ibsubmit);
'''
#subprocess.call(['python', 'get_ibpos.py'])       
def proc_orders():
    global systems
    for sys in systems:
        systemdata=pd.read_sql(sql='select * from '+sys, con=conn)
        #systemdata=systemdata.reset_index()
        process_systems(systemdata)
        get_c2executions(systemdata) 

    
def proc_sig_adj(systemid,apikey):
    data=get_working_signals(systemid,apikey);
    jsondata = json.loads(data)
    if len(jsondata['response']) > 0:
        dataSet=json_normalize(jsondata['response'])
        for i in dataSet.index:
            row=dataSet.ix[i]
            cancel_signal(row['signal_id'], systemid,apikey)
            #time.sleep(1)

def proc_signal_v4_live(debug, ordersDict):
    start_time = time.time()
    systems = ordersDict.keys()
    if debug:
        showPlots=False
        #systemfile='./data/systems/system_'+sys.argv[2]+'_live.csv'
        #systemfile='D:/ML-TSDP/data/systems/system_v4futures.csv'
        dbPath='./data/futures.sqlite3' 
        dataPath='D:/ML-TSDP/data/csidata/v4futures4_debug/'
        savePath= './data/results/' 
        savePath2 = './data/results/' 
        feedfile='D:/ML-TSDP/data/systems/system_ibfeed.csv'
        #logging.basicConfig(filename='C:/logs/c2_live.log',level=logging.DEBUG)
        #test last>old
        #dataPath2=savePath2
        #signalPath = './data/signals/' 
        
        #test last=old
        dataPath2='D:/ML-TSDP/data/'
        
        #signalPath ='D:/ML-TSDP/data/signals2/'
        signalPath = './data/signals/' 
        signalSavePath = './data/signals/' 
        systemPath = './data/systems/' 
        portfolioPath = './data/results/c2_' 
    else:
        #systemfile='./data/systems/system_'+sys.argv[2]+'_live.csv'
        feedfile='./data/systems/system_ibfeed.csv'
        dbPath='./data/futures.sqlite3'
        dataPath='./data/csidata/v4futures4/'
        dataPath2='./data/'
        savePath='./data/'
        signalPath = './data/signals2/' 
        signalSavePath = './data/signals2/' 
        savePath2 = './data/results/'
        systemPath =  './data/systems/'
        portfolioPath = './data/portfolio/c2_' 
        #logging.basicConfig(filename='/logs/c2_live.log',level=logging.DEBUG)

    conn = sqlite3.connect(dbPath)
    for sys in systems:
        systemdata=ordersDict[sys]
        seen=dict()

        #cancel all pending orders
        for i in systemdata.index:
                system=systemdata.ix[i]
                if system['c2submit'] and not seen.has_key(str(system['c2id'])):
                    try:
                        proc_sig_adj(str(system['c2id']),system['c2api'])
                        seen[str(system['c2id'])]=1
                    except Exception as e:
                        print 'Error on', system['c2id']


        #send new orders
        if not checkTableExists(conn, 'c2sigid'):
            pd.DataFrame(pd.Series(data=1), columns=['c2sigid']).to_sql(name='c2sigid',con=conn, index=False)
        #systemdata=systemdata.reset_index()
        start_trade(systemdata, dbPath)
        sleep(1)
        get_c2executions(systemdata, portfolioPath)

if __name__ == "__main__":
    if len(sys.argv)==1:
        debug=True
        feedfile='D:/ML-TSDP/data/systems/system_ibfeed.csv'
        csiPath='D:/ML-TSDP/data/csidata/v4futures2/'
        dbPathRead='D:/ML-TSDP/data/futures.sqlite3'
    else:
        debug=False
        feedfile='./data/systems/system_ibfeed.csv'
        csiPath='./data/csidata/v4futures2/'
        dbPathRead='./data/futures.sqlite3'

    def lastCsiDownloadDate(csiDataPath):
        datafiles = os.listdir(csiDataPath)
        dates = []
        for f in datafiles:
            lastdate = pd.read_csv(csiDataPath+f, index_col=0).index[-1]
            if lastdate not in dates:
                dates.append(lastdate)
                
        return max(dates)
        
    csidate=lastCsiDownloadDate(csiPath)
    
    readcon= sqlite3.connect(dbPathRead)
    webSelection=pd.read_sql('select * from webSelection where timestamp=\
            (select max(timestamp) from webSelection)', con=readcon)
    systems = eval(webSelection.selection[0]).keys()
    ordersDict={}
    for account in systems:
        ordersDict[account]=pd.read_sql('select * from (select * from %s\
                order by timestamp ASC) group by CSIsym' % (account+'_live'),\
                con=readcon,  index_col='CSIsym')
    
    #proc_signal_v4_live(debug, ordersDict)
    proc_signal_v4_live(debug, {'v4micro':ordersDict['v4micro']})
    #errors,ordersDictWithErrors=check_systems_live(debug, ordersDict, csidate)
    #print 'total errors found', errors