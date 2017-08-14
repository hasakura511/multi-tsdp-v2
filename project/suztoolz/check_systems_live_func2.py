import requests
import os
import numpy as np
import pandas as pd
import subprocess
import numpy as np
import pandas as pd
import time
import calendar
#from ibapi.place_order import place_order as place_iborder
#from c2api.place_order import place_order as place_c2order
import json
from pandas.io.json import json_normalize
#from ibapi.get_exec import get_ibpos, get_exec_open as get_ibexec_open, get_ibpos_from_csv
from c2api.get_exec import get_c2livepos, retrieveSignalsWorking, get_c2lastEquity

#from seitoolz.signal import get_dps_model_pos, get_model_pos
#from seitoolz.order import adj_size
#from seitoolz.get_exec import get_executions
from time import gmtime, strftime, localtime, sleep
import logging
import sys
import threading
from datetime import datetime as dt
import sqlite3
import traceback
#logging.basicConfig(filename='/logs/check_systems.log',level=logging.DEBUG)

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
    
def reconcileWorkingSignals(account, workingSignals, sym, sig, c2sig, qty, c2qty, signal_check, qty_check):
    errors=0
    print 'position mismatch: ', sym, 's:'+str(sig), 'c2s:'+str(c2sig), 'q:'+str(qty), 'c2q:'+str(c2qty),
    #position size adjustements
    if sig!=c2sig:
        #signal reversal there is no adjustment
        c2qty2=0
    else:
        #signal is the same there is an adjustment
        #c2qty2 existing,orders is adjustment
        c2qty2=c2qty
        
    if 'symbol' in workingSignals[account] and sym in workingSignals[account].symbol.values:
        orders = workingSignals[account][workingSignals[account].symbol==sym]
        orders.quant = orders.quant.astype(int)
        #Open orders
        if sig==1 and 'BTO' in orders.action.values:
            print 'working sig OK',
            signal_check ='OK: Open Order: Entry Order'
            #new open adjustment plus existing position equals qty in system file
            if orders[orders.action=='BTO'].quant.values[0] +c2qty2 == qty:
                print 'qty OK'
                qty_check ='OK: Open Order: Entry Order'
            else:
                print 'qty ERROR'
                qty_check ='ERROR: Open Order: Entry Order'
                errors+=1
        elif sig==-1 and 'STO' in orders.action.values:
            print 'working sig OK',
            signal_check ='OK: Open Order: Entry Order'
            if orders[orders.action=='STO'].quant.values[0] +c2qty2== qty:
                print 'qty OK'
                qty_check ='OK: Open Order: Entry Order'
            else:
                print 'qty ERROR'
                qty_check ='ERROR: Open Order: Entry Order'
                errors+=1
        #Close Orders where sig/qty = 0
        elif (sig==0 and ('STC' in orders.action.values or 'BTC' in orders.action.values))\
                or (qty==0 and ('STC' in orders.action.values or 'BTC' in orders.action.values)):
            print 'working sig OK',
            signal_check ='OK: Open Order: Exit Order'
            if orders.quant.values[0] == c2qty:
                print 'qty OK'
                qty_check ='OK: Open Order: Exit Order'
            else:
                print 'qty ERROR'
                qty_check ='ERROR: Open Order: Exit Order'
                errors+=1
        else:
            #Close Orders where qty adjustments
            if sig==c2sig:
                print 'working sig OK',
                signal_check ='OK: Open Order: qty adjustments'
                #check for position adjustment
                if orders.action.values[0]=='BTC' and c2qty2-orders[orders.action=='BTC'].quant.values[0] == qty:
                    print 'qty OK'
                    qty_check ='OK: Open Order: qty adjustments'
                elif orders.action.values[0]=='STC' and c2qty2-orders[orders.action=='STC'].quant.values[0] == qty:
                    print 'qty OK'
                    qty_check ='OK: Open Order: qty adjustments'
                else:
                    #unknown scenario
                    print 'Wrong working qty found!'
                    qty_check ='ERROR: wrong qty found'
                    errors+=1

            else:
                #unknown scenario
                print 'Wrong working signal found!'
                signal_check ='ERROR: wrong signal found'
                errors+=1
    else:
        print 'ERROR no working orders found!'
        #signal_check ='ERROR: no signal found'
        errors+=1
    return errors, signal_check, qty_check

def check_systems_live(debug, ordersDict, csidate):
    start_time = time.time()
    accounts = ordersDict.keys()
    order_status_dict={}
    if debug:
        #systems = ['v4micro']
        #systems = ['v4futures','v4mini','v4micro']
        logging.basicConfig(filename='C:/logs/c2.log',level=logging.DEBUG)
        logging.info('test')
        #savePath='D:/ML-TSDP/data/portfolio/'
        systemPath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/systems/'
        dbPath='C:/Users/Hidemi/Desktop/Python/TSDP/ml/data/futures.sqlite3' 
        dbPath2='D:/ML-TSDP/data/futures.sqlite3' 
        def place_order2(a,b,c,d,e,f,g):
            return "0"
    else:
        #systems = ['v4futures','v4mini','v4micro']
        logging.basicConfig(filename='/logs/c2.log',level=logging.DEBUG)
        #savePath='./data/portfolio/'
        systemPath =  './data/systems/'
        dbPath=dbPath2='./data/futures.sqlite3'
        from c2api.place_order import place_order2

    #conn = sqlite3.connect(dbPath)
    writeConn = sqlite3.connect(dbPath)
    readConn =  sqlite3.connect(dbPath2)
    
    c2openpositions={}
    workingSignals={}
    #futuresDict={}
    totalerrors=0
        
        
    for account in accounts:
        #subprocess.call(['python', 'get_ibpos.py'])       
        print account, 'Getting c2 positions...'
        #systemdata=pd.read_csv(systemPath+'system_'+account+'_live.csv')
        ordersDict[account]=systemdata=ordersDict[account]
        ordersDict[account].index=ordersDict[account].c2sym
        #portfolio and equity
        #c2list=get_c2_list(systemdata)
        #c2systems=c2list.keys()
        #for systemname in c2systems:
            #(systemid, apikey)=c2list[systemname]
        apikey = systemdata.c2api[0]
        systemid = systemdata.c2id[0]
        c2openpositions[account]=get_c2livepos(systemid, apikey, account)
        c2open=c2openpositions[account].copy()
        if len(c2open)>0:
            c2open['system']=account
            c2open['Date']=csidate
            c2open['timestamp']=int(calendar.timegm(dt.utcnow().utctimetuple()))
            c2open.to_sql(name='c2_portfolio',con=writeConn, index=False, if_exists='append')
            print  'Saved',account,'c2_portfolio to sql db',dbPath
        
        sleep(1)
        response = json.loads(retrieveSignalsWorking(systemid, apikey))['response']
        if len(response)>0:
            workingSignals[account]= json_normalize(response)
        else:
            workingSignals[account]= response
        #trades
        #get_executions(systemdata)
        #subprocess.call(['python', 'get_ibpos.py'])
        sleep(1)
        #print 'success!'
        print account, 'Getting c2 equity...',
        equity=get_c2lastEquity(systemdata)
        if len(equity)>0 and 'modelAccountValue' in equity.columns:
            print equity['modelAccountValue'][0]
            equity['system']=account
            equity['Date']=csidate
            equity['timestamp']=int(calendar.timegm(dt.utcnow().utctimetuple()))
            equity.to_sql(name='c2_equity',con=writeConn, index=False, if_exists='append')
            print  'Saved',account,'c2_equity to sql db',dbPath
        else:
            print 'Could not get last equity from c2'

                        
    for account in c2openpositions.keys():
        print account, 'Position Checking..'
        ostatus_cols=['account','order_type','contract','broker','selection','system_signal',\
                                'broker_position','signal_check','system_qty','broker_qty','qty_check','openedWhen','Date','timestamp']
        order_status_dict[account]=pd.DataFrame(columns=ostatus_cols)
        exitList=[]
        broker='c2'
        account=account
        Date=int(csidate)
        timestamp=int(calendar.timegm(dt.utcnow().utctimetuple()))
        #sig reconciliation
        c2_count=0
        sys_count=0
        mismatch_count=0
        exit_count=0
        error_count=0
        c2openpositions[account]['signal']=np.where(c2openpositions[account]['long_or_short'].values=='long',1,-1)

        #check contracts in c2 file not in system file.
        for sym in c2openpositions[account].index:
            contract = sym
            c2_count+=1
            if sym in ordersDict[account].index:
                selection = ordersDict[account].ix[sym].selection
                order_type = ordersDict[account].ix[sym].ordertype
                openedWhen=c2openpositions[account].ix[sym].openedWhen
                c2sig =broker_position= int(c2openpositions[account].ix[sym].signal)
                sig=system_signal=int(ordersDict[account].ix[sym].signal)
                qty=system_qty=int(ordersDict[account].ix[sym].c2qty)
                c2qty=broker_qty=int(c2openpositions[account].ix[sym].quant_opened)-int(c2openpositions[account].ix[sym].quant_closed)
                signal_check='ERROR: No Open Orders Found' if sig != c2sig else 'OK'
                qty_check='ERROR: No Open Orders Found' if qty != c2qty else 'OK'
                if sig != c2sig or qty != c2qty:
                    mismatch_count+=1
                    errors, signal_check, qty_check=reconcileWorkingSignals(account, workingSignals, sym, sig, c2sig, qty, c2qty, signal_check, qty_check)
                    error_count+=errors
                    
                dfdict={}
                for i in ostatus_cols:
                    dfdict[i] = locals()[i]
                order_status_dict[account]= order_status_dict[account].append(pd.DataFrame(data=dfdict, index=[sym]))
                    
            else:
                #exit if not in the main file
                systemdata_csi=pd.read_sql('select * from %s where timestamp=\
                            (select max(timestamp) from %s as maxtimestamp)' % (account, account), con=readConn,  index_col='c2sym')
                if sym not in systemdata_csi.index:
                    exit_count+=1
                    c2sig = int(c2openpositions[account].ix[sym].signal)
                    c2qty=int(c2openpositions[account].ix[sym].quant_opened)-int(c2openpositions[account].ix[sym].quant_closed)
                    symInfo=systemdata_csi.ix[[x for x in systemdata_csi.index if sym[:-2] in x][0]]
                    if c2sig==1:
                        action='STC'
                    else:
                        action='BTC'
                        
                    #place order to exit the contract.

                    #old contract does not exist in system file so use the new contract c2id and api
                    response = place_order2(action, c2qty, sym, symInfo.c2type, symInfo.c2id, True, symInfo.c2api)
                    exitList.append(sym+' not in system file. exiting contract!!.. '+response)
        #check contracts in system file not in c2.
        for sym in ordersDict[account].index:
            contract = sym
            selection = ordersDict[account].ix[sym].selection
            order_type = ordersDict[account].ix[sym].ordertype
            openedWhen=''
            sig=system_signal=int(ordersDict[account].ix[sym].signal)
            qty=system_qty=int(ordersDict[account].ix[sym].c2qty)
            systemSym = (sig !=0 and qty !=0)
            if systemSym:
                sys_count+=1
            if sym not in c2openpositions[account].index and systemSym:
                mismatch_count+=1
                broker_position=0
                broker_qty=0
                errors, signal_check, qty_check=reconcileWorkingSignals(account, workingSignals, sym, sig, 0, qty, 0, signal_check, qty_check)
                error_count+=errors
                dfdict={}
                for i in ostatus_cols:
                    dfdict[i] = locals()[i]
                order_status_dict[account]= order_status_dict[account].append(pd.DataFrame(data=dfdict, index=[sym]))
        totalerrors+=error_count
        for e in exitList:
            print e
        print 'c2:'+str(c2_count)+' account:'+str(sys_count)+' mismatch:'+str(mismatch_count)+' exit:'+str(exit_count)+' errors:'+str(error_count)
        print 'DONE!\n'
    pd.DataFrame(pd.Series(data=totalerrors), columns=['totalerrors']).to_sql(name='checkSystems',con=writeConn, index=False, if_exists='replace')
    print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes ', dt.now()
    
    for account, orderstatusDF in order_status_dict.items():
        table='checkSystems_'+account
        orderstatusDF.to_sql(name=table,con=writeConn, index=False, if_exists='replace')
        print 'saved', table,'to',dbPath
    return totalerrors

if __name__ == "__main__":
    debug=False
    def lastCsiDownloadDate(csiDataPath):
        datafiles = os.listdir(csiDataPath)
        dates = []
        for f in datafiles:
            lastdate = pd.read_csv(csiDataPath+f, index_col=0).index[-1]
            if lastdate not in dates:
                dates.append(lastdate)
                
        return max(dates)
        
    csidate=lastCsiDownloadDate('./data/csidata/v4futures2/')
    dbPathRead='./data/futures.sqlite3'
    readcon= sqlite3.connect(dbPathRead)
    webSelection=pd.read_sql('select * from webSelection where timestamp=\
            (select max(timestamp) from webSelection)', con=readcon)
    systems = eval(webSelection.selection[0]).keys()
    ordersDict={}
    for account in systems:
        ordersDict[account]=pd.read_sql('select * from (select * from %s\
                order by timestamp ASC) group by CSIsym' % (account+'_live'),\
                con=readcon,  index_col='CSIsym')
    errors=check_systems_live(debug, ordersDict, csidate)
    print 'total errors found', errors