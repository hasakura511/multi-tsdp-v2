# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 10:46:08 2016

@author: Hidemi
"""
import time
import math
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
import sqlite3
import io
import traceback
import json
import imp
import urllib
import urllib2
import requests
import webbrowser
import re
import datetime
from datetime import datetime as dt
import calendar
import time
import inspect
import os
import os.path
import sys
import ssl
from copy import deepcopy
#from suztoolz.transform import ATR2
#import matplotlib.pyplot as plt
#import seaborn as sns
#from suztoolz.datatools.seasonalClass import seasonalClassifier
import logging
def vol_adjsize_board(debug, threadlist):
    try:
        start_time = time.time()
        version='v4'
        systems = ['v4futures','v4mini','v4micro']
        riskEquity=1000
        riskEquity_mini=250
        riskEquity_micro=250      
        c2safef=1  
        #range (-1 to 1) postive for counter-trend negative for trend i.e.
        #-1 would 0 safef ==1 and double safef==2
        #1 would 0 safef ==2 and double safef==1
        safefAdjustment=0
        offline =['AC','AEX','CC','CGB','CT','DX','EBL','EBM','EBS','ED','FCH','FDX','FEI','FFI','FLG','FSS','HCM','HIC','KC','KW','LB','LCO','LGO','LRC','LSU','MEM','MFX','MW','O','OJ','RR','RS','SB','SIN','SJB','SMI','SSG','STW','SXE','TF','VX','YA','YB','YT2','YT3',]
        offline_mini = ['AC','AD','AEX','BO','BP','CC','CD','CGB','CT','DX','EBL','EBM','EBS','ED','FC','FCH','FDX','FEI','FFI','FLG','FSS','FV','GC','HCM','HIC','HO','KC','KW','LB','LC','LCO','LGO','LH','LRC','LSU','MEM','MFX','MP','MW','NE','NIY','NQ','O','OJ','PA','PL','RB','RR','RS','S','SB','SF','SI','SIN','SJB','SMI','SSG','STW','SXE','TF','US','VX','YA','YB','YM','YT2','YT3',]
        offline_micro =['AC','AD','AEX','BP','C','CC','CD','CGB','CL','CT','CU','DX','EBL','EBM','EBS','ED','EMD','FC','FCH','FDX','FEI','FFI','FLG','FSS','FV','GC','HCM','HIC','HO','JY','KC','KW','LB','LC','LCO','LGO','LH','LRC','LSU','MEM','MFX','MP','MW','NE','NIY','NQ','O','OJ','PA','PL','RB','RR','RS','S','SB','SF','SI','SIN','SJB','SM','SMI','SSG','STW','SXE','TF','TU','US','VX','W','YA','YB','YM','YT2','YT3',]


        lookback=20
        refresh=False
        currencyFile = 'currenciesATR.csv'
        systemFilename='system_v4futures.csv'
        systemFilename2='system_v4mini.csv'
        systemFilename3='system_v4micro.csv'
        c2id_macro=107146997
        c2id_mini=101359768
        c2id_micro=101533256
        c2key='tXFaL4E6apdfmLtGasIovtGnUDXH_CQso7uBpOCUDYGVcm1w0w'

        corecomponents =[
                        'RiskOn',
                        'RiskOff',
                        'LastSEA',
                        'AntiSEA',
                        'prevACT',
                        'AntiPrevACT',
                        '0.75LastSIG',
                        '0.5LastSIG',
                        '1LastSIG',
                        'Anti1LastSIG',
                        'Anti0.75LastSIG',
                        'Anti0.5LastSIG',
                        'Custom',
                        'AntiCustom',
                        'None',
                        ]
                        
        reverseComponentsDict ={
                        'None':'Off',
                        'prevACT':'Previous',
                        'AntiPrevACT':'Anti-Previous',
                        'RiskOn':'RiskOn',
                        'RiskOff':'RiskOff',
                        'Custom':'Custom',
                        'AntiCustom':'Anti-Custom',
                        '0.75LastSIG':'50/50',
                        '0.5LastSIG':'LowestEquity',
                        '1LastSIG':'HighestEquity',
                        'Anti1LastSIG':'AntiHighestEquity',
                        'Anti0.75LastSIG':'Anti50/50',
                        'Anti0.5LastSIG':'AntiLowestEquity',
                        'LastSEA':'Seasonality',
                        'AntiSEA':'Anti-Seasonality',
                        }
                        
        if debug:
            debug=True
            showPlots=False
            #refreshSea tries to recreate the first run futuresATR file after new signals have been generated
            refreshSea=False
            dbPath='C:/Users/Hidemi/Desktop/Python/TSDP/ml/data/futures.sqlite3' 
            dbPath2='D:/ML-TSDP/data/futures.sqlite3'
            dbPathWeb = 'D:/ML-TSDP/web/tsdp/db.sqlite3'
            dataPath='D:/ML-TSDP/data/csidata/v4futures2/'
            savePath= 'C:/Users/Hidemi/Desktop/Python/TSDP/ml/data/results/' 
            pngPath=savePath2 = 'C:/Users/Hidemi/Desktop/Python/TSDP/ml/data/results/' 
            feedfile='D:/ML-TSDP/data/systems/system_ibfeed.csv'
            #test last>old
            #dataPath2=savePath2
            #signalPath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/signals/' 
            
            #test last=old
            dataPath2='D:/ML-TSDP/data/'
            
            signalPath ='D:/ML-TSDP/data/signals/'
            signalSavePath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/signals/' 
            systemPath = 'C:/Users/Hidemi/Desktop/Python/SharedTSDP/data/systems/' 
            logging.basicConfig(filename='C:/logs/vol_adjsize_board_error.log',level=logging.DEBUG)
            
        else:
            debug=False
            showPlots=False
            #if set to on its probably going to mess up the db.
            refreshSea=False
            dbPathWeb ='./web/tsdp/db.sqlite3'
            dbPath=dbPath2='./data/futures.sqlite3'
            dataPath='./data/csidata/v4futures2/'
            dataPath2='./data/'
            savePath='./data/'
            signalPath = './data/signals/' 
            signalSavePath = './data/signals/' 
            savePath2 = './data/results/'
            pngPath= './web/tsdp/betting/static/images/'
            systemPath =  './data/systems/'
            feedfile='./data/systems/system_ibfeed.csv'
            logging.basicConfig(filename='/logs/vol_adjsize_board_error.log',level=logging.DEBUG)
            
        def is_int(s):
            try: 
                int(s)
                return True
            except ValueError:
                return False
                
        def to_signals(df, Anti=False):
            df2=df.copy()
            if Anti:
                df2[df>0]=-1
                df2[df<0]=1
            else:
                df2[df>0]=1
                df2[df<0]=-1
            return df2
            
        def checkTableExists(dbconn, tablename):
            dbcur = dbconn.cursor()
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


            
        writeConn = sqlite3.connect(dbPath)
        readConn =  sqlite3.connect(dbPath2)
        readWebConn = sqlite3.connect(dbPathWeb)

        #request='http://www.globalsystemsmanagement.net/last_userselection/'
        #selectionDF = pd.DataFrame(requests.get(request).json())
        #selectionDict=eval(selectionDF.selection[0])
        selectionDF=pd.read_sql('select * from betting_userselection where timestamp=\
                (select max(timestamp) from betting_userselection as maxtimestamp)', con=readWebConn, index_col='userID')
        selectionDict=eval(selectionDF.selection.values[0])
        c2system_macro=c2system=selectionDict["v4futures"][0]
        c2system_mini=selectionDict["v4mini"][0]
        c2system_micro=selectionDict["v4micro"][0]

        '''
        signals = ['ACT','prevACT','AntiPrevACT','RiskOn','RiskOff','Custom','AntiCustom',\
                        'LastSIG', '0.75LastSIG','0.5LastSIG','1LastSIG','Anti1LastSIG','Anti0.75LastSIG','Anti0.5LastSIG',\
                        'prevSEA','LastSEA','AntiSEA','AdjSEA','AntiAdjSEA',\
                        'Voting','Voting2','Voting3','Voting4','Voting5','Voting6','Voting7','Voting8','Voting9',\
                        'Voting10','Voting11','Voting12','Voting13','Voting14','Voting15']

        '''
        #find any changes to userselection
        if checkTableExists(readConn, 'webSelection'):
            last_selectionWeb=selectionDF.reset_index().drop(['id'],axis=1).to_dict()
            #read from writeconn for debugging purpose
            last_selectionBack=pd.read_sql('select * from webSelection where timestamp=\
                                    (select max(timestamp) from webSelection as maxtimestamp)',\
                                    con=readConn, index_col='userID').reset_index().to_dict()
            # if selectionDF is same as before then exit.
            # need to do this by system. 
            selectionDict_old=eval(last_selectionBack['selection'][0])
            if selectionDict == selectionDict_old:
                #no change sysexit
                print 'no change in user selection', selectionDict
                sys.exit('no change in user selection')
            else:
                #write new selection to backend db
                selectionDF.reset_index().drop(['id'],axis=1).to_sql(name='webSelection', if_exists='append',\
                            con=writeConn, index=False)
                print 'new user selection found. appending webSelection to', dbPath
                #check each system to see if its changed.
                #need to check each system in between MOC and CSI data release
                #we don't revert back to the prior day's. this will happen when only one system
                #changes and others don't.
                
                newselectionDict={}
                for key in selectionDict.keys():
                    if selectionDict[key] != selectionDict_old[key]:
                        print key, 'found new', selectionDict[key], 'old', selectionDict_old[key]
                        newselectionDict[key]=selectionDict[key]
                #reset the selection dict with only the systems that's changed.
                selectionDict = newselectionDict.copy()
        else:  
            #create selection
            selectionDF.reset_index().drop(['id'],axis=1).to_sql(name='webSelection', if_exists='replace',\
                        con=writeConn, index=False)
            print 'could not find table webSelection. wrote webSelection to ',dbPath
        print '\n'
            
        #loadlast futures ATR data. live dumps into table 'futuresATR'
        #using the EOD all because ithink if we are at this point, we are processing immediate orders.
        #or new orders (change in system selection since MOC). right??
        futuresDF=pd.read_sql('select * from futuresDF_all where timestamp=\
                                (select max(timestamp) from futuresDF_all as maxtimestamp)',\
                                con=readConn, index_col='CSIsym')
        componentsignals=futuresDF[corecomponents]

        systemDict={}
        orderDict={}
        for key in selectionDict.keys():
            immediate = eval(selectionDict[key][1])
            if immediate:
                ComponentsDict = eval(selectionDF[key].values[0])
                votingSystems = { key: ComponentsDict[key] for key in [x for x in ComponentsDict if is_int(x)] }
                #add voting systems
                systemDict[key]={key: to_signals(futuresDF[ComponentsDict[key]].sum(axis=1)) for key in votingSystems.keys()}
                #add anti-voting systems
                systemDict[key].update({'Anti-'+key: to_signals(futuresDF[ComponentsDict[key]].sum(axis=1), Anti=True)\
                                                        for key in votingSystems.keys()})
                #check (systemDict[key]['1']+systemDict[key]['Anti-1']).sum()
                systemDict[key].update({ reverseComponentsDict[key]: componentsignals[key] for key in componentsignals})

                #load last csi system file
                systemdata = pd.read_sql('select * from %s' % key, con=readConn)
                systemdata.index = [x.split('_')[1] for x in systemdata.System]
                systemdata.signal = systemDict[key][selectionDict[key][0]]
                orderDict[key]=systemdata.ix[([x[0] for x in threadlist])]
                print key, 'added system',selectionDict[key][0],'to orderDict for IMMEDIATE processing.'
        return orderDict
        
    except Exception as e:
        logging.exception("message")
        logging.info( str(dt.now()))
