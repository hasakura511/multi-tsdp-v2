import numpy as np
import pandas as pd
import time
import json
from pandas.io.json import json_normalize
from c2api.get_exec import get_c2equity, get_c2lastEquity, get_executions, get_c2pos
from time import gmtime, strftime, localtime, sleep
import logging
import sys
import threading
from datetime import datetime as dt

logging.basicConfig(filename='/logs/refresh_c2.log',level=logging.DEBUG)
start_time = time.time()

        
systems = ['v4futures','v4mini', 'v4micro']
for system in systems:
    #subprocess.call(['python', 'get_ibpos.py'])       
    systemdata=pd.read_csv('./data/systems/system_'+system+'_live.csv')
    systemdata=systemdata.reset_index()
    get_c2pos(systemdata)
    get_executions(systemdata)
    data=get_c2equity(systemdata)
    #subprocess.call(['python', 'get_ibpos.py'])


print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes ', dt.now()