from subprocess import Popen, PIPE, check_output
import time
import pandas as pd
import threading
from datetime import datetime as dt
start_time = time.time()

def runThreads(threadlist):

    def runInThread(sym, popenArgs):
        print 'starting thread for', sym
        
        with open(logPath+sym+'_v4.txt', 'w') as f:
            with open(logPath+sym+'_v4_error.txt', 'w') as e:
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
        
logPath='/logs/'
csiDataPath=  './data/csidata/v4futures2/'
futuresdict = pd.read_csv('./data/systems/futuresdict.csv', index_col='CSIsym')
runPath='./run_futures.py'
threadlist = [(csiRunSym,['python', runPath,csiRunSym,'0']) for csiRunSym in futuresdict.index]
print len(threadlist), 'threads found..'
runThreads(threadlist)
print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes ', dt.now()
