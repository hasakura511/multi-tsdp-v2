import os
import numpy as np
import pandas as pd
import json
from pandas.io.json import json_normalize
import time
import sys
from datetime import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import sqlite3
import calendar

#this needs to run before 12AM EST or IB wipes out the execution data for the day.
start_time = time.time()
systemName='v4futures'
if len(sys.argv)==1:
    debug=True
    showPlots=True
    commission=2.5
    start_slip=20161128
    figsize=(6,8)
    fontsize=12
    dataPath='./data/'
    portfolioPath = './data/portfolio/'
    dbPathWrite='./data/futures.sqlite3' 
    dbPathRead = './data/futures.sqlite3'
    #savePath= 'D:/Dropbox/Python/TSDP/ml/data/results/' 
    savePath = savePath2 = pngPath='./data/results/' 
    systemPath =  './data/systems/'
    #timetablePath=   './data/systems/timetables_debug/'
    timetablePath=   './data/systems/timetables/'
    feedfile='./data/systems/system_ibfeed.csv'
else:
    debug=False
    showPlots=False
    commission=2.5
    start_slip=20161128
    figsize=(8,13)
    fontsize=20
    dbPathWrite=dbPathRead='./data/futures.sqlite3'
    dataPath=savePath='./data/'
    portfolioPath = savePath2 ='./data/portfolio/'
    pngPath =  './web/tsdp/betting/static/images/'
    systemPath =  './data/systems/'
    timetablePath=   './data/systems/timetables/'
    feedfile='./data/systems/system_ibfeed.csv'
    #from ibapi.moc_live_ib import filterIBexec
    #print('requesting last executions from IB..')
    #print(filterIBexec())

readConn= sqlite3.connect(dbPathRead)
writeConn = sqlite3.connect(dbPathWrite)

IB2CSI_multiplier_adj={
    'HG':100,
    'SI':100,
    'JPY':100,
    }
    
def plotSlip(slipDF, pngPath, filename, title, figsize, fontsize, showPlots=False):
    #plt.figure(figsize=(8,13))
    font = {
            'weight' : 'normal',
            'size'   : 22}
    matplotlib.rc('font', **font)
    
    def align_xaxis3(ax1, ax2):
        """Align zeros of the two axes, zooming them out by same ratio"""
        axes = (ax1, ax2)
        extrema = [ax.get_xlim() for ax in axes]
        tops = [extr[1] / (extr[1] - extr[0]) for extr in extrema]
        # Ensure that plots (intervals) are ordered bottom to top:
        if tops[0] > tops[1]:
            axes, extrema, tops = [list(reversed(l)) for l in (axes, extrema, tops)]

        # How much would the plot overflow if we kept current zoom levels?
        tot_span = tops[1] + 1 - tops[0]

        b_new_t = extrema[0][0] + tot_span * (extrema[0][1] - extrema[0][0])
        t_new_b = extrema[1][1] - tot_span * (extrema[1][1] - extrema[1][0])
        axes[0].set_xlim(extrema[0][0], b_new_t)
        axes[1].set_xlim(t_new_b, extrema[1][1])


    fig = plt.figure(figsize=figsize) # Create matplotlib figure
    ax = fig.add_subplot(111) # Create matplotlib axes
    #ax = slipDF.slippage.plot.bar(color='r', width=0.5)
    ax2 = ax.twiny()

    width=.3
    slipDF.dollarslip.plot(kind='barh', color='red',width=width, ax=ax, position=1)
    slipDF.delta.plot(kind='barh', color='blue', width=width,ax=ax2, position=0)

    ax.set_xlabel('$ Slippage (red)')
    ax2.set_xlabel('Slippage Minutes (blue)')
    ax.grid(b=True)
    ax2.grid(b=False)
    #ax2.set_xlim(-40,140)
    #ax2 = slipDF.hourdelta.plot.bar(color='b', width=0.5)
    #plt.axvline(0, color='k')
    plt.text(0.5, 1.08, title,
             horizontalalignment='center',
             fontsize=fontsize,
             transform = ax2.transAxes)
    align_xaxis3(ax, ax2)
    #plt.ylim(0,80)
    #plt.xticks(np.arange(-1,1.25,.25))
    #plt.grid(True)

    if pngPath != None and filename != None:
        plt.savefig(pngPath+filename, bbox_inches='tight')
        print 'Saved '+pngPath+filename
    if len(sys.argv)==1 and showPlots:
        #print data.index[0],'to',data.index[-1]
        plt.show()
    plt.close()
    
#feeddata=pd.read_csv(feedfile,index_col='ibsym')
contractsDF = pd.read_sql('select * from ib_contracts where timestamp=\
            (select max(timestamp) from ib_contracts as maxtimestamp)', con=readConn,  index_col='ibsym')


lastExecutions= pd.read_sql('select * from ib_executions where timestamp=\
            (select max(timestamp) from ib_executions as maxtimestamp)', con=readConn,  index_col='contract').drop_duplicates()


if lastExecutions.shape[0] >0:
    lastExecutions.times = pd.to_datetime(lastExecutions.times)
    lastExecutions.lastAppend= pd.to_datetime(lastExecutions.lastAppend).dt.strftime('%Y%m%d')
    executions=pd.DataFrame()
    for idx in lastExecutions.index:
        #print type(lastExecutions.ix[idx]), isinstance(lastExecutions.ix[idx], pd.Series)
        if not isinstance(lastExecutions.ix[idx], pd.Series):
            if idx not in executions.index:
                executions=executions.append(lastExecutions.ix[idx].sort_values(by=['qty'], ascending=False).iloc[0])
        else:
            executions=executions.append(lastExecutions.ix[idx])
    #calc slipage for current contracts
    executions = executions.ix[[x for x in executions.index if x in contractsDF.contracts.values]].copy()
        
    if executions.shape[0]>0:
        executions['CSIsym']=contractsDF.ix[executions.ibsym].CSIsym.values
        csidate=executions.lastAppend[0]
        #csidate='20170422'
        futuresDF=pd.read_sql('select * from futuresDF_results where timestamp=\
                    (select max(timestamp) from futuresDF_results where Date=%s)' % (csidate), con=readConn,  index_col='CSIsym')
        system = pd.read_sql('select * from (select * from signals_live where Date=%s and Name=\'%s\'\
                            order by timestamp ASC) group by CSIsym' %(csidate, systemName),\
                            con=readConn,  index_col='CSIsym')
        
        if system.shape[0]<1:
            sys.exit('{} not in signals_live'.format(csidate))
        if futuresDF.shape[0]<1:
            sys.exit('{} not in futuresDF_results'.format(csidate))
        selection=system.selection[0]
        if os.path.isfile(timetablePath+str(csidate)+'.csv'):
            timetable = pd.read_csv(timetablePath+str(csidate)+'.csv', index_col=0)
            for col in timetable.columns:
                timetable[col]=pd.to_datetime(timetable[col])
            idx_close = [x for x in timetable.index if 'close' in x]
            idx_trigger = [x for x in timetable.index if 'trigger' in x]
        else:
            print 'timetable for', csidate,'not found! Ending script.'
            sys.exit("timetable not found")
            
        slipDF = pd.DataFrame()
        for contract in executions.index:
            ibsym = executions.ix[contract].ibsym
            CSIsym=executions.ix[contract].CSIsym
            if ibsym in IB2CSI_multiplier_adj.keys():
                ib_price=executions.ix[contract].price*IB2CSI_multiplier_adj[ibsym]
            else:
                ib_price=executions.ix[contract].price
                
            ib_timestamp=executions.ix[contract].times
            qty =executions.ix[contract].qty
            #current contracts in contractsDF taken from system file which creates directly from csidata
            if CSIsym in futuresDF.index:
                csiPrice = futuresDF.ix[CSIsym].LastClose
                slippage=(ib_price-csiPrice)/csiPrice
                signal = system.ix[CSIsym].signal
                #for close trades. 
                if signal==0:
                    if executions.ix[contract].side == 'BOT':
                        signal=-1
                    else:
                        signal=1
                        
                closetime=timetable.ix[[x for x in idx_close if ibsym in x]][csidate][0]
                commissions = commission*qty
                cv = futuresDF.ix[CSIsym].contractValue
                dollarslip = int(-slippage*signal*qty*cv)
                #print contract, ib_price, csiPrice,slippage
                #rowName = str(ib_timestamp)+' ctwo:'+str(ib_price)+' csi:'+str(csiPrice)+' '+contract
                rowName = 'ib: '+str(ib_timestamp)+' '+str(qty)+' '+contract+' $'+str(dollarslip)
                slipDF.set_value(rowName, 'contract', contract)
                slipDF.set_value(rowName, 'ibsym', ibsym)
                slipDF.set_value(rowName, 'CSIsym', CSIsym)
                slipDF.set_value(rowName, 'ib_timestamp', ib_timestamp)
                slipDF.set_value(rowName, 'ib_price', ib_price)
                slipDF.set_value(rowName, 'closetime', closetime)
                slipDF.set_value(rowName, 'csiPrice', csiPrice)
                slipDF.set_value(rowName, 'slippage', slippage)
                slipDF.set_value(rowName, 'abs_slippage', abs(slippage))
                slipDF.set_value(rowName, 'Type', 'Open')
                slipDF.set_value(rowName, 'signal', signal)
                slipDF.set_value(rowName, 'dollarslip', dollarslip)
                slipDF.set_value(rowName, 'commissions', commissions)
                
        #slipDF['timedelta']=slipDF.ib_timestamp-slipDF.closetime
        slipDF['delta']=(slipDF.ib_timestamp-slipDF.closetime).astype('timedelta64[m]')
        #slipDF['delta']=slipDF.timedelta/np.timedelta64(1,'D')
        #if slipDF.shape[0] != portfolioDF.shape[0]:
        #    print 'Warning! Some values may be mising'
        slipDF= slipDF.sort_values(by='dollarslip', ascending=True)
        totalslip = int(slipDF.dollarslip.sum())
        filename=systemName+'_ib_slippage_'+str(csidate)+'.png'
        title = 'IB '+systemName+' '+selection+': '+str(slipDF.shape[0])+' Trades, $'+str(totalslip)\
                    +' Slippage, CSI Data as of '+str(csidate)
        plotSlip(slipDF, pngPath, filename, title, figsize, fontsize, showPlots=showPlots)

            
        slipDF.index.name = 'rowname'
        filename=systemName+'_ib_slippage_report_'+str(csidate).split()[0].replace('-','')+'.csv'
        slipDF = slipDF.sort_values(by='abs_slippage', ascending=True)
        slipDF.to_csv(savePath+systemName+'_ib_slippage_report.csv', index=True)
        print 'Saved '+savePath+systemName+'_ib_slippage_report.csv'
        slipDF.to_csv(savePath2+filename, index=True)
        print 'Saved '+savePath2+filename
        slipDF['Name']=systemName
        slipDF['Date']=csidate
        slipDF['timestamp']=int(calendar.timegm(dt.utcnow().utctimetuple()))
        slipDF.to_sql(name= 'ib_slippage', if_exists='append', con=writeConn, index=False)
        print 'Saved ib_slippage to',dbPathWrite
    else:
        print 'Only found executions for expired contracts'

'''
    ###########################################################
    #average slippage file/png
    files=os.listdir(portfolioPath)
    slipFiles = [x for x in files if systemName+'_slippage_report_' in x]
    slipFiles = [x for x in slipFiles if is_int(x.split('_')[-1].split('.')[0])]
    slipFiles = [x for x in slipFiles if int(x.split('_')[-1].split('.')[0])>start_slip]
    
    cons = pd.DataFrame()
    for f in slipFiles:
        fi = pd.read_csv(portfolioPath+f,index_col='rowname')
        cons=cons.append(fi)
    if cons.shape[0] !=0:
        avgslip=pd.DataFrame()
        for sym in cons.contract.unique():
            trades =len(cons[cons.contract==sym].c2timestamp.unique())
            abs_slip=abs(cons[cons.contract==sym].abs_slippage.mean())
            delta=cons[cons.contract==sym].delta.mean()
            #print sym, abs_slip
            avgslip.set_value(sym, 'slippage', abs_slip)
            avgslip.set_value(sym, 'delta', delta)
            avgslip.set_value(sym, 'trades', trades)
        avgslip=avgslip.sort_values(by='slippage', ascending=True)
        print systemName, str(avgslip.shape[0]), 'Symbols found in the average absolute slippage DF...'
        print avgslip.index.values

        index = str(trades)+'trades '+sym
        
        system_sym=[ x for x  in system[system.c2qty !=0].c2sym.values if x in avgslip.index.values]
        system_slip=avgslip.ix[system_sym].sort_values(by='slippage', ascending=True)
        system_slip.index = [str(int(system_slip.ix[i].trades))+' trades '+i for i in system_slip.index]
        filename=systemName+'_avg_slippage.png'
        if len(slipFiles)>1:
            title=systemName+' Avg. Slippage of '+str(system_slip.shape[0])+' Contracts from '\
                    +slipFiles[1].split('_')[3][:-4]+' to '+slipFiles[-1].split('_')[3][:-4]
        else:
            title=systemName+' Avg. Slippage of '+str(system_slip.shape[0])+' Contracts from '\
                    +slipFiles[-1].split('_')[3][:-4]
        plotSlip(system_slip, pngPath, filename, title,figsize, fontsize, showPlots=showPlots)
        #slippage by system


        filename=systemName+'_slip_cons.csv'
        cons.to_csv(savePath2+filename, index=True)
        print 'Saved '+savePath2+filename+'\n'
    else:
        print 'no slippage reports found. skipping consolidated report.'
'''

print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes ', dt.now()