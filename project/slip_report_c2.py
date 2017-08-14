import os
import numpy as np
import pandas as pd
import json
from pandas.io.json import json_normalize
import calendar
import time
import sys
from datetime import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import sqlite3

start_time = time.time()

if len(sys.argv)==1:
    debug=True
    showPlots=True
    createAvgSlip=False
    commission=2.5
    start_slip=20161230
    figsize=(6,8)
    fontsize=12
    dataPath='./data/'
    portfolioPath = './data/portfolio/'
    dbPathWrite='./data/futures.sqlite3' 
    dbPathRead = './data/futures.sqlite3'
    #savePath= 'D:/Dropbox/Python/TSDP/ml/data/results/' 
    savePath = savePath2 = pngPath='./ml/data/results/' 
    systemPath =  './data/systems/'
    #timetablePath=   './data/systems/timetables_debug/'
    timetablePath=   './data/systems/timetables/'
    feedfile='./data/systems/system_ibfeed.csv'
else:
    debug=False
    showPlots=False
    createAvgSlip=False
    commission=2.5
    start_slip=20161230
    figsize=(8,13)
    fontsize=20
    dbPathWrite=dbPathRead='./data/futures.sqlite3'
    dataPath=savePath='./data/'
    portfolioPath = savePath2 ='./data/portfolio/'
    pngPath = './web/tsdp/betting/static/images/'
    systemPath =  './data/systems/'
    timetablePath=   './data/systems/timetables/'
    feedfile='./data/systems/system_ibfeed.csv'

readConn= sqlite3.connect(dbPathRead)
writeConn = sqlite3.connect(dbPathWrite)

#accountInfo=pd.read_sql('select * from accountInfo where timestamp=\
#            (select max(timestamp) from accountInfo as maxtimestamp)', con=readConn,  index_col='index')
#systems = [x for x in accountInfo.columns if x not in ['Date','timestamp']]
webSelection=pd.read_sql('select * from webSelection where timestamp=\
        (select max(timestamp) from webSelection)', con=readConn)
systems = eval(webSelection.selection[0]).keys()
#selectionDict=eval(webSelection.selection.values[0])
#atrFilename = 'futuresATR.csv'
#futuresDF = pd.read_csv(dataPath+atrFilename, index_col=0)
#prevTriggerCutoffGuess = dt.strptime(futuresDF.index.name, '%Y-%m-%d %H:%M:%S').replace(hour=20)
#prevTriggerCutoffGuess = prevTriggerCutoffGuess.replace(day=prevTriggerCutoffGuess.day-1)
#csidate = futuresDF.index.name.split()[0].replace('-','')

csidates=pd.read_sql('select distinct Date from futuresDF_results', con=readConn).Date.tolist()
#assuming we run this script after csi download and vol_adjsize.
prevcsidate=str(csidates[-2])
csidate=str(csidates[-1])
futuresDF=pd.read_sql('select * from futuresDF_results where timestamp=\
            (select max(timestamp) from futuresDF_results as maxtimestamp) and Date=%s' % (csidate), con=readConn,  index_col='CSIsym')
#prevTriggerCutoffGuess=pd.read_sql('select distinct Date from futuresDF_all order by Date ASC', con=readConn).Date.tolist()[-2]
triggerCutoffGuess = dt.strptime(str(csidate), '%Y%m%d').replace(hour=17)
prevTriggerCutoffGuess = dt.strptime(str(prevcsidate), '%Y%m%d').replace(hour=17)
#prevTriggerCutoffGuess = prevTriggerCutoffGuess.replace(day=prevTriggerCutoffGuess.day-1)
feeddata=pd.read_csv(feedfile,index_col='c2sym')

fixed_signals = ['RiskOn','RiskOff','Custom','AntiCustom','LastSEA','AntiSEA','AdjSEA','AntiAdjSEA']

signals = ['ACT','prevACT','AntiPrevACT','RiskOn','RiskOff','Custom','AntiCustom',\
                'LastSIG', '0.75LastSIG','0.5LastSIG','1LastSIG','Anti1LastSIG','Anti0.75LastSIG','Anti0.5LastSIG',\
                'LastSEA','AntiSEA','AdjSEA','AntiAdjSEA',\
                'Voting','Voting2','Voting3','Voting4','Voting5','Voting6','Voting7','Voting8','Voting9',\
                'Voting10','Voting11','Voting12','Voting13','Voting14','Voting15']
#adjustments
adjDict={
            '@CT':100,
            '@JY':0.01,
            'QSI':0.01
            }
def is_int(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False
        
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
    


#import timetable
#ttfiles = os.listdir(timetablePath)
#ttdates = []
#for f in ttfiles:
#    if '.csv' in f and is_int(f.split('.')[0]):
#        ttdates.append(int(f.split('.')[0]))
#ttdate=max(ttdates)
if os.path.isfile(timetablePath+str(csidate)+'.csv'):
    timetable = pd.read_csv(timetablePath+str(csidate)+'.csv', index_col=0)
    for col in timetable.columns:
        timetable[col]=pd.to_datetime(timetable[col])
    idx_close = [x for x in timetable.index if 'close' in x]
    idx_trigger = [x for x in timetable.index if 'trigger' in x]
else:
    print 'timetable for', csidate,'not found! Ending script.'
    systems=[]


for systemName in systems:
    print '\n\n',systemName
    #selection=selectionDict[systemName][0]
    tradeFilename='c2_'+systemName+'_trades.csv'
    portfolioFilename = 'c2_'+systemName+'_portfolio.csv'
    #systemFilename='system_'+systemName+'.csv'
    #system = pd.read_csv(systemPath+systemFilename)
    #csidate='20170330'
    tablename=systemName+'_live'
    system=pd.read_sql('select * from (select * from {} where Date={} order by timestamp ASC)\
                                    group by CSIsym'.format(tablename,csidate), con=readConn)
    if system.shape[0]<1:
        print 'csidate not in', tablename
        continue
    selection=system.selection[0]
    #entry trades
    portfolioDF = pd.read_csv(portfolioPath+portfolioFilename)
    portfolioDF.index = feeddata.ix[[x[:-2] for x in portfolioDF.symbol.values]].ibsym.values
    portfolioDF = portfolioDF.ix[[True if type(x)==str else False for x in portfolioDF.index]]
    portfolioDF['openedWhen'] = pd.to_datetime(portfolioDF['openedWhen'])
    portfolioDF = portfolioDF.sort_values(by='openedWhen', ascending=False)
    #exit trades
    tradesDF = pd.read_csv(portfolioPath+tradeFilename)
    tradesDF.index = feeddata.ix[[x[:-2] for x in tradesDF.symbol.values]].ibsym.values
    tradesDF=tradesDF.ix[[True if type(x)==str else False for x in tradesDF.index]]
    tradesDF=tradesDF.drop(['expir','putcall','strike','symbol_description','underlying','markToMarket_time'], axis=1).dropna()
    tradesDF['closedWhen'] = pd.to_datetime(tradesDF['closedWhen'])
    tradesDF=tradesDF[tradesDF.closedWhen > prevTriggerCutoffGuess].copy()
    tradesDF = tradesDF.sort_values(by='closedWhen', ascending=False)
    #csi close data
    
    #csidata download at 8pm est
    slipDF = pd.DataFrame()
    #print 'sym', 'c2price', 'csiPrice', 'slippage'
    #new entry trades
    #newOpen=portfolioDF[portfolioDF['openedWhen']>=futuresDate].symbol.values
    newOpen={}
    system.index= system.c2sym
    for index in idx_close:
        sym=index.split()[0]
        if sym in portfolioDF.index:
            if portfolioDF.index.tolist().count(sym) == 1:
                row=portfolioDF.ix[sym].copy()
            else:
                row=portfolioDF.ix[sym].iloc[0].copy()
            if csidate in timetable:
                timestamp=timetable.ix[index][csidate]
                print sym,
                #print sym,'close', timestamp,
                if row.openedWhen>=timestamp:
                    #print 'new open', row.openedWhen,'>=',index,timestamp,'adding.. opened after current csi close',csidate
                    print 'adding.. opened after ',csidate,' timetable close',timestamp,'opened',row.openedWhen
                    newOpen[row.symbol]=[timestamp, row.quant_opened]
                else:
                    if prevcsidate in timetable:
                        #prevcsidate may not be in timetable
                        prev_timestamp=timetable.ix[index][prevcsidate]
                        if row.openedWhen>=prev_timestamp:
                            print 'adding.. ',selection,'opened after prevcsidate timetable close',prev_timestamp,'opened',row.openedWhen
                            newOpen[row.symbol]=[timestamp, row.quant_opened]
                        else:
                            print 'skipping..', selection,'opened before prevcsidate timetable close',prev_timestamp, 'opened', row.openedWhen
                    else:
                        if row.openedWhen>prevTriggerCutoffGuess:
                            print 'adding..', selection,'opened after prevTriggerCutoffGuess',prevTriggerCutoffGuess, 'opened', row.openedWhen
                            newOpen[row.symbol]=[timestamp, row.quant_opened]
                        else:
                            print 'skipping..', selection,'opened before prevTriggerCutoffGuess',prevTriggerCutoffGuess, 'opened', row.openedWhen

            else:
                if row.openedWhen>prevTriggerCutoffGuess:
                    print 'adding..', selection,'opened after prevTriggerCutoffGuess',prevTriggerCutoffGuess, 'opened', row.openedWhen
                    newOpen[row.symbol]=[triggerCutoffGuess, row.quant_opened]
                else:
                    print 'skipping..', selection,'opened before prevTriggerCutoffGuess',prevTriggerCutoffGuess, 'opened', row.openedWhen
        else:
            #print sym,' no open trades found skipping..'
            pass
            
    print systemName, len(newOpen.keys()), 'Open Trades Found'
    print systemName, newOpen.keys(),'\n'
    
    newCloses=pd.DataFrame()
    for index in idx_close:
        sym=index.split()[0]
        if sym in tradesDF.index:
            if tradesDF.index.tolist().count(sym) == 1:
                row=tradesDF.ix[sym].copy()
            else:
                row=tradesDF.ix[sym].iloc[0].copy()

            if csidate in timetable:
                row['closetime']=timetable.ix[index][csidate]
                print sym,
                #print sym,'close', timetable.ix[index][csidate],
                if row.closedWhen>=timetable.ix[index][csidate]:
                    #print 'new close', row.closedWhen,'>=',index,timetable.ix[index][csidate],'adding.. closed after current csi close',csidate
                    print 'adding.. closed after current timetable close',timetable.ix[index][csidate],'closed',row.closedWhen
                    
                    newCloses = newCloses.append(row)
                else:
                    if prevcsidate in timetable:
                        #prevcsidate may not be in timetable
                        timestamp=timetable.ix[index][prevcsidate]
                        if row.closedWhen>=timestamp:
                            print 'adding.. ',selection,'closed after prevcsidate timetable close',timestamp,'opened',row.closedWhen
                            
                            newCloses = newCloses.append(row)
                        else:
                            print 'skipping..', selection,'closed before prevcsidate timetable close',timestamp, 'opened', row.closedWhen
                    else:
                        if row.closedWhen>prevTriggerCutoffGuess:
                            print 'adding..', selection,'opened after prevTriggerCutoffGuess',prevTriggerCutoffGuess, 'closed', row.closedWhen
                            
                            newCloses = newCloses.append(row)
                        else:
                            print 'skipping..', selection,'closed before prevTriggerCutoffGuess',prevTriggerCutoffGuess, 'closed', row.closedWhen
                    
            else:
                #use guess
                row['closetime']=triggerCutoffGuess
                print index,csidate,'not found in timetable,',
                if row.closedWhen>prevTriggerCutoffGuess:
                    print 'adding..', selection,'opened after prevTriggerCutoffGuess',prevTriggerCutoffGuess, 'closed', row.closedWhen
                    newCloses = newCloses.append(row)
                else:
                    print 'skipping..', selection,'closed before prevTriggerCutoffGuess',prevTriggerCutoffGuess, 'closed', row.closedWhen
        else:
            #print sym,' no close trades found skipping..'
            pass
            
    print systemName, newCloses.shape[0], 'Close Trades Found'
    if newCloses.shape[0]>0:
        newCloses.index = newCloses.symbol
        print systemName, newCloses.index
        for sym in portfolioDF.symbol:
            if sym in newCloses.index:
                #only include close trades where new signal is 0, otherwise slippage should cancel out.
                print sym, 'new signal not 0. dropping.'
                newCloses =  newCloses.drop([sym], axis=0)
    print systemName, newCloses.shape[0], 'Close Trades for slip report\n'
            

    for contract in newOpen.keys():
        c2price=portfolioDF[portfolioDF.symbol ==contract].opening_price_VWAP.values[0]
        c2timestamp=pd.Timestamp(portfolioDF[portfolioDF.symbol ==contract].openedWhen.values[0])
        if contract in futuresDF.Contract.values:
            if contract[:-2] in adjDict.keys():
                csiPrice = futuresDF[futuresDF.Contract ==contract].LastClose.values[0]*adjDict[contract[:-2]]
            else:
                csiPrice = futuresDF[futuresDF.Contract ==contract].LastClose.values[0]
            slippage=(c2price-csiPrice)/csiPrice
            signal = system.ix[contract].signal
            qty =newOpen[contract][1]
            commissions = commission*qty
            cv = futuresDF[futuresDF.Contract ==contract].contractValue.values[0]
            dollarslip = int(-slippage*signal*qty*cv)
            #print contract, c2price, csiPrice,slippage
            #rowName = str(c2timestamp)+' ctwo:'+str(c2price)+' csi:'+str(csiPrice)+' '+contract
            rowName = 'ctwo opened: '+str(c2timestamp)+' '+str(qty)+' '+contract+' $'+str(dollarslip)
            slipDF.set_value(rowName, 'contract', contract)
            slipDF.set_value(rowName, 'c2timestamp', c2timestamp)
            slipDF.set_value(rowName, 'c2price', c2price)
            slipDF.set_value(rowName, 'closetime', newOpen[contract][0])
            slipDF.set_value(rowName, 'csiPrice', csiPrice)
            slipDF.set_value(rowName, 'slippage', slippage)
            slipDF.set_value(rowName, 'abs_slippage', abs(slippage))
            slipDF.set_value(rowName, 'Type', 'Open')
            slipDF.set_value(rowName, 'signal', signal)
            slipDF.set_value(rowName, 'dollarslip', dollarslip)
            slipDF.set_value(rowName, 'commissions', commissions)
    #newCloses=tradesDF[tradesDF['closedWhen']>=futuresDate]
    
    for contract in newCloses.index:
        c2price=newCloses.ix[contract].closing_price_VWAP
        c2timestamp=pd.Timestamp(newCloses.ix[contract].closedWhen)
        closetime = pd.Timestamp(newCloses.ix[contract].closetime)
        if contract in futuresDF.Contract.values:
            if contract[:-2] in adjDict.keys():
                csiPrice = futuresDF[futuresDF.Contract ==contract].LastClose.values[0]*adjDict[contract[:-2]]
            else:
                csiPrice = futuresDF[futuresDF.Contract ==contract].LastClose.values[0]
            slippage=(c2price-csiPrice)/csiPrice
            if newCloses.ix[contract].long_or_short == 'long':
                signal=-1
            else:
                signal =1
            qty = newCloses.ix[contract].quant_closed
            commissions = commission*qty
            cv = futuresDF[futuresDF.Contract ==contract].contractValue.values[0]
            dollarslip = int(-slippage*signal*qty*cv)
            #print contract, c2price, csiPrice,slippage
            #rowName = str(c2timestamp)+' ctwo:'+str(c2price)+' csi:'+str(csiPrice)+' '+contract
            rowName = 'ctwo closed: '+str(c2timestamp)+' '+str(qty)+' '+contract+' $'+str(dollarslip)
            slipDF.set_value(rowName, 'contract', contract)
            slipDF.set_value(rowName, 'c2timestamp', c2timestamp)
            slipDF.set_value(rowName, 'c2price', c2price)
            slipDF.set_value(rowName, 'closetime', closetime)
            slipDF.set_value(rowName, 'csiPrice', csiPrice)
            slipDF.set_value(rowName, 'slippage', slippage)
            slipDF.set_value(rowName, 'abs_slippage', abs(slippage))
            slipDF.set_value(rowName, 'Type', 'Close')
            slipDF.set_value(rowName, 'signal', signal)
            slipDF.set_value(rowName, 'dollarslip', dollarslip)
            slipDF.set_value(rowName, 'commissions', commissions)
            
    if slipDF.shape[0]==0:
        print systemName, 'No new trades yesterday, skipping daily report'
    else:
        slipDF['timedelta']=slipDF.c2timestamp-slipDF.closetime
        slipDF['delta']=slipDF['timedelta'].astype('timedelta64[m]')
        #slipDF['delta']=slipDF.timedelta/np.timedelta64(1,'D')
        #if slipDF.shape[0] != portfolioDF.shape[0]:
        #    print 'Warning! Some values may be mising'

        
        slipTrades = slipDF.sort_values(by='dollarslip', ascending=True)
        totalslip = int(slipTrades.dollarslip.sum())
        filename=systemName+'_c2_slippage_'+str(csidate)+'.png'
        title = 'C2 ' +systemName+' '+selection+': '+str(slipTrades.shape[0])+' Trades, $'+str(totalslip)\
                    +' Slippage, CSI Data as of '+str(csidate)
        if slipTrades.shape[0] !=0:
            plotSlip(slipTrades, pngPath, filename, title, figsize, fontsize, showPlots=showPlots)
        else:
            print title
        '''
        closedTrades = slipDF[slipDF['Type']=='Close'].sort_values(by='abs_slippage', ascending=True)
        totalslip = int(closedTrades.dollarslip.sum())
        title = systemName+': '+str(closedTrades.shape[0])+' Closed Trades, $'+str(totalslip)\
                    +' Slippage, CSI Data as of '+str(csidate)
        filename=systemName+'_close_slippage.png'
        if closedTrades.shape[0] != 0:
            plotSlip(closedTrades, pngPath, filename, title, figsize, fontsize, showPlots=showPlots)
        else:
            print title
        '''
        slipDF.index.name = 'rowname'
        filename=systemName+'_slippage_report_'+str(csidate).split()[0].replace('-','')+'.csv'
        slipDF = slipDF.sort_values(by='abs_slippage', ascending=True)
        slipDF.to_csv(savePath+systemName+'_slippage_report.csv', index=True)
        print 'Saved '+savePath+systemName+'_slippage_report.csv'
        slipDF.to_csv(savePath2+filename, index=True)
        print 'Saved '+savePath2+filename
        slipDF['Name']=systemName
        slipDF['ibsym']=feeddata.ix[[x[:-2] for x in slipDF.contract.values]].ibsym.values
        slipDF['csiDate']=csidate
        slipDF['timestamp']=int(calendar.timegm(dt.utcnow().utctimetuple()))
        slipDF.index = feeddata.ix[[x[:-2] for x in slipDF.contract.values]].CSIsym.values
        slipDF.index.name = 'CSIsym'
        slipDF.to_sql(name= 'slippage', if_exists='append', con=writeConn, index=True, index_label='CSIsym')
        print 'Saved '+systemName+' to sql'

    ###########################################################
    if createAvgSlip:
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
            '''
            i=0
            for x in futuresDF[futuresDF.finalQTY != 0].Contract:
                if x not in cons.symbol.unique():
                    i+=1
                    print x,
            print i, 'Symbols missing!'
            '''
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

print 'Elapsed time: ', round(((time.time() - start_time)/60),2), ' minutes ', dt.now()