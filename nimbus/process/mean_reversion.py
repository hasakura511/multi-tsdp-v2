#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 15:05:04 2018

mean reversion classifier


@author: hidemiasakura
"""

import math
import numpy as np
from numpy import log
import pandas as pd
import arch
from nimbus.process.transform import roofing_filter
from os import listdir
from os.path import isfile, join
import statsmodels.tsa.stattools as ts
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
from cycler import cycler
import seaborn as sns

class MeanReversion(object):
    def __init__(self):
        pass
    
    def garch(self, returns, verbose=False):
        am = arch.arch_model(returns*100)
        #am = arch.arch_model(returns*100, p=1, o=1, q=1)
        #am = arch.arch_model(returns*100, p=1, o=1, q=1, power=1.0)
        #am = arch.arch_model(returns*100, p=1, o=1, q=1, power=1.0, dist='StudentsT')
        #res = am.fit(iter=10)
        res = am.fit(disp='off')
        #res.plot()
        forecast = np.sqrt(res.params['omega'] + res.params['alpha[1]'] * res.resid**2 + res.conditional_volatility**2 * res.params['beta[1]'])
        if verbose:
            print(res.summary())
        return forecast
        
    def garch2(self, returns, verbose=False):
        #am = arch.arch_model(returns*100)
        am = arch.arch_model(returns*100, p=1, o=1, q=1)
        #am = arch.arch_model(returns*100, p=1, o=1, q=1, power=1.0)
        #am = arch.arch_model(returns*100, p=1, o=1, q=1, power=1.0, dist='StudentsT')
        #res = am.fit(iter=10)
        res = am.fit(disp='off')
        #res.plot()
        forecast = np.sqrt(res.params['omega'] + res.params['alpha[1]'] * res.resid**2 + res.conditional_volatility**2 * res.params['beta[1]'])
        if verbose:
            print(res.summary())
        return forecast
        
    def garch3(self, returns, verbose=False):
        #am = arch.arch_model(returns*100)
        #am = arch.arch_model(returns*100, p=1, o=1, q=1)
        am = arch.arch_model(returns*100, p=1, o=1, q=1, power=1.0)
        #am = arch.arch_model(returns*100, p=1, o=1, q=1, power=1.0, dist='StudentsT')
        #res = am.fit(iter=10)
        res = am.fit(disp='off')
        #res.plot()
        forecast = np.sqrt(res.params['omega'] + res.params['alpha[1]'] * res.resid**2 + res.conditional_volatility**2 * res.params['beta[1]'])
        if verbose:
            print(res.summary())
        return forecast
        
    def garch4(self, returns, verbose=False):
        #am = arch.arch_model(returns*100)
        #am = arch.arch_model(returns*100, p=1, o=1, q=1)
        #am = arch.arch_model(returns*100, p=1, o=1, q=1, power=1.0)
        am = arch.arch_model(returns*100, p=1, o=1, q=1, power=1.0, dist='StudentsT')
        #res = am.fit(iter=10)
        res = am.fit(disp='off')
        #res.plot()
        forecast = np.sqrt(res.params['omega'] + res.params['alpha[1]'] * res.resid**2 + res.conditional_volatility**2 * res.params['beta[1]'])
        if verbose:
            print(res.summary())
        return forecast
        
    def adf(self, p,bars, threshold=-1.5, showPlot=False, ticker='', savePath=None):
        #if len(p)-bars <3:
        #    bars=len(p)-3
        index = p.index
        index2=p.index[bars:]
        #returns = p.pct_change().fillna(0)
    
        #print rfGarch.shape, p.shape
        if type(p) is pd.core.series.Series:
            p = p.values
            
        nrows=p.shape[0]
        #Dimen=np.zeros(nrows)
        #Hurst=np.zeros(nrows)
        #SmoothHurst=np.zeros(nrows)
        #gar=np.zeros(nrows)
        #gar[:bars]=garch(returns[:bars])
        #gar2=np.zeros(nrows)
        #gar2[:bars]=garch2(returns[:bars])
        #gar3=np.zeros(nrows)
        #gar3[:bars]=garch3(returns[:bars])
        #gar4=np.zeros(nrows)
        #gar4[:bars]=garch4(returns[:bars])
        adfpv=np.zeros(nrows)
        adfClass=np.zeros(nrows)
        #SmoothGarch=np.zeros(nrows)
        #minmaxGarch=np.zeros(nrows)
        #minmaxGarch2=np.zeros(nrows)
        #minmaxGarch3=np.zeros(nrows)
        #minmaxGarch4=np.zeros(nrows)
        #if bars%2>0:
        #    bars=bars-1
            
        a1 = math.exp(-math.sqrt(2)*math.pi/10.0)
        b1 = 2.0*a1*math.cos(math.sqrt(2)*math.radians(180)/10.0)
    
        c3 = -a1*a1
        c2 = b1
        c1 = 1-c2-c3
    
        for i,lb in enumerate(range(bars, nrows)):
            '''
            #print i,lb
            N3 = (max(p[i:lb]) - min(p[i:lb]))/float(bars)
            N2 = (max(p[i:lb-bars/2]) - min(p[i:lb-bars/2]))/float(bars/2)
            #print i, lb-bars/2, p[i:lb-bars/2]
            N1 = (max(p[lb-bars/2:lb]) - min(p[lb-bars/2:lb]))/float(bars/2)
            #print p[lb-bars/2:lb]
            if N1>0 and N2>0 and N3>0:
                Dimen[lb] = .5*((log(N1+N2)-log(N3))/log(2)+Dimen[lb-1])
            Hurst[lb]=2-Dimen[lb]
            #print Hurst
            SmoothHurst[lb]=c1*(Hurst[lb]+Hurst[lb-1])/2+c2*SmoothHurst[lb-1]\
                                    +c3*SmoothHurst[lb-2]
            '''
            #gar[lb] = garch(returns[:lb])[-1]
            #minmaxGarch[lb] =minmax_scale(gar[i:lb]) [-1]
            #gar2[lb] = garch2(returns[:lb])[-1]
            #minmaxGarch2[lb] =minmax_scale(gar2[i:lb]) [-1]
            #gar3[lb] = garch3(returns[:lb])[-1]
            #minmaxGarch3[lb] =minmax_scale(gar3[i:lb]) [-1]
            #gar4[lb] = garch4(returns[i:lb])[-1]
            #mmG=minmax_scale(garch4(returns[i:lb]))
            #minmaxGarch4[lb] =mmG[-1]
            #SmoothGarch[lb]=c1*(mmG[-1]+mmG[-2])/2\
            #                +c2*SmoothGarch[lb-1]\
            #                +c3*SmoothGarch[lb-2]
            adf = ts.adfuller(p[i:lb],1)
            adfpv[lb]=adf[0]
    
            if adf[0]<threshold:
                adfClass[lb]=0
            else:
                adfClass[lb]=1
    
            #print gar4[lb], len(returns[i:lb]), minmaxGarch4[lb]
            
        #SmoothGarch= roofingFilter(gar,bars)
        #SmoothGarch = np.nan_to_num(SmoothGarch)
        #softmaxGarch = minmax_scale
        #softmaxGarch = softmax(gar,bars,1)
        #softmaxGarch = softmax_score(gar)
        #print minmaxGarch
        #print gar4, minmaxGarch4
        #SmoothGarch= roofingFilter(gar,bars)
        #to return
        #SmoothHG = np.maximum(SmoothHurst,minmaxGarch)
        #scaledVolatility = minmaxGarch
        #scaledVolatility2 = minmaxGarch2
        #scaledVolatility3 = minmaxGarch3
        #scaledVolatility4 = SmoothGarch
        #scaledVolatility = np.minimum(SmoothHurst,SmoothGarch)
        #modes = np.where(scaledVolatility<threshold,0,1)
        #modes2 = np.where(scaledVolatility2<threshold,0,1)
        #modes3 = np.where(scaledVolatility3<threshold,0,1)
        #modes4 = np.where(scaledVolatility4<threshold,0,1)
        #modes = np.where(minmaxGarch<threshold,0,1)
        #mode2 = np.where(Hurst[bars:]<threshold,0,1)
        modes4=adfClass
        #mode = modes[bars:]
        #mode2 = modes2[bars:]
        #mode3 = modes3[bars:]
        mode4 = modes4[bars:]
        adfpv= adfpv[bars:]
        p=p[bars:]
        index = index2
        nrows=p.shape[0]
        #print nrows, len(p)
        #Hurst=Hurst[bars:]
        #SmoothHurst=SmoothHurst[bars:]
        #SmoothGarch=SmoothGarch[bars:]
        #scaledVolatility=scaledVolatility[bars:]
        #scaledVolatility2=scaledVolatility2[bars:]
        #scaledVolatility3=scaledVolatility3[bars:]
        #scaledVolatility4=scaledVolatility4[bars:]
        #print gar, minmaxGarch, SmoothGarch
        #minmaxGarch=minmaxGarch[bars:]
        #gar=gar[bars:]
        def format_date(x, pos=None):
            thisind = np.clip(int(x + 0.5), 0, nrows - 1)
            #print thisind,index
            return index[thisind].strftime("%Y-%m-%d %H:%M")
            
    
        fig = plt.figure()
        ax = fig.add_subplot(111, xlim=(0, len(p)), ylim=(p.min()*0.99, p.max()*1.01))
        #print bars, len(p),p
        ax.plot(np.arange(len(p)), p, 'r-', alpha=0.5)
        ax.scatter(np.arange(len(p))[mode4 == 0], p[mode4 == 0], color='g', label='0 CycleMode')
        ax.scatter(np.arange(len(p))[mode4 == 1], p[mode4 == 1], color='r', label='1 TrendMode')
    
        handles, labels = ax.get_legend_handles_labels()
        lgd2 = ax.legend(handles, labels, loc='upper right',prop={'size':10})
        #ax.plot(np.arange(len(p))[self.pivots != 0], p[self.pivots != 0], 'k-')
        #ax.scatter(np.arange(len(p))[self.pivots == 1], p[self.pivots == 1], color='g')
        #ax.scatter(np.arange(len(p))[self.pivots == -1], p[self.pivots == -1], color='r')
        ax.xaxis.set_major_formatter(tick.FuncFormatter(format_date))
        
        #annotate last index
        ax.annotate(index[-1].strftime("%Y-%m-%d %H:%M"),\
                    xy=(0.79, 0.02), ha='left', va='top', xycoords='axes fraction', fontsize=12)
                    
        ax2=ax.twinx()
        #same color to dashed and non-dashed
        #ax2.set_prop_cycle(cycler('color', sorted(sns.color_palette("husl", 4))))
        #ax2.plot(np.arange(len(p)),Hurst,color='b', label='Hurst')
        #ax2.plot(np.arange(len(p)),SmoothHurst,color='b', label='smoothHurst')
        #ax2.plot(np.arange(len(p)),SmoothGarch, label='smoothGarch')
        #ax2.plot(np.arange(len(p)),scaledVolatility, label='scaledVolatility')
        #ax2.plot(np.arange(len(p)),scaledVolatility2, label='scaledVolatility2')
        #ax2.plot(np.arange(len(p)),scaledVolatility3, label='scaledVolatility3')
        #ax2.plot(np.arange(len(p)),scaledVolatility4, label='scaledVolatility4')
        ax2.plot(np.arange(len(p)),adfpv, label='adf pvalue')
        t = np.zeros(len(p))
        t.fill(threshold)
        ax2.plot(np.arange(len(p)),t,color='k', label='threshold')
        #ax2.plot(np.arange(len(p)),minmaxGarch,color='c', label='minmaxGarch')
        #ax2.plot(np.arange(len(p)),gar,color='b', label='Garch')
        handles, labels = ax2.get_legend_handles_labels()
        #lgd2 = ax2.legend(handles, labels, loc='lower right',prop={'size':10})
        #ax2.scatter(np.arange(len(p))[mode2 == 0], p[mode2 == 0], color='b', label='hurstCycleMode')
        #ax2.scatter(np.arange(len(p))[mode2 == 1], p[mode2 == 1], color='y', label='hurstTrendMode')
    
        #ax2.plot(np.arange(nrows),dpsEquity, label='dps '+system, ls=next(linecycle))
        #ax2.xaxis.set_major_formatter(tick.FuncFormatter(format_date))
        if mode4[-1] ==0:
            title = ticker+' Mean Reverting: Cycle Mode '
        else:
            title = ticker+' Non-Mean Reverting: Trend Mode '
        ax2.set_title(title+str(bars)+' bars, threshold '+str(threshold))
        ax.set_xlim(0, nrows)        
        ax2.set_xlim(0, nrows)       
        fig.autofmt_xdate()
        if showPlot:
            plt.show()
        #if savePath != None:
        #    print 'Saving '+savePath+'.png'
        #    fig.savefig(savePath+'.png', bbox_inches='tight')
        plt.close()
            
        return modes4
    
    def adf2(self, p, bars, threshold=-1.5, showPlot=False, ticker='', savePath=None):
        #if len(p)-bars <3:
        #    bars=len(p)-3
        index = p.index
        index2=p.index[bars*2:]
        returns = p.pct_change().fillna(0)
    
        #print rfGarch.shape, p.shape
        if type(p) is pd.core.series.Series:
            p = p.values
            
        nrows=p.shape[0]
        Dimen=np.zeros(nrows)
        Hurst=np.zeros(nrows)
        SmoothHurst=np.zeros(nrows)
        gar=np.zeros(nrows)
        gar[:bars]=self.garch(returns[:bars])
        #gar2=np.zeros(nrows)
        #gar2[:bars]=garch2(returns[:bars])
        #gar3=np.zeros(nrows)
        #gar3[:bars]=garch3(returns[:bars])
        #gar4=np.zeros(nrows)
        #gar4[:bars]=garch4(returns[:bars])
        adfpv=np.zeros(nrows)
        adfClass=np.zeros(nrows)
        #SmoothGarch=np.zeros(nrows)
        #minmaxGarch=np.zeros(nrows)
        #minmaxGarch2=np.zeros(nrows)
        #minmaxGarch3=np.zeros(nrows)
        #minmaxGarch4=np.zeros(nrows)
        #if bars%2>0:
        #    bars=bars-1
            
        a1 = math.exp(-math.sqrt(2)*math.pi/10.0)
        b1 = 2.0*a1*math.cos(math.sqrt(2)*math.radians(180)/10.0)
    
        c3 = -a1*a1
        c2 = b1
        c1 = 1-c2-c3
    
        for i,lb in enumerate(range(bars, nrows)):
            
            #print i,lb
            N3 = (max(p[i:lb]) - min(p[i:lb]))/float(bars)
            self.p=p
            #print(i,lb,bars)
            end=int(bars/2)
            N2 = (max(p[i:lb-end]) - min(p[i:lb-end]))/float(end)
            #print i, lb-bars/2, p[i:lb-bars/2]
            N1 = (max(p[lb-end:lb]) - min(p[lb-end:lb]))/float(end)
            #print p[lb-bars/2:lb]
            if N1>0 and N2>0 and N3>0:
                Dimen[lb] = .5*((log(N1+N2)-log(N3))/log(2)+Dimen[lb-1])
            Hurst[lb]=2-Dimen[lb]
            #print Hurst
            SmoothHurst[lb]=c1*(Hurst[lb]+Hurst[lb-1])/2+c2*SmoothHurst[lb-1]\
                                    +c3*SmoothHurst[lb-2]
            
            #gar[lb] = garch(returns[:lb])[-1]
            #minmaxGarch[lb] =minmax_scale(gar[i:lb]) [-1]
            #gar2[lb] = garch2(returns[:lb])[-1]
            #minmaxGarch2[lb] =minmax_scale(gar2[i:lb]) [-1]
            #gar3[lb] = garch3(returns[:lb])[-1]
            #minmaxGarch3[lb] =minmax_scale(gar3[i:lb]) [-1]
            #gar4[lb] = garch4(returns[i:lb])[-1]
            #mmG=minmax_scale(garch4(returns[i:lb]))
            #minmaxGarch4[lb] =mmG[-1]
            #SmoothGarch[lb]=c1*(mmG[-1]+mmG[-2])/2\
            #                +c2*SmoothGarch[lb-1]\
            #                +c3*SmoothGarch[lb-2]
            adf = ts.adfuller(p[i:lb],1)
            adfpv[lb]=adf[0]
            #print "Test-Stat", adf[0]
            #for key in adf[4]:
            #    print "Critical Values:",key, adf[4][key],
            #    if adf[0] < adf[4][key]:
            #        print 'PASS'
            #    else:
            #        print 'FAIL'
    
            if adf[0]<threshold:
                adfClass[lb]=0
            else:
                adfClass[lb]=1
    
            #print gar4[lb], len(returns[i:lb]), minmaxGarch4[lb]
            
        #SmoothGarch= roofingFilter(gar,bars)
        #SmoothGarch = np.nan_to_num(SmoothGarch)
        #softmaxGarch = minmax_scale
        #softmaxGarch = softmax(gar,bars,1)
        #softmaxGarch = softmax_score(gar)
        #print minmaxGarch
        self.SmoothHurst= roofing_filter(SmoothHurst[bars:],bars)
        self.SmoothADF= roofing_filter(adfpv[bars:],bars)
        #print SmoothHurst, SmoothHurst.shape
        #print adfpv[bars:], '\n', SmoothADF[bars:], SmoothADF[bars:].shape
        #to return
        #SmoothHG = np.maximum(SmoothHurst,minmaxGarch)
        #scaledVolatility = minmaxGarch
        #scaledVolatility2 = minmaxGarch2
        #scaledVolatility3 = minmaxGarch3
        #scaledVolatility4 = SmoothGarch
        #scaledVolatility = np.minimum(SmoothHurst,SmoothGarch)
        #modes = np.where(scaledVolatility<threshold,0,1)
        #modes2 = np.where(scaledVolatility2<threshold,0,1)
        #modes3 = np.where(scaledVolatility3<threshold,0,1)
        modes4 = np.where((self.SmoothADF<threshold) & (self.SmoothADF>(1-threshold)),0,1)
        #modes = np.where(minmaxGarch<threshold,0,1)
        #mode2 = np.where(Hurst[bars:]<threshold,0,1)
        #modes4=adfClass
        #mode = modes[bars:]
        #mode2 = modes2[bars:]
        #mode3 = modes3[bars:]
        mode4 = modes4[bars:]
        #adfpv= adfpv[bars*2:]
        SmoothADF = self.SmoothADF[bars:]
        SmoothHurst=self.SmoothHurst[bars:]
        p=p[bars*2:]
        index = index2
        #print index, index.shape
        nrows=p.shape[0]
        #print nrows, len(p)
        #Hurst=Hurst[bars:]
        #SmoothHurst=SmoothHurst[bars:]
        #SmoothGarch=SmoothGarch[bars:]
        #scaledVolatility=scaledVolatility[bars:]
        #scaledVolatility2=scaledVolatility2[bars:]
        #scaledVolatility3=scaledVolatility3[bars:]
        #scaledVolatility4=scaledVolatility4[bars:]
        #print gar, minmaxGarch, SmoothGarch
        #minmaxGarch=minmaxGarch[bars:]
        #gar=gar[bars:]
        def format_date(x, pos=None):
            thisind = np.clip(int(x + 0.5), 0, nrows - 1)
            #print thisind,index
            return index[thisind].strftime("%Y-%m-%d %H:%M")
            
    
        fig = plt.figure()
        ax = fig.add_subplot(111, xlim=(0, len(p)), ylim=(p.min()*0.99, p.max()*1.01))
        #print bars, len(p),p
        ax.plot(np.arange(len(p)), p, 'r-', alpha=0.5)
        ax.scatter(np.arange(len(p))[mode4 == 0], p[mode4 == 0], color='g', label='0 TrendMode')
        ax.scatter(np.arange(len(p))[mode4 == 1], p[mode4 == 1], color='r', label='1 Counter-TrendMode')
    
        handles, labels = ax.get_legend_handles_labels()
        lgd2 = ax.legend(handles, labels, loc='upper right',prop={'size':10})
        #ax.plot(np.arange(len(p))[self.pivots != 0], p[self.pivots != 0], 'k-')
        #ax.scatter(np.arange(len(p))[self.pivots == 1], p[self.pivots == 1], color='g')
        #ax.scatter(np.arange(len(p))[self.pivots == -1], p[self.pivots == -1], color='r')
        ax.xaxis.set_major_formatter(tick.FuncFormatter(format_date))
        
        #annotate last index
        ax.annotate(index[-1].strftime("%Y-%m-%d %H:%M"),\
                    xy=(0.79, 0.02), ha='left', va='top', xycoords='axes fraction', fontsize=12)
                    
        ax2=ax.twinx()
        #same color to dashed and non-dashed
        ax2.set_prop_cycle(cycler('color', sorted(sns.color_palette("husl", 4))))
        #ax2.plot(np.arange(len(p)),Hurst,color='b', label='Hurst')
        #print len(p), len(SmoothADF)
        ax2.plot(np.arange(len(p)),SmoothADF,color='b', label='SmoothADF')
        ax2.plot(np.arange(len(p)),SmoothHurst,color='y', label='SmoothHurst')
        #ax2.plot(np.arange(len(p)),SmoothGarch, label='smoothGarch')
        #ax2.plot(np.arange(len(p)),scaledVolatility, label='scaledVolatility')
        #ax2.plot(np.arange(len(p)),scaledVolatility2, label='scaledVolatility2')
        #ax2.plot(np.arange(len(p)),scaledVolatility3, label='scaledVolatility3')
        #ax2.plot(np.arange(len(p)),scaledVolatility4, label='scaledVolatility4')
        #ax2.plot(np.arange(len(p)),adfpv, label='adf pvalue')
        t = np.zeros(len(p))
        t.fill(threshold)
        ax2.plot(np.arange(len(p)),t,color='k', label='threshold'+str(threshold))
        t2 = np.zeros(len(p))
        t2.fill(1-threshold)
        ax2.plot(np.arange(len(p)),t2,color='k', label='threshold'+str(1-threshold))
        ax2.set_ylim(-0.1,1.1)
        #ax2.plot(np.arange(len(p)),minmaxGarch,color='c', label='minmaxGarch')
        #ax2.plot(np.arange(len(p)),gar,color='b', label='Garch')
        handles, labels = ax2.get_legend_handles_labels()
        lgd2 = ax2.legend(handles, labels, loc='lower right',prop={'size':10})
        #ax2.scatter(np.arange(len(p))[mode2 == 0], p[mode2 == 0], color='b', label='hurstCycleMode')
        #ax2.scatter(np.arange(len(p))[mode2 == 1], p[mode2 == 1], color='y', label='hurstTrendMode')
    
        #ax2.plot(np.arange(nrows),dpsEquity, label='dps '+system, ls=next(linecycle))
        #ax2.xaxis.set_major_formatter(tick.FuncFormatter(format_date))
        if mode4[-1] ==0:
            title = ticker+' Mean Reverting: Cycle Mode '
        else:
            title = ticker+' Non-Mean Reverting: Trend Mode '
        ax2.set_title(title+str(bars)+' bars, threshold '+str(threshold))
        ax.set_xlim(0, nrows)        
        ax2.set_xlim(0, nrows)       
        fig.autofmt_xdate()
        if showPlot:
            plt.show()

        plt.close()
            
        return modes4
if __name__ == "__main__":
    asset='FUT'
    bars=25
    validationLength=50
    threshold=-1
    modeDict={}
    modeDict2={}
    dataPath = './nimbus/data/csidata/futures/'
    files = [ f for f in listdir(dataPath) if isfile(join(dataPath,f)) ]
    auxFutures = [x.split('_')[0] for x in files]
    for contract in auxFutures:    
        #if 'F_'+contract+'.txt' in files and (ticker[0:3] in contract or ticker[3:6] in contract):
        filename = contract+'_B.CSV'
        data = pd.read_csv(dataPath+filename, index_col=0, header=None)[-(bars*2+validationLength):]
        
        #data = data.drop([' P',' R', ' RINFO'],axis=1)
        #data = ratioAdjust(data)
        data.index = pd.to_datetime(data.index,format='%Y%m%d')
        data.columns = ['Open','High','Low','Close','Volume','OI','R','S']
        data.index.name = 'Dates'
        #contract = ''.join([i for i in contract if not i.isdigit()])
        if 'YT' not in contract:
            contract = ''.join([i for i in contract.split('_')[0] if not i.isdigit()])
        else:
            contract=contract.split('_')[0]
    p=data.Close
    p.index = p.index.to_datetime()
    mr=MeanReversion()
    modeDict[contract]=mr.adf(p,bars,threshold=-1, showPlot=True, ticker=contract)
    modeDict2[contract]=mr.adf2(p,bars,threshold=0.8, showPlot=True, ticker=contract)