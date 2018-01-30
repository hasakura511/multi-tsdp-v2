#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 15:20:18 2018

zig zag with modifications

@author: hidemiasakura
"""

import numpy as np
import pandas as pd
#import talib as ta
#import arch
#import random
import time
#from os import listdir
#from os.path import isfile, join
#import statsmodels.tsa.stattools as ts
#import datetime
from datetime import datetime as dt
#from pytz import timezone
#from numpy import cumsum, log, polyfit, sqrt, std, subtract
#from numpy.random import randn
#from statsmodels.sandbox.stats.runs import runstest_1samp
#from scipy import stats
#from sklearn.externals import joblib
#import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
from cycler import cycler
import seaborn as sns
from itertools import cycle
from matplotlib.finance import candlestick_ohlc
#from matplotlib.finance import volume_overlay3
#from matplotlib.dates import num2date
from matplotlib.dates import date2num
from matplotlib.dates import DateFormatter, WeekdayLocator,\
                                            DayLocator, MONDAY, HourLocator
#import matplotlib.mlab as mlab
import collections
#import warnings
#warnings.simplefilter('error')

#mpl.rcParams["axes.formatter.useoffset"] = False
class zigzag(object):
    '''
    	all list parameters are expected to be an one dimensional
    	list of nominal prices, e.g. [1,1.2,.3,10,.3,25]
    '''
    def __init__(self, sym, prices, zigzag_stdev=2, MIN_DATAPOINTS=3):
        self.symbol=sym
        #self.prices = prices.Close
        self.prices=(prices.Close.pct_change().fillna(0)+1).cumprod()
        #self.SR_LOOKBACK=SR_LOOKBACK
        self.MIN_DATAPOINTS=MIN_DATAPOINTS
        self.prices_std= prices.Close.pct_change().std()
        self.candlesticks=zip(date2num(prices.index.to_pydatetime()),prices['Open'],prices['High'],prices['Low'],prices['Close'],prices['Volume'])
        self.zigzag_stdev=zigzag_stdev
        self.up_thresh = self.prices_std*self.zigzag_stdev
        self.down_thresh = -self.prices_std*self.zigzag_stdev
        self.pivots = self.peak_valley_pivots()
        self.initial_pivot = self._identify_initial_pivot()
        #self.eCurves = {}
      
    def peak_valley_pivots(self):
        """
        Finds the peaks and valleys of a series.
        """
        #print 'dt', self.down_thresh
        if self.down_thresh > 0:
            raise ValueError('The down_thresh must be negative.')
    
        initial_pivot = self._identify_initial_pivot()
    
        t_n = len(self.prices)
        pivots = np.zeros(t_n, dtype='i1')
        pivots[0] = initial_pivot
    
        # Adding one to the relative change thresholds saves operations. Instead
        # of computing relative change at each point as x_j / x_i - 1, it is
        # computed as x_j / x_1. Then, this value is compared to the threshold + 1.
        # This saves (t_n - 1) subtractions.
        up_thresh = 1 +self.up_thresh
        down_thresh = 1 + self.down_thresh
    
        trend = -initial_pivot
        last_pivot_t = 0
        last_pivot_x = self.prices[0]
        for t in range(1, len(self.prices)):
            x = self.prices[t]
            r = x / last_pivot_x
    
            if trend == -1:
                if r >= up_thresh:
                    pivots[last_pivot_t] = trend
                    trend = 1
                    last_pivot_x = x
                    last_pivot_t = t
                elif x < last_pivot_x:
                    last_pivot_x = x
                    last_pivot_t = t
            else:
                if r <= down_thresh:
                    pivots[last_pivot_t] = trend
                    trend = -1
                    last_pivot_x = x
                    last_pivot_t = t
                elif x > last_pivot_x:
                    last_pivot_x = x
                    last_pivot_t = t
    
        if last_pivot_t == t_n-1:
            pivots[last_pivot_t] = trend
        elif pivots[t_n-1] == 0:
            pivots[t_n-1] = trend
    
        return pivots
        
    def _identify_initial_pivot(self):
        """Quickly identify the X[0] as a peak or valley."""
        PEAK, VALLEY = 1, -1
        x_0 = self.prices[0]
        max_x = x_0
        max_t = 0
        min_x = x_0
        min_t = 0
        up_thresh = 1 +self.up_thresh
        down_thresh = 1 + self.down_thresh
    
        for t in range(1, len(self.prices)):
            x_t = self.prices[t]
    
            if x_t / min_x >= up_thresh:
                return VALLEY if min_t == 0 else PEAK
    
            if x_t / max_x <= down_thresh:
                return PEAK if max_t == 0 else VALLEY
    
            if x_t > max_x:
                max_x = x_t
                max_t = t
    
            if x_t < min_x:
                min_x = x_t
                min_t = t
    
        t_n = len(self.prices)-1
        return VALLEY if x_0 < self.prices[t_n] else PEAK
        
    def compute_segment_returns(self):
        """Return a numpy array of the pivot-to-pivot returns for each segment."""
        pivot_points = np.array([self.prices[i] for i,x in enumerate(~np.equal(self.pivots,0)) if x == True])
        return pivot_points[1:] / pivot_points[:-1] - 1.0
    
    def pivots_to_modes(self):
        """
        Translate pivots into trend modes.
        Parameters
        ----------
        pivots : the result of calling peak_valley_pivots
        Returns
        -------
        A numpy array of trend modes. That is, between (VALLEY, PEAK] it is 1 and
        between (PEAK, VALLEY] it is -1.
        """
        modes = np.zeros(len(self.pivots), dtype='i1')
        modes[0] = self.pivots[0]
        mode = -modes[0]
        for t in range(1, len(self.pivots)):
            x = self.pivots[t]
            if x != 0:
                modes[t] = mode
                mode = -x
            else:
                modes[t] = mode
        self.modes_no_zeros=modes
        return self.modes_no_zeros
    
    def pivots_to_modes_with_big_swings(self, threshold_std):
        """
        Translate pivots into trend modes.
        Parameters
        ----------
        pivots : the result of calling peak_valley_pivots
        Returns
        -------
        A numpy array of trend modes. That is, between (VALLEY, PEAK] it is 1 and
        between (PEAK, VALLEY] it is -1.
        """
        threshold = self.prices_std*threshold_std*100
        modes = np.zeros(len(self.pivots), dtype='i1')
        #self.pivots_with_zeros = self.pivots.copy()
        
        returns=self.compute_segment_returns()*100
        pivot_index = np.arange(len(self.prices))[self.pivots != 0]
        pivot_index =np.delete(pivot_index,0)
        #print(pivot_index)
        #modes[0] = self.pivots[0]
        #mode = -modes[0]
        i=-1
        for t in range(len(self.pivots)-1,-1,-1):
            
            if t in pivot_index:
                
                ret = returns[i]
                i-=1
                if abs(ret)>threshold:
                    if ret >0:
                        mode=1
                    else:
                        mode=-1
                else:
                    mode=0
            #print(t, t in pivot_index, i, mode, ret)
            modes[t]=mode
            
        
        self.modes_with_big_swings=modes
        #self.pivots_with_zeros=self.pivots[self.modes_with_big_swings ==0]=0
        '''
        self.pivots_with_zeros[0]=self.modes_with_big_swings[0]
        for i in range(1, len(self.pivots)):
            if self.modes_with_big_swings[i] != self.modes_with_big_swings[i-1]:
                print(i, self.modes_with_big_swings[i],self.modes_with_big_swings[i-1])
                self.pivots_with_zeros[i]=self.modes_with_big_swings[i]
            else:
                self.pivots_with_zeros[i]=0
            
        '''
        return self.modes_with_big_swings
    
    def modes_with_big_swings_to_modes_with_small_swings(self):
        """
        Translate modes with big swing into small swings.
        Parameters
        ----------
        """
        modes = np.zeros(len(self.pivots), dtype='i1')
        
        for i in range(len(self.pivots)):
            if self.modes_with_big_swings[i]==0:
                modes[i]=self.modes_no_zeros[0]
            else:
                modes[i]=0
        
        self.modes_with_small_swings=modes
        return self.modes_with_small_swings
        
    def max_drawdown(self):
        """
        Return the absolute value of the maximum drawdown of sequence X.
        Note
        ----
        If the sequence is strictly increasing, 0 is returned.
        """
        mdd = 0
        peak = self.prices[0]
        for x in self.prices:
            if x > peak: 
                peak = x
            dd = (peak - x) / peak
            if dd > mdd:
                mdd = dd
        return mdd
      
        
    def get_minor_major(self):
        #gets minor peaks and valleys
        majorPeak = majorValley = minorPeak = minorValley = None
        zz_std=self.zigzag_stdev
        data = self.prices.reset_index()
        LAST_INDEX=data.index[-1]
        peaks=[]
        valleys=[]
        #decrease stdev until there are four peaks and valleys that meet the criteria
        i=0
        while majorPeak ==None or majorValley == None or minorPeak ==None\
                        or minorValley == None:
        #while len(peaks)<4 and len(valleys)<4:
            self.up_thresh = self.prices_std*zz_std
            self.down_thresh = -self.prices_std*zz_std
            #print(self.up_thresh, self.down_thresh)
            self.pivots = self.peak_valley_pivots()
            self.initial_pivot = self._identify_initial_pivot()
            self.peaks_valleys =  self.peak_valley_pivots()
            
            #list of peaks and valleys
            peaks=data.loc[self.peaks_valleys==1].index.tolist()
            peaks=[x for x in peaks if LAST_INDEX-x>self.MIN_DATAPOINTS]
            valleys=data.loc[self.peaks_valleys==-1].index.tolist()
            valleys=[x for x in valleys if LAST_INDEX-x>self.MIN_DATAPOINTS]
            print(i, zz_std, 'up, down', self.up_thresh, self.down_thresh)
            #major peaks and valleys
            peaks_sorted_index=data.Close.iloc[peaks].sort_values().index.tolist()
            print('peaks',peaks_sorted_index)
            if len(peaks_sorted_index)>0:
                majorPeak=peaks_sorted_index[-1]
                
                
            valleys_sorted_index=data.Close.iloc[valleys]\
                            .sort_values(ascending=False).index.tolist()
            print('valleys',valleys_sorted_index)
            if len(valleys_sorted_index)>0:
                majorValley=valleys_sorted_index[-1]
            
            #find the next closest peak/valley
            minor_peaks=[peak for peak in peaks_sorted_index\
                             if abs(majorPeak-peak) > self.MIN_DATAPOINTS]
            if len(minor_peaks)>1:
                    idx = (np.abs(np.array(minor_peaks)-majorPeak)).argmin()
                    #print('minor peak idx', idx)
                    minorPeak = minor_peaks[idx]
                    
            minor_valleys=[valley for valley in valleys_sorted_index\
                           if abs(majorValley-valley) > self.MIN_DATAPOINTS]
            if len(minor_valleys)>1:
                    idx = (np.abs(np.array(minor_valleys)-majorValley)).argmin()
                    #print('minor peak idx', idx)
                    minorValley = minor_valleys[idx]                   
                    
            #decrement the standard deviation to create more pivot points
            zz_std*=.9
            i+=1
            
        pv_sorted = np.asarray(sorted(peaks_sorted_index+valleys_sorted_index))
        print('pv_sorted', pv_sorted)
        shortStart=pv_sorted[-1]
        #self.majorPeak=(data.loc[majorPeak].Dates, data.loc[majorPeak].Close)
        #self.majorValley=(data.loc[majorValley].Dates, data.loc[majorValley].Close)
        #self.minorPeak=(data.loc[minorPeak].Dates, data.loc[minorPeak].Close)
        #self.minorValley=(data.loc[minorValley].Dates, data.loc[minorValley].Close)
        
        self.shortStart=(shortStart, data.loc[shortStart].Close)
        self.majorPeak=(majorPeak, data.loc[majorPeak].Close)
        self.majorValley=(majorValley, data.loc[majorValley].Close)
        self.minorPeak=(minorPeak, data.loc[minorPeak].Close)
        self.minorValley=(minorValley, data.loc[minorValley].Close)
        #self.peaksSorted = peaksSorted
        #self.valleysSorted = valleysSorted
        self.pv_sorted=pv_sorted
        self.plot_pivots()               
                      
        
    def get_peaks_valleys(self, show_plots=False, verbose=False, **kwargs):
        #prioritises recent data
        self.return_threshold_stdev=kwargs.get('return_threshold_stdev',4)
        #min_peaks=kwargs.get('min_peaks',4)
        min_peaks_valleys=kwargs.get('min_peaks_valleys',4)
        self.trend = None
        self.majorPeak = self.majorValley = self.minorPeak = self.minorValley = None
        lastPeak = prevPeak = lastValley = prevValley = None
        zz_std=self.zigzag_stdev
        data = self.prices.reset_index()
        LAST_INDEX=data.index[-1]
        peaks=[]
        valleys=[]
        #decrease stdev until there are four peaks and valleys that meet the criteria
        i=0
        #while majorPeak ==None or majorValley == None or minorPeak ==None\
        #                or minorValley == None:
        while len(peaks)<min_peaks_valleys and len(valleys)<min_peaks_valleys:
            self.up_thresh = self.prices_std*zz_std
            self.down_thresh = -self.prices_std*zz_std
            #print(self.up_thresh, self.down_thresh)
            self.pivots = self.peak_valley_pivots()
            self.initial_pivot = self._identify_initial_pivot()
            self.peaks_valleys =  self.peak_valley_pivots()
            
            #list of peaks and valleys
            peaks=data.loc[self.peaks_valleys==1].index.tolist()
            peaks=[x for x in peaks if LAST_INDEX-x>self.MIN_DATAPOINTS]
            valleys=data.loc[self.peaks_valleys==-1].index.tolist()
            valleys=[x for x in valleys if LAST_INDEX-x>self.MIN_DATAPOINTS]
            #print(i, zz_std, 'up, down', self.up_thresh, self.down_thresh)
            
            #decrement the standard deviation to create more pivot points
            zz_std*=.9
            i+=1

        
        peaks_sorted_index=data.Close.iloc[peaks].sort_values().index.tolist()
        
        if len(peaks_sorted_index)>0:
            majorPeak=peaks_sorted_index[-1]
            
            
        valleys_sorted_index=data.Close.iloc[valleys]\
                        .sort_values(ascending=False).index.tolist()
        
        if len(valleys_sorted_index)>0:
            majorValley=valleys_sorted_index[-1]
                    
        lastPeak=peaks[-1]
        prevPeak=peaks[-2]
        lastValley=valleys[-1]
        prevValley=valleys[-2]
        
        
        #first_datapoint=modes[0]
        #last_datapoint=modes[-1]
        
        if data.loc[lastValley].Close > data.loc[prevValley].Close:
            valley_higher_high=True
        else:
            valley_higher_high=False
            
        if data.loc[lastPeak].Close > data.loc[prevPeak].Close:
            peak_higher_high=True
        else:
            peak_higher_high=False
        
        if valley_higher_high and peak_higher_high:
            trend = 1
        else:
            trend = -1
        #if the recent peaks valleys contained within major peaks/valleys
        #it's mean reverting
        if data.loc[majorPeak].Close >= data.loc[lastPeak].Close\
                and data.loc[majorPeak].Close >= data.loc[prevPeak].Close\
                and data.loc[majorValley].Close <= data.loc[lastValley].Close\
                and data.loc[majorValley].Close <= data.loc[prevValley].Close\
                and data.iloc[-1].Close < data.loc[majorPeak].Close\
                and data.iloc[-1].Close > data.loc[majorValley].Close:
            trend=0
        
        #making new highs/lows
        if data.iloc[-1].Close>data.loc[majorPeak].Close:
            trend=1
        if data.iloc[-1].Close<data.loc[majorValley].Close:
            trend=-1        
        
        #special cases
        #lower lows, lower highs with recent major valley =last valley
        #and last price hasn't broken previous resistance
        if data.iloc[-1].Close<data.loc[lastPeak].Close\
                and data.loc[lastPeak].Close < data.loc[prevPeak].Close\
                and data.loc[prevPeak].Close < data.loc[majorPeak].Close\
                and data.loc[lastValley].Close < data.loc[prevValley].Close\
                and data.loc[lastValley].Close == data.loc[majorValley].Close\
                and data.iloc[-1].Close < data.loc[prevValley].Close:
            trend=-1
            
        #higher highs, higher lows with recent major peak =last preak
        #and last price hasn't broken previous support
        if data.iloc[-1].Close<data.loc[lastPeak].Close\
                and data.loc[lastPeak].Close > data.loc[prevPeak].Close\
                and data.loc[prevValley].Close > data.loc[majorValley].Close\
                and data.loc[lastValley].Close > data.loc[prevValley].Close\
                and data.loc[lastPeak].Close == data.loc[majorPeak].Close\
                and data.iloc[-1].Close > data.loc[prevPeak].Close:
            trend=1
        self.trend = trend
        
        pv_sorted = np.asarray(sorted(peaks+valleys))
        #print('pv_sorted', pv_sorted)
        shortStart=pv_sorted[-1]
        self.majorPeak=(majorPeak, data.loc[majorPeak].Close)
        self.majorValley=(majorValley, data.loc[majorValley].Close)
        #self.minorPeak=(data.loc[minorPeak].Dates, data.loc[minorPeak].Close)
        #self.minorValley=(data.loc[minorValley].Dates, data.loc[minorValley].Close)
        
        self.shortStart=(shortStart, data.loc[shortStart].Close)
        self.lastPeak=(lastPeak, data.loc[lastPeak].Close)
        self.prevPeak=(prevPeak, data.loc[prevPeak].Close)
        self.lastValley=(lastValley, data.loc[lastValley].Close)
        self.prevValley=(prevValley, data.loc[prevValley].Close)
        #self.peaksSorted = peaksSorted
        #self.valleysSorted = valleysSorted
        self.pv_sorted=pv_sorted
        
        #creates self.modes_no_zeros and self.modes_with_big_swings
        self.pivots_to_modes()
        self.pivots_to_modes_with_big_swings(self.return_threshold_stdev)
        self.modes_with_big_swings_to_modes_with_small_swings()
        self.modes_with_big_count=pd.Series(collections.Counter(self.modes_with_big_swings))
        self.modes_with_big_count_pct=(self.modes_with_big_count/self.modes_with_big_count.sum())
        self.modes_with_small_count=pd.Series(collections.Counter(self.modes_with_small_swings))
        self.modes_with_small_count_pct=(self.modes_with_small_count/self.modes_with_small_count.sum())
        
        if show_plots:
            self.plot_pivots()
        
    
        if verbose:
            print('i', i, 'std', zz_std, 'up', self.up_thresh, 'down', self.down_thresh)
            print('min_peaks_valleys',min_peaks_valleys)
            print(len(peaks_sorted_index), 'peaks',peaks_sorted_index)
            print(len(valleys_sorted_index), 'valleys',valleys_sorted_index)
            print(self.symbol, 'trend', self.trend,
                  'modes_no_zeros', self.modes_no_zeros[-1],
                  'modes_with_big_swings', self.modes_with_big_swings[-1],
                  'modes_with_small_swings', self.modes_with_big_swings[-1]
                  )
            print('big swings', self.modes_with_big_count.to_dict())
            print('small swings', self.modes_with_small_count.to_dict())
        return self.trend, self.modes_no_zeros[-1]
    
    def signals(self):
        if self.trend == None:
            print('get_peaks_valleys() first')
            return
        
        self.signals=pd.Series(self.pivots_to_modes(),
                               index=self.prices.index).shift(-1).fillna(0)
        self.signals.name='signals_ZIGZAG_'+self.symbol
        return self.signals.astype(int)
        
        
        
             

        
    def plot_hist(self):
        hist, bins = np.histogram(self.compute_segment_returns(), bins=50)
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        plt.bar(center, hist, align='center', width=width)
        plt.show()
        
    def plot_pivots(self, **kwargs):
        l=kwargs.get('l',6)
        w=kwargs.get('w',6)
        startPeak=kwargs.get('startPeak',None)
        startValley=kwargs.get('startValley',None)
        majorPeak=self.majorPeak
        majorValley=self.majorValley
        minorPeak=self.minorPeak
        minorValley=self.minorValley
        lastPeak=self.lastPeak
        prevPeak=self.prevPeak
        lastValley=self.lastValley
        prevValley=self.prevValley
        shortStart=self.shortStart
        #cycleList = kwargs.get('cycleList',None)
        #indicators = kwargs.get('indicators',None)
        #signals = kwargs.get('signals',None)
        #chartTitle = kwargs.get('chartTitle',1)
        #mode = kwargs.get('mode',None)
        #savePath = kwargs.get('savePath',None)
        #debug = kwargs.get('debug',True)
        
        def format_date(x, pos=None):
            thisind = np.clip(int(x + 0.5), 0, self.prices.shape[0] - 1)
            return self.prices.index[thisind].strftime("%Y-%m-%d %H:%M")
            
        fig = plt.figure(figsize=(w,l))
        ax = fig.add_subplot(111, xlim=(0, len(self.prices)), ylim=(self.prices.min()*0.99, self.prices.max()*1.01))
        #ax.plot(np.arange(len(self.prices)), self.prices, 'k:', alpha=0.5)
        #ax.plot(np.arange(len(self.prices))[self.pivots != 0], self.prices[self.pivots != 0], 'k-')
        #ax.scatter(np.arange(len(self.prices))[self.pivots == 1], self.prices[self.pivots == 1], color='g')
        #ax.scatter(np.arange(len(self.prices))[self.pivots == -1], self.prices[self.pivots == -1], color='r')
        
        ax.plot(np.arange(len(self.prices)), self.prices, 'k:', alpha=0.5)
        
        ax.scatter(np.arange(len(self.prices))[self.modes_with_big_swings == 1], self.prices[self.modes_with_big_swings == 1], color='g')
        ax.scatter(np.arange(len(self.prices))[self.modes_with_big_swings == -1], self.prices[self.modes_with_big_swings == -1], color='r')
        ax.plot(np.arange(len(self.prices))[self.pivots != 0], self.prices[self.pivots != 0], 'k-')
        #ax.scatter(np.arange(len(self.prices))[self.pivots == 1], self.prices[self.pivots == 1], color='g')
        #ax.scatter(np.arange(len(self.prices))[self.pivots == -1], self.prices[self.pivots == -1], color='r')
        ax.xaxis.set_major_formatter(tick.FuncFormatter(format_date))
        
        returns=self.compute_segment_returns()*100
        pivots = np.arange(len(self.prices))[self.pivots != 0]
        pivots =np.delete(pivots,0)
        segments=[(round(returns[i],2), x, self.prices[x]) for i,x in enumerate(pivots)]
        for r,x,y in segments:
            if r<0:
                ax.annotate(str(r)+'%', (x,y), color='red',
                xytext=(0, 0), textcoords='offset points')
            else:
                ax.annotate(str(r)+'%', (x,y), color='green',
                xytext=(0, 0), textcoords='offset points')
        
        #if mode is not None:
        #    ax.scatter(np.arange(len(self.prices))[mode.values == 0], self.prices[mode.values == 0], color='g', label='0 CycleMode')
        #    ax.scatter(np.arange(len(self.prices))[mode.values == 1], self.prices[mode.values == 1], color='r', label='1 TrendMode')
        #annotate last index
        ax.annotate(self.prices.index[-1].strftime("%Y-%m-%d"),\
                    xy=(0.79, 0.02), ha='left', va='top', xycoords='axes fraction', fontsize=12)
        
        fig.autofmt_xdate()
        if startPeak is not None and startValley is not None:
            ax.annotate('peak start', startPeak,
            xytext=(0, 20), textcoords='offset points',
            arrowprops=dict(facecolor='green', shrink=0.05),
            )
            ax.annotate('valley start', startValley,
            xytext=(0, -20), textcoords='offset points',
            arrowprops=dict(facecolor='red', shrink=0.05),
            )
        if minorPeak is not None and minorValley is not None:
            ax.annotate('minor peak', minorPeak,
            xytext=(0, 20), textcoords='offset points',
            arrowprops=dict(facecolor='green', shrink=0.05),
            )
            ax.annotate('minor valley', minorValley,
            xytext=(0, -20), textcoords='offset points',
            arrowprops=dict(facecolor='red', shrink=0.05),
            )
        if shortStart is not None:
            ax.annotate(str(shortStart[0])+' short start', shortStart,
            xytext=(-70, 0), textcoords='offset points',
            arrowprops=dict(facecolor='blue', shrink=0.05),
            )
        '''
        if majorPeak is not None and majorValley is not None:
            ax.annotate('major peak', majorPeak,
            xytext=(0, 20), textcoords='offset points',
            arrowprops=dict(facecolor='green', shrink=0.05),
            )
            ax.annotate('major valley', majorValley,
            xytext=(0, -20), textcoords='offset points',
            arrowprops=dict(facecolor='red', shrink=0.05),
            )
        '''
        last_peak_text=''
        prev_peak_text=''
        if majorPeak is not None:
            if majorPeak==lastPeak:
                last_peak_text=' major,'
            elif majorPeak==prevPeak:
                prev_peak_text=' major,'
            else:
                ax.annotate('major peak', majorPeak,
                xytext=(0, 20), textcoords='offset points',
                arrowprops=dict(facecolor='green', shrink=0.05),
                )
        last_valley_text=''
        prev_valley_text=''
        if majorValley is not None:
            if majorValley==lastValley:
                last_valley_text=' major,'
            elif majorValley==prevValley:
                prev_valley_text=' major,'
            else:
                ax.annotate('major valley', majorValley,
                xytext=(0, -20), textcoords='offset points',
                arrowprops=dict(facecolor='red', shrink=0.05),
                )
        if lastValley is not None and lastPeak is not None:
            ax.annotate(str(lastPeak[0])+last_peak_text+' last peak', lastPeak,
            xytext=(0, 20), textcoords='offset points',
            arrowprops=dict(facecolor='green', shrink=0.05),
            )
            ax.annotate(str(lastValley[0])+last_valley_text+' last valley', lastValley,
            xytext=(0, -20), textcoords='offset points',
            arrowprops=dict(facecolor='red', shrink=0.05),
            )
        if prevPeak is not None and prevValley is not None:
            ax.annotate(str(prevPeak[0])+prev_peak_text+' prev peak', prevPeak,
            xytext=(0, 20), textcoords='offset points',
            arrowprops=dict(facecolor='green', shrink=0.05),
            )
            ax.annotate(str(prevValley[0])+prev_valley_text+' prev valley', prevValley,
            xytext=(0, -20), textcoords='offset points',
            arrowprops=dict(facecolor='red', shrink=0.05),
            )
        if self.trend == 0:
            chart_title = 'Mean Reverting'
        elif self.trend == -1:
            chart_title = 'Down Trend'
        else:
            chart_title = 'Up Trend'
            
        chart_title='{} {} {} std {}% threshold \n{}'.format(self.symbol,
                     chart_title, self.return_threshold_stdev, 
                     round(self.return_threshold_stdev*self.prices_std*100,2),
                     str(self.modes_with_big_count_pct.round(2).to_dict()))
        
        ax.set_title(chart_title)
        
        ax.set_xlim(0, len(self.prices))
        plt.show()
        
    def plot_pivots2(self, **kwargs):
        #used in the old signal system.
        l=kwargs.get('l',6)
        w=kwargs.get('w',6)
        startPeak=kwargs.get('startPeak',None)
        startValley=kwargs.get('startValley',None)
        majorPeak=kwargs.get('majorPeak',None)
        majorValley=kwargs.get('majorValley',None)
        minorPeak=kwargs.get('minorPeak',None)
        minorValley=kwargs.get('minorValley',None)
        shortStart=kwargs.get('shortStart',None)
        cycleList = kwargs.get('cycleList',None)
        indicators = kwargs.get('indicators',None)
        signals = kwargs.get('signals',None)
        chartTitle = kwargs.get('chartTitle',1)
        mode = kwargs.get('mode',None)
        savePath = kwargs.get('savePath',None)
        debug = kwargs.get('debug',True)
        barsize=kwargs.get('barsize',None)
        
        fig = plt.figure(figsize=(w,l*2))
        #ax = fig.add_subplot(111)
        ax=plt.subplot2grid((2,1), (0,0), rowspan=1, colspan=1)
        plt.title(chartTitle)
        
        if barsize==None:
            width=0.6
            major = WeekdayLocator(MONDAY)        # major ticks on the mondays
            minor = DayLocator()              # minor ticks on the days
            majorFormat = DateFormatter('%b %d %Y')  # e.g., Jan 12
            minorFormat = DateFormatter('%d')      # e.g., 12
        else:
            #major = WeekdayLocator(MONDAY)        # major ticks on the mondays
            width=0.1
            minor = HourLocator(byhour=range(4,24,4))
            major = DayLocator()              # minor ticks on the days
            majorFormat = DateFormatter('%b %d %Y')  # e.g., Jan 12
            if len(self.prices)<30:
                minorFormat = DateFormatter('%H:%M')      
                ax.xaxis.set_minor_formatter(minorFormat)
                
            

        #plt.ylabel(ticker)
        #fig, ax = plt.subplots()
        fig.subplots_adjust(bottom=0.2)
        ax.xaxis.set_major_locator(major)
        ax.xaxis.set_minor_locator(minor)
        ax.xaxis.set_major_formatter(majorFormat)
        dates = [x[0] for x in self.candlesticks]
        dates = np.asarray(dates)
        
        #volume
        volume = [x[5] for x in self.candlesticks]
        volume = np.asarray(volume)
        ax.fill_between(dates,0, volume, facecolor='#0079a3', alpha=0.4)
        #scale the x-axis tight
        #ax.set_xlim(min(dates),max(dates))
        # the y-ticks for the bar were too dense, keep only every third one
        yticks = ax.get_yticks()
        ax.set_yticks(yticks[::2])
        ax.yaxis.set_label_position("left")
        ax.set_ylabel('Volume', size=12)
        ax.grid(True)
        plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
        plt.setp(ax.xaxis.get_minorticklabels(), rotation=45, horizontalalignment='right')
        #price candles
        ax2p=ax.twinx()
        candlestick_ohlc(ax2p, self.candlesticks, width=width, colorup='g')
        
        ax2p.xaxis_date()
        ax2p.autoscale_view()

        sma=pd.rolling_mean(self.prices,5)
        bbu=sma+pd.rolling_std(self.prices,5)
        bbl=sma-pd.rolling_std(self.prices,5)
        runs=pd.DataFrame(np.where(self.prices<sma,-1,1),columns=['col'])
        runs['block'] = (runs['col'] != runs['col'].shift(1)).astype(int).cumsum()
        runs['count'] = runs.groupby('block').transform(lambda x: range(1, len(x) + 1))
        
        #plt.rc('lines', linewidth=1)
        ax2p.plot(dates,sma,'k:',label='SMA5', linewidth=1.5)
        ax2p.plot(dates,bbu,'k-.',label='BBU', linewidth=1.5)
        ax2p.plot(dates,bbl,'k-.',label='BBL', linewidth=1.5)
        handles, labels = ax2p.get_legend_handles_labels()
        lgd = ax2p.legend(handles, labels, loc='best',prop={'size':10})
        ax2p.yaxis.set_label_position("right")
        ax2p.set_ylabel('Price', size=12)
        #print runs['count']
        for i,count in enumerate(runs['count']):
            if runs['col'][i]<0:
                xytext=(-3,-20)
            else:
                xytext=(-3,20)
                
            ax2p.annotate(str(count), (dates[i], self.prices[i]),
                         xytext=xytext, textcoords='offset points',
                         size='medium')
                
        #indicators
        ax3 = plt.subplot2grid((2,1), (1,0), rowspan=1, colspan=1)

        def format_date(x, pos=None):
            thisind = np.clip(int(x + 0.5), 0, self.prices.shape[0] - 1)
            return self.prices.index[thisind].strftime("%Y-%m-%d %H:%M")

        #fig = plt.figure(figsize=(w,l))
        #ax3 = fig.add_subplot(111, xlim=(0, len(self.prices)), ylim=(self.prices.min()*0.99, self.prices.max3()*1.01))
        
        ax3.plot(np.arange(len(self.prices)), self.prices, 'k:', alpha=0.5)
        ax3.plot(np.arange(len(self.prices))[self.pivots != 0], self.prices[self.pivots != 0], 'k-')
        ax3.scatter(np.arange(len(self.prices))[self.pivots == 1], self.prices[self.pivots == 1], color='g')
        ax3.scatter(np.arange(len(self.prices))[self.pivots == -1], self.prices[self.pivots == -1], color='r')

        
        
        #ax3.scatter(np.arange(len(self.prices))[self.pivots == 1], self.prices[self.pivots == 1], color='g')
        #ax3.scatter(np.arange(len(self.prices))[self.pivots == -1], self.prices[self.pivots == -1], color='r')
        ax3.xaxis.set_major_formatter(tick.FuncFormatter(format_date))
        
        if mode is not None:
            ax3.scatter(np.arange(len(self.prices))[mode.values == 0], self.prices[mode.values == 0], color='g', label='0 CycleMode')
            ax3.scatter(np.arange(len(self.prices))[mode.values == 1], self.prices[mode.values == 1], color='r', label='1 TrendMode')
        #annotate last index
        ax3.annotate(self.prices.index[-1].strftime("%Y-%m-%d %H:%M"),\
                    xy=(0.79, 0.02), ha='left', va='top', xycoords='axes fraction', fontsize=12)
        
        #fig.autofmt_xdate()
        plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
        if startPeak is not None and startValley is not None:
            ax3.annotate('peak start', startPeak,
            xytext=(0, 20), textcoords='offset points',
            arrowprops=dict(facecolor='green', shrink=0.05),
            )
            ax3.annotate('valley start', startValley,
            xytext=(0, -20), textcoords='offset points',
            arrowprops=dict(facecolor='red', shrink=0.05),
            )
        if majorPeak is not None and majorValley is not None:
            ax3.annotate('major peak', majorPeak,
            xytext=(0, 20), textcoords='offset points',
            arrowprops=dict(facecolor='green', shrink=0.05),
            )
            ax3.annotate('major valley', majorValley,
            xytext=(0, -20), textcoords='offset points',
            arrowprops=dict(facecolor='red', shrink=0.05),
            )
        if minorPeak is not None and minorValley is not None:
            ax3.annotate('minor peak', minorPeak,
            xytext=(0, 20), textcoords='offset points',
            arrowprops=dict(facecolor='green', shrink=0.05),
            )
            ax3.annotate('minor valley', minorValley,
            xytext=(0, -20), textcoords='offset points',
            arrowprops=dict(facecolor='red', shrink=0.05),
            )
        if shortStart is not None:
            ax3.annotate('short start', shortStart,
            xytext=(-70, 0), textcoords='offset points',
            arrowprops=dict(facecolor='blue', shrink=0.05),
            )            
        if cycleList is not None:
            for l in cycleList:
                 ax3.annotate(str(l[0]), l[1],
                 xytext=(5, -5), textcoords='offset points',
                 size='medium')
                 
        if indicators is not None:
            #ax3.plot(np.arange(len(self.prices)), self.prices, 'ko', alpha=0.5)
            plt.rc('lines', linewidth=1)
            plt.rc('axes', prop_cycle=(cycler('color', ['r', 'b', 'g', 'm','y','c']) +\
                                       cycler('linestyle', ['-', '-', '-', '-.','--',':'])))
            ax4=ax3.twinx()
            ax4.xaxis.set_major_formatter(tick.FuncFormatter(format_date))
            
            for i,ind in enumerate(indicators):
                ax4.plot(np.arange(len(indicators)),indicators[ind], label=ind)
            handles, labels = ax4.get_legend_handles_labels()
            lgd2 = ax4.legend(handles, labels, loc='best',prop={'size':10})
            ax4.set_xlim(0, len(indicators))
            ax4.set_title(chartTitle)
            
        if signals is not None:
            plt.rc('lines', linewidth=2)
            #sns.palplot(sns.hls_palette(len(signals.columns), l=.3, s=.8))
            #plt.style.use('ggplot')
            linecycle = cycle(['-', '--'])
            #plt.rc('axes', prop_cycle=(cycler('color',\
            #            [plt.cm.cool(i) for i in np.linspace(0, 1, len(signals.columns))])))
            nrows = len(self.prices)
            ax4=ax3.twinx()
            #same color to dashed and non-dashed
            ax4.set_prop_cycle(cycler('color',sorted(sns.color_palette("husl", len(signals))*2)))
            ax4.xaxis.set_major_formatter(tick.FuncFormatter(format_date))
            ax4.set_title(chartTitle)
            #ga_pct = self.prices.pct_change().shift(-1).fillna(0)
            #ga_pct.name = 'gainAhead'

            for system in signals:
                nodpsEquity = signals[system][-nrows:].netEquity
                dpsEquity = signals[system][-nrows:].dpsNetEquity
                nodpsComm=round(signals[system][-nrows:].nodpsComm.sum(),0)
                dpsComm=round(signals[system][-nrows:].dpsCommission.sum(),0)
                signal=signals[system].signals[-1]
                nodpsSafef=signals[system].nodpsSafef[-1]
                dpsSafef=signals[system].dpsSafef[-1]
                #print system, chartTitle, nodpsEquity, dpsEquity
                ax4.plot(np.arange(nrows),nodpsEquity, label=system+' noDpsComm: '\
                                +str(nodpsComm)+' Sig: '+str(signal)+' Safef: '+str(nodpsSafef),\
                                ls=next(linecycle))
                ax4.plot(np.arange(nrows),dpsEquity, label='dps '+system+ ' dpsComm: '\
                            +str(dpsComm)+' Sig: '+str(signal)+' Safef: '+str(dpsSafef),\
                            ls=next(linecycle))

                    
            handles, labels = ax4.get_legend_handles_labels()
            lgd2 = ax4.legend(handles, labels, loc='lower left',prop={'size':10})
            ax4.set_xlim(0, nrows)
            #ax4.get_xaxis().get_major_formatter().set_useOffset(False)
            #ax4.get_xaxis().get_major_formatter().set_scientific(False)
                
        ax3.set_xlim(0, len(self.prices))
        
        if debug:
            plt.show()
        #if savePath != None:
        #    print 'Saving '+savePath+'.png'
        #    fig.savefig(savePath+'.png', bbox_inches='tight')
            
        plt.close()

if __name__ == '__main__':
    from gsm import Service
    from nimbus.csidata import Futures
    
    '''
    if 'myVar' in locals():
      # myVar exists.
    To check the existence of a global variable:
    
    if 'myVar' in globals():
      # myVar exists.
    To check if an object has an attribute:
    
    if hasattr(obj, 'attr_name'):
      # obj.attr_name exists.
    '''
    if 'start_time' not in globals():
        start_time = time.time()
        #
        #gsm.create(4)
        #print(gsm.portfolio.history)
        futures = Futures()
        futures_generator=futures.create_simulation_data(2)
        futures = next(futures_generator)
    else:
        start_time = time.time()
    

    
    sym='BP'
    
    #self.train(sym)
    total_big_count=pd.DataFrame(index=[-1,0,1])
    total_small_count=pd.DataFrame(index=[-1,0,1])
    for sym in futures.dic.index:
        
        
        
        data=futures.data_dict[sym][-120:]
        #ZZM_120D_3S_7PV -1 35% 0 19% 1 46%
        #ZZM_120D_4S_4PV -1 38% 0 15% 1 47%
        #ZZM_120D_4S_5PV -1 35% 0 19% 1 45%
        #ZZM_120D_4S_6PV -1 32% 0 25% 1 43%
        #*ZZM_120D_4S_7PV -1 30% 0 31% 1 39%
        #ZZM_120D_7S_7PV -1 12% 0 65% 1 21%
        '''stdev starts peak/valley, then grinds down, lower number, less iterations'''
        self=zigzag(sym, data, zigzag_stdev=4) 
        '''
        min_peaks_valleys - stops when reaches # peaks of valleys specified
        return_threshold_stdev - higher number more zeros in ZZT signal
        '''
        self.get_peaks_valleys(return_threshold_stdev=4, min_peaks_valleys=7,
                               verbose=True, show_plots=False)
        #data=futures.data_dict[sym][-90:]
        #*ZZM_90D_3S_8PV -1 34%, 0 30% 1 35%
        #ZZM_90D_4S_6PV -1 29%, 0 38% 1 32%
        #ZZM_90D_4S_5PV -1 33%, 0 30% 1 37%
        

        
        #data=futures.data_dict[sym][-60:]  
        #ZZM_60D_3S_5PV -1 28%, 0 32% 1 40%
        #ZZM_60D_3S_6PV -1 25%, 0 37% 1 37%
        #*ZZM_60D_2S_9PV -1 32%, 0 30% 1 37%
        #ZZM_60D_2S_5PV -1 38%, 0 14% 1 47%
        #ZZM_60D_1S_5PV -1 46%, 0 2% 1 51%
        #ZZM_60D_1S_7PV -1 46%, 0 4% 1 50%  
        '''stdev starts peak/valley, then grinds down, lower number, less iterations'''
        #self=zigzag(sym, data, zigzag_stdev=4) 
        '''
        min_peaks_valleys - stops when reaches # peaks of valleys specified
        return_threshold_stdev - higher number more zeros in ZZT signal
        '''
        #self.get_peaks_valleys(return_threshold_stdev=2, min_peaks_valleys=9,
        #                       verbose=True, show_plots=False)
        
        total_big_count[sym]=self.modes_with_big_count
        total_small_count[sym]=self.modes_with_small_count
        
    total_big_count['totals']=total_big_count.sum(axis=1)
    total_big_count['totals_per']=total_big_count['totals']/total_big_count['totals'].sum()
    print('big swings\n', total_big_count[['totals','totals_per']])
    total_small_count['totals']=total_small_count.sum(axis=1)
    total_small_count['totals_per']=total_small_count['totals']/total_small_count['totals'].sum()
    print('small swings\n', total_small_count[['totals','totals_per']])
    print('Elapsed time: {} minutes. {}'\
      .format(round(((time.time() - start_time) / 60), 2), dt.now()))