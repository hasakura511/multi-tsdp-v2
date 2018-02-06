# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 20:57:32 2015

@author: hidemi
"""
import math
import numpy as np
import pandas as pd

class Ratios(object):
	'''
	all list parameters are expected to be an one dimensional
	list of nominal prices, e.g. [1,1.2,.3,10,.3,25]
	'''
	def __init__(self, prices, benchmark = None):
		self.prices = prices
		self.n = len(self.prices)
		self.benchmark = self._prepare_benchmark(benchmark)
		self.ret = np.diff(self.prices)
		self.b_ret = np.diff(self.benchmark)

	def sharpe(self):

		adj_ret = [a - b for a, b in zip(self.ret, self.b_ret)]
		std = np.std(self.ret)

		return self._get_info_ratio(adj_ret, std)

	def sortino(self):
		'''
		sortino is an adjusted ratio which only takes the 
		standard deviation of negative returns into account
		'''
		adj_ret = [a - b for a, b in zip(self.ret, self.b_ret)]
		avg_ret = np.mean(adj_ret)

		# Take all negative returns.
		neg_ret = [a ** 2 for a in adj_ret if a < 0]
		# Sum it.
		neg_ret_sum = np.sum(neg_ret)
		# And calculate downside risk as second order lower partial moment.
		down_risk = np.sqrt(neg_ret_sum / self.n)

		if down_risk > 0.0001:
			sortino = avg_ret / down_risk
		else:
			sortino = 0

		return sortino

	def _get_info_ratio(self, ret, std):

		avg = np.mean(ret)

		if std > 0.0001:
			return avg * np.sqrt(self.n) / std
		else:
			return 0

	def _prepare_benchmark(self, benchmark):

		if benchmark == None:
			benchmark = np.zeros(self.n)
		if len(benchmark) != self.n:
			raise Exception("benchmark mismatch")

		return benchmark

def get_ga_hv_lv(gain_ahead, GA_VOLATILITY_CUTOFF):
    cutoff=abs(gain_ahead).quantile(GA_VOLATILITY_CUTOFF)
    hv_index=gain_ahead[(gain_ahead<-cutoff) | (gain_ahead>cutoff)].index
    lv_index=[x for x in gain_ahead.index if x not in hv_index]
    for d in gain_ahead.index:
        if d == gain_ahead.index[0]:
            ga_lv_signals=pd.Series(index=gain_ahead.index)
            ga_hv_signals=pd.Series(index=gain_ahead.index)
            
        if d in hv_index:
            signal =  1 if gain_ahead.loc[d] > 0 else -1
            ga_hv_signals.set_value(d,signal)
        else:
            ga_hv_signals.set_value(d,0)
            
        if d in lv_index:
            signal =  1 if gain_ahead.loc[d] > 0 else -1
            ga_lv_signals.set_value(d,signal)
        else:
            ga_lv_signals.set_value(d,0)
        
        if d == gain_ahead.index[-1]:
            #last index is 0
            ga_lv_signals.set_value(d,0)
            ga_hv_signals.set_value(d,0)
            
    return ga_lv_signals, ga_hv_signals

def is_int(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False
    
def to_signals(df):
    signals=[]
    for i in range(df.shape[0]):
        if df.iloc[i]>0:
            signals.append(1)
        elif df.iloc[i]<0:
            signals.append(-1)
        else:
            signals.append(0)
    return pd.Series(signals, index=df.index).astype(int)

def DPO(p,lb):
    # Detrended price oscillator. 
    # A high pass filter.
    # p, the series being transformed.
    # lb, the lookback period, a real number.
    # Uses pandas ewma function.
    # Return is a numpy array with values centered on 0.0.
    #nrows = p.shape[0]
    ma = p.ewm(span=lb).mean()
    return (p-ma)/ma
    
def roofing_filter(p,lb,bars=10.0):
    if lb <3:
        print('lookback < 3. adjusting lookback minimum to 3')
        lb =3    
    if type(p) is pd.core.series.Series:
        p = p.values
        
    rad360 = math.radians(360)
    sqrt2div2 = math.sqrt(2)/2
    alpha = (math.cos(sqrt2div2*rad360/bars)+\
            math.sin(sqrt2div2*rad360/bars)-1)/\
            math.cos(rad360*sqrt2div2/bars)
    a1 = math.exp(-math.sqrt(2)*math.pi/10.0)
    b1 = 2.0*a1*math.cos(math.sqrt(2)*math.radians(180)/10.0)
    
    c3 = -a1*a1
    c2 = b1
    c1 = 1-c2-c3
    
    nrows = p.shape[0]
    highpass = np.zeros(nrows)
    
    for i in range(2,nrows):
        highpass[i] = (1-alpha/2.0)*(1-alpha/2.0)*\
                (p[i]-2*p[i-1]+p[i-2])+2*(1-alpha)*\
                highpass[i-1]-(1-alpha)*(1-alpha)*highpass[i-2]
    
    filt=np.zeros(nrows)
    for i in range(2,nrows):
        filt[i] = c1*(highpass[i]+highpass[i-1])/2.0+\
                c2*filt[i-1]+c3*filt[i-2]
    
    high=np.zeros(nrows)
    low=np.zeros(nrows)
    #soften ramp up
    for i in range(3,lb):
        high[i] = max(filt[2:i])
        low[i] = min(filt[2:i])
                
    stoch=np.zeros(nrows)
    rstoch=np.zeros(nrows)
    for i in range(lb,nrows):
        high[i] = max(filt[i-lb:i])
        low[i] = min(filt[i-lb:i])
        stoch[i] = (filt[i]-low[i])/(high[i]-low[i])
        rstoch[i] = c1*(stoch[i]+stoch[i-1])/2.0+\
                c2*rstoch[i-1]+c3*rstoch[i-2]
    return rstoch
    

def average_true_range(ph,pl,pc,lb):
    #not as %of close
    # Average True Range technical indicator.
    # ph, pl, pc are the series high, low, and close.
    # lb, the lookback period.  An integer number of bars.
    # True range is computed as a fraction of the closing price.
    # Return is a numpy array of floating point values.
    # Values are non-negative, with a minimum of 0.0.

    nrows = pc.shape[0]
    th = np.zeros(nrows)
    tl = np.zeros(nrows)
    tc = np.zeros(nrows)
    tr = np.zeros(nrows)
    trAvg = np.zeros(nrows)
    
    for i in range(1,nrows):
        if ph[i] > pc[i-1]:
            th[i] = ph[i]
        else:
            th[i] = pc[i-1]
        if pl[i] < pc[i-1]:
            tl[i] = pl[i]
        else:
            tl[i] = pc[i-1]
        tr[i] = th[i] - tl[i]
    for i in range(lb,nrows):
        trAvg[i] = tr[i]            
        for j in range(1,lb-1):
            trAvg[i] = trAvg[i] + tr[i-j]
        trAvg[i] = trAvg[i] / lb
        trAvg[i] = trAvg[i]
    return(trAvg)
    
def atr_df(df,lb):
    #not as %of close
    # Average True Range technical indicator.
    # ph, pl, pc are the series high, low, and close.
    # lb, the lookback period.  An integer number of bars.
    # True range is computed as a fraction of the closing price.
    # Return is a numpy array of floating point values.
    # Values are non-negative, with a minimum of 0.0.
    ph=df.High
    pl=df.Low
    pc=df.Close
    nrows = pc.shape[0]
    th = np.zeros(nrows, dtype=float)
    th.fill(np.nan)
    tl = np.zeros(nrows, dtype=float)
    tl.fill(np.nan)
    tc = np.zeros(nrows, dtype=float)
    tc.fill(np.nan)
    tr = np.zeros(nrows, dtype=float)
    tr.fill(np.nan)
    trAvg = np.zeros(nrows, dtype=float)
    trAvg.fill(np.nan)
    
    for i in range(1,nrows):
        if ph[i] > pc[i-1]:
            th[i] = ph[i]
        else:
            th[i] = pc[i-1]
        if pl[i] < pc[i-1]:
            tl[i] = pl[i]
        else:
            tl[i] = pc[i-1]
        tr[i] = th[i] - tl[i]
    for i in range(lb,nrows):
        trAvg[i] = tr[i]            
        for j in range(1,lb-1):
            trAvg[i] = trAvg[i] + tr[i-j]
        trAvg[i] = trAvg[i] / lb
        trAvg[i] = trAvg[i]
    return pd.Series(trAvg, index=df.index)