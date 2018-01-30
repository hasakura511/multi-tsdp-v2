#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 11:29:40 2018

@author: hidemiasakura

seasonality
inputs futures data
output signals, charts
"""


import numpy as np
import pandas as pd
#from nimbus.portfolio import Portfolio
#from nimbus.process.transform import ATR2
#import matplotlib.pyplot as plt
#import seaborn as sns
from datetime import timedelta
#from os import listdir
#from os.path import isfile, join
#import calendar
#from datetime import datetime as dt
#import warnings
#warnings.simplefilter('error')

class zigzag(object):
    '''
    	all list parameters are expected to be an one dimensional
    	list of nominal prices, e.g. [1,1.2,.3,10,.3,25]
    '''
    def __init__(self, prices, up_thresh, down_thresh):
        self.prices = prices
        self.up_thresh = up_thresh
        self.down_thresh = down_thresh
        self.pivots = self.peak_valley_pivots()
        self.initial_pivot = self._identify_initial_pivot()
        self.eCurves = {}
      
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
            pivots[t_n-1] = -trend
    
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
        return modes
    
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
        
class Seasonality(object):
    
    def __init__(self):
        self.lookback=270
        self.chart_length=8
        self.chart_width=8
        self.price_zigzag_std=2.5
        self.seasonality_zigzag_std=3.5
        self.zscore_window=60
        self.roll_corr_window=10
        self.pivotdate_lookforward=5
        self.min_validation_length=20
        self.save_path='./nimbus/debug/'
        self.data = pd.DataFrame()
        self.symbol=''
        self.last_date=''
        
    def get_signals(self, symbol, data):
        #data2=data
        if data.shape[0]<self.lookback:
            print('Warning! datapoints {} < required minimum'.format(data.shape[0], self.lookback))
        self.symbol=symbol
        self.data=data.copy()
        self.last_date=self.data.index[-1].strftime("%Y%m%d")
        
        pstd=self.data.Close.pct_change().std()
        self.zzp = zigzag(self.data.Close.values,\
                     pstd*self.price_zigzag_std,pstd*-self.price_zigzag_std)
        self.zzp_pivots=self.zzp.peak_valley_pivots()
        
        sstd=self.data.S.pct_change().std()
        self.zzs = zigzag(self.data.S.values,\
                     sstd*self.seasonality_zigzag_std,sstd*-self.seasonality_zigzag_std)
        self.zzs_pivots=self.zzs.peak_valley_pivots()
        
        #create comparison runs where positive is sea trending with price, and negative is anti-trending
        self.runs=pd.DataFrame(np.where(self.zzp.pivots_to_modes()==self.zzs.pivots_to_modes(),1,-1),columns=['normal'])
        self.runs['block'] = (self.runs['normal'] != self.runs['normal'].shift(1)).astype(int).cumsum()
        self.runs['count'] = self.runs.groupby('block').transform(lambda x: range(1, len(x) + 1)[-1])
                
        self.current_run = self.runs['normal'].iloc[-1]*self.runs['count'].iloc[-1]
        
        #uses the seasonality rather than the price because the last zzpivot for the price may change in the future
        i=-2
        self.validation_start_date=self.data.index[np.nonzero(self.zzp_pivots)[0]][i]
        self.validation_length=len(self.data.loc[self.validation_start_date:])
        #print 'sea',self.data.index[np.nonzero(self.zzs_pivots)[0]],len(self.data.loc[self.data.index[np.nonzero(self.zzs_pivots)[0]][i]:]), self.validation_start_date
        #print 'price',self.data.index[np.nonzero(self.zzp_pivots)[0]],len(self.data.loc[self.data.index[np.nonzero(self.zzp_pivots)[0]][i]:]), self.validation_start_date
        while self.validation_length<self.min_validation_length:
            i-=1
            self.validation_start_date=self.data.index[np.nonzero(self.zzp_pivots)[0]][i]
            self.validation_length=len(self.data.loc[self.validation_start_date:])
            #print 'price',self.data.index[np.nonzero(self.zzp_pivots)[0]],len(self.data.loc[self.data.index[np.nonzero(self.zzp_pivots)[0]][i]:])
            #print 'sea',self.data.index[np.nonzero(self.zzs_pivots)[0]],len(self.data.loc[self.data.index[np.nonzero(self.zzs_pivots)[0]][i]:]), self.validation_start_date
            #print 'price',self.data.index[np.nonzero(self.zzp_pivots)[0]],self.validation_length, self.validation_start_date
        i=-2
        validationStartDate2=self.data.index[np.nonzero(self.zzs_pivots)[0]][i]
        validationLength2=len(self.data.loc[validationStartDate2:])

        while validationLength2<self.min_validation_length:
            i-=1
            validationStartDate2=self.data.index[np.nonzero(self.zzs_pivots)[0]][i]
            validationLength2=len(self.data.loc[validationStartDate2:])
        
        #print self.validation_start_date, validationStartDate2, self.validation_length, validationLength2
        #print self.validation_start_date> validationStartDate2, self.validation_length> validationLength2
        if self.validation_length> validationLength2:
            self.validation_start_date =validationStartDate2
            self.validation_length=validationLength2
            
        #find next seasonal pivot, +5 for to lookahead of weekend/larger lookforward bias
        i=1 
        #print(self.data.index[np.nonzero(self.zzs_pivots)[0][i]].to_pydatetime())
        pivotDate=(self.data.index[np.nonzero(self.zzs_pivots)[0][i]].to_pydatetime().year+1)*10000+\
                        self.data.index[np.nonzero(self.zzs_pivots)[0][i]].to_pydatetime().month*100\
                        +self.data.index[np.nonzero(self.zzs_pivots)[0][i]].to_pydatetime().day
        #currentDate=self.data.index[-1].to_pydatetime().year*10000+self.data.index[-1].to_pydatetime().month*100\
        #                    +self.data.index[-1].to_pydatetime().day+self.pivotdate_lookforward
        currentDate=int((self.data.index[-1].to_pydatetime()+ timedelta(days=self.pivotdate_lookforward)).strftime('%Y%m%d'))
        #print(i,self.symbol, 'next seasonal pivot',pivotDate,'>',currentDate, currentDate<pivotDate)
        self.currentDate=currentDate
        while currentDate>pivotDate:
            i+=1
            pivotDate=(self.data.index[np.nonzero(self.zzs_pivots)[0][i]].to_pydatetime().year+1)*10000+\
                            self.data.index[np.nonzero(self.zzs_pivots)[0][i]].to_pydatetime().month*100\
                            +self.data.index[np.nonzero(self.zzs_pivots)[0][i]].to_pydatetime().day
            #print(i,self.symbol, 'next seasonal pivot',pivotDate,'>',currentDate, currentDate<pivotDate)
        
        self.next_seasonal_pivot=pivotDate
        self.next_seasonality=round(self.data.S.iloc[np.nonzero(self.zzs_pivots)[0][i]],2)
        
        self.current_seasonality=round(self.data.S[-1],2)
        self.signal = 1 if self.next_seasonality>self.current_seasonality else -1
        self.signal_adjsea= 1 if self.current_run*self.signal>0 else -1
        return self.signal, self.signal_adjsea, self.validation_start_date
        '''  
        corr = pd.rolling_corr(self.data.Close.pct_change().dropna().values,\
                                        self.data.S.pct_change().shift(-1).dropna().values,window=self.roll_corr_window) 
        #corr=pd.Series(np.insert(corr[:-1],0,np.nan), index=self.data.index).ewm(com=0.5).mean()
        corr=pd.ewma(pd.Series(np.insert(corr,0,np.nan), index=self.data.index),com=0.5)
        
        #ax4 spread
        #res = ols(y=data2.Close, x=data2.S)
        res = sm.OLS(data2.Close, data2.S).fit()

        #spread=data2.Close-res.beta.x*data2.S      
        spread=data2.Close-res.params.S*data2.S
        #zs_spread= ((spread - spread.rolling(zs_window).mean())/spread.rolling(zs_window).std()).ix[self.data.index]
        zs_spread= ((spread - pd.rolling_mean(spread,self.zscore_window))/pd.rolling_std(spread,self.zscore_window)).ix[self.data.index]
        '''
    
    def show_chart(self, save=False):
        if len(self.data)<1:
            print('get_signals() first')
            return
        
        import matplotlib.pyplot as plt
        #import matplotlib.ticker as tick
        import matplotlib.dates as mdates
        '''
        def format_date(x, pos=None):
            thisind = np.clip(int(x + 0.5), 0, self.lookback - 1)
            #print thisind,index
            return self.data.index[thisind].strftime("%Y-%m-%d %H:%M")
            
        def align_yaxis(ax1, v1, ax2, v2):
            """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
            _, y1 = ax1.transData.transform((0, v1))
            _, y2 = ax2.transData.transform((0, v2))
            inv = ax2.transData.inverted()
            _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
            miny, maxy = ax2.get_ylim()
            ax2.set_ylim(miny+dy, maxy+dy)
        '''    
        months = mdates.MonthLocator()        # major ticks on the mondays
        weeks = mdates.WeekdayLocator(mdates.MONDAY)              # minor ticks on the days
        years = mdates.DateFormatter('%b %Y')  # e.g., Jan 12
        #minorFormat = DateFormatter('%d')      # e.g., 12
        #correlated 
        self.correlated =self.data.index[self.runs['normal'] ==1]
        #anticorrelated 
        self.anticorrelated = self.data.index[self.runs['normal'] ==-1]
        
        self.price_runs=pd.DataFrame(self.zzp.pivots_to_modes(),columns=['normal'])
        self.price_runs['block'] = (self.price_runs['normal'] != self.price_runs['normal'].shift(1)).astype(int).cumsum()
        self.price_runs['count'] = self.price_runs.groupby('block').transform(lambda x: range(1, len(x) + 1)[-1])

        #top axis
        fig = plt.figure(figsize=(self.chart_width,self.chart_length))
        fig.subplots_adjust(bottom=0.2)
        #ax=plt.subplot2grid((1,1), (0,0), rowspan=1, colspan=1)
        ax = fig.add_subplot(111, autoscale_on=True)
        ax.xaxis.set_major_formatter(years)
        ax.xaxis.set_major_locator(months)
        ax.xaxis.set_minor_locator(weeks)
        ax.set_xlim(self.data.index[0], self.data.index[-1])
        ax.format_xdata=mdates.DateFormatter('%Y-%m-%d')
        ax.grid(True)
        ax.plot(self.data.index, self.data.Close, 'b:', alpha=0.5, label=str(self.next_seasonal_pivot)+' Bias '+str(self.signal))
        ax.plot(self.data.index[self.zzp_pivots != 0], self.data.Close[self.zzp_pivots != 0], alpha=0.4, color='c',ls='-',\
                    label='SRUN'+str(self.current_run)+' ZZ'+str(self.price_zigzag_std))
        #v start
        ax.annotate('', (self.data.index[-self.validation_length], self.data.Close.iloc[-self.validation_length]),
                         arrowprops=dict(facecolor='magenta', shrink=0.03), xytext=(-20,0), textcoords='offset points',
                         size='medium', alpha=0.6)
        #lb
        #ax.annotate('', (self.data.index[-self.zscore_window], self.data.Close.iloc[-self.zscore_window]),
        #                 arrowprops=dict(facecolor='violet', shrink=0.03), xytext=(-20,0), textcoords='offset points',
        #                 size='medium', alpha=0.6)
        
        text='validation start {} / last date {}'.format(self.validation_start_date.strftime(" %Y-%m-%d"),\
                               self.data.index[-1].strftime("%Y-%m-%d"))
        ax.annotate(text, xy=(0.30, 0.03), ha='left', va='top', xycoords='axes fraction', fontsize=12)        
        ax.yaxis.set_label_position("left")
        ax.set_ylabel('Price', size=12)
        ax.set_title(self.symbol+' Price vs. Seasonality')
        #ax.grid(which='major', linestyle='-', color='white') 
        plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
        #ax.scatter(correlated, self.data.Close.loc[correlated], color='k', label=str(int((float(len(correlated))/lb)*100))+'% Correlated')
        #ax.scatter(anticorrelated, self.data.Close.loc[anticorrelated], color='r', label=str(int((float(len(anticorrelated))/lb)*100))+'% Anti-Correlated')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc='lower right',prop={'size':10}, bbox_to_anchor=(1.1, 1))
        #ax.legend(handles, labels, loc='lower right',prop={'size':10})
        
        #top axis 2
        ax2p=ax.twinx()
        ax2p.plot(self.data.index, self.data.S, 'g:', alpha=0.5, label=str(self.current_seasonality)+' Seasonality')
        ax2p.plot(self.data.index[self.zzs_pivots != 0], self.data.S[self.zzs_pivots != 0], alpha=0.8, color='green',ls='-',\
                            label=str(self.next_seasonality)+' ZZ'+str(self.seasonality_zigzag_std)+' Seasonality')
        ax2p.axhline(self.next_seasonality, color='magenta', alpha=0.6)
        ax2p.axhline(self.current_seasonality, color='violet', alpha=0.8)

        mask = (self.runs['normal'] != self.runs['normal'].shift(-1))
        for i in self.runs[mask].index:
            if self.runs['normal'][i]<0:
                xytext=(0,0)
                color='r'
            else:
                xytext=(0,0)
                color='k'
                
            ax2p.annotate(str(self.runs['count'][i]), (self.data.index[i], self.data.S[i]),
                         xytext=xytext, textcoords='offset points',color=color,
                         size='medium')
            #ax.annotate(str(self.runs['count'][i]), (self.data.index[i], self.data.Close[i]),
            #             xytext=xytext, textcoords='offset points',color=color,
            #             size='medium')
            
        mask = (self.price_runs['normal'] != self.price_runs['normal'].shift(-1))
        mask[0]=False
        returns=self.zzp.compute_segment_returns()*100
        for h,i in enumerate(self.price_runs[mask].index):
            if self.price_runs['normal'][i]<0:
                xytext=(0,0)
                xytext2=(0,-10)
                color='r'
            else:
                xytext=(0,0)
                xytext2=(0,10)
                color='k'
                
            ax.annotate(str(self.price_runs['count'][i]), (self.data.index[i], self.data.Close[i]),
                         xytext=xytext, textcoords='offset points',color=color,
                         size='medium')
            ax.annotate(str(round(returns[h],2))+'%', (self.data.index[i], self.data.Close[i]),
             xytext=xytext2, textcoords='offset points',color=color,
             size='medium')
            
        handles, labels = ax2p.get_legend_handles_labels()
        ax2p.legend(handles, labels, loc='lower left',prop={'size':10}, bbox_to_anchor=(-0.15, 1))
        ax2p.yaxis.set_label_position("right")
        ax2p.set_ylabel('Seasonality', size=12)
        #ax2p.grid(which='major', linestyle='--', color='white') 
        #ax2p.set_yticks(np.linspace(ax2p.get_yticks()[0], ax2p.get_yticks()[-1], len(ax.get_yticks())))

        '''
        #bottom axis 1
        ax3 = plt.subplot2grid((2,1), (1,0), rowspan=1, colspan=1, sharex=ax)
        #ax3 = fig.add_subplot(212, autoscale_on=False)
        ax3.xaxis.set_major_formatter(majorFormat)
        #ax.xaxis.set_major_formatter(tick.FuncFormatter(format_date))
        ax3.xaxis.set_major_locator(major)
        ax3.xaxis.set_minor_locator(minor)
        #ax3.plot(self.data.index, np.nan_to_num(corr), 'r:', alpha=0.5)
        ax3.plot(self.data.index, corr, 'k:', alpha=0.5, label=str(round(corr.iloc[-1],2))+' Correlation lb'+str(self.roll_corr_window)+' lag1')
        ax3.scatter(self.correlated, corr.loc[self.correlated], color='g', alpha=0.6,\
                        label=str(int((float(len(self.correlated))/lb)*100))+'% Correlated')
        ax3.scatter(self.anticorrelated, corr.loc[self.anticorrelated], color='r', alpha=0.6,\
                        label=str(int((float(len(self.anticorrelated))/lb)*100))+'% Anti-Correlated')
        ax3.yaxis.set_label_position("left")
        ax3.set_ylabel('Correlation', size=12)
        ax3.set_title(self.symbol+' Spread & Correlation')
        ax3.set_ylim((-1,1))
        #ax3.axhline(0, color='white')
        handles, labels = ax3.get_legend_handles_labels()
        ax3.legend(handles, labels, loc='lower left',prop={'size':10}, bbox_to_anchor=(0, .94))

        #annotate last index
        ax3.annotate(self.data.Close.index[-1].strftime("%Y-%m-%d %H:%M"),\
                    xy=(0.78, 0.025), ha='left', va='top', xycoords='axes fraction', fontsize=12)        

        
        plt.setp(ax3.get_xticklabels(), rotation=45, horizontalalignment='right')
        
        #bottom axis 2
        ax4=ax3.twinx()
        #ax4.plot(self.data.index, zs_spread, 'k-', alpha=0.5, label='ZS Spread lb'+str(zs_window))
        ax4.fill_between(self.data.index, zs_spread, color='#0079a3', alpha=0.4,\
                                label=str(round(zs_spread[-1],2))+' ZS Spread lb'+str(self.zscore_window)) 
        ax4.yaxis.set_label_position("right")
        ax4.set_ylabel('Spread', size=12)
        ax4.set_ylim((np.floor(min(zs_spread.fillna(0))),np.ceil(max(zs_spread.fillna(0)))))
        #ax4.grid(which='major', linestyle='--', color='white') 
        align_yaxis(ax3, 0, ax4, 0)
        handles, labels = ax4.get_legend_handles_labels()
        ax4.legend(handles, labels, loc='lower right',prop={'size':10}, bbox_to_anchor=(1, .97))
        ax4.set_xlim(self.data.index[0],self.data.index[-1])
        
        #annotate self.runs
        for i in self.runs[mask].index:
            if not np.isnan(corr[i]):
                if self.runs['normal'][i]<0:
                    xytext=(2,-2)
                    color='r'
                else:
                    xytext=(2,-2)
                    color='k'
                #print self.runs['count'][i], (self.data.index[i], corr[i])
                ax3.annotate(str(self.runs['count'][i]), (self.data.index[i], corr[i]),\
                             xytext=xytext, textcoords='offset points',color=color, size='medium')
        '''
        #save/show plots
        plt.show()
        
        if save:
            self.filename = self.save_path+self.last_date+'_'+self.symbol+'_SEASONALITY.png'
            print('Saving '+self.filename+'.png')
            fig.savefig(self.filename, bbox_inches='tight')
            
        plt.close()


if __name__ == '__main__':
    self=Seasonality()
    self.get_signals('TEST', data)
    self.show_chart()
