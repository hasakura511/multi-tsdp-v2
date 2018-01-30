#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 16:15:37 2018

@author: hidemiasakura

appends signals to the atr files

"""
import sys
import numpy as np
import pandas as pd
from nimbus.process.transform import average_true_range, to_signals, get_ga_hv_lv
from nimbus.process.seasonality import Seasonality
from nimbus.process.zigzag import zigzag
from nimbus.process.ml import ML

class Signals(object):
    
    def __init__(self, futures):
        #we are directly modifing the futures object
        self.futures=futures
        self.SHORT_LOOKBACK=60
        self.MID_LOOKBACK=90
        self.LONG_LOOKBACK=120
        self.START_ZIGZAG_STD=4.0
        self.GA_VOLATILITY_CUTOFF=0.5
        self.training_signals={}
        
    def add_previous_signals(self):
        print('adding previous signals..')
        for sym in self.futures.data_dict:
            close = self.futures.data_dict[sym].Close
            
            #prev signals
            act1 = -1 if (close[-1]/close[-2]) < 0 else 1
            act2 = -1 if (close[-1]/close[-3])-1 < 0 else 1
            act3 = -1 if(close[-1]/close[-4])-1 < 0 else 1
            act5 = -1 if (close[-1]/close[-6])-1 < 0 else 1
            act10 = -1 if(close[-1]/close[-11])-1 < 0 else 1
            act20 = -1 if(close[-1]/close[-21])-1 < 0 else 1
            
            #price change related signals
            self.futures.atr.set_value(sym,'signals_PREVIOUS_1D',act1)
            self.futures.atr.set_value(sym,'signals_ANTI_PREVIOUS_1D',-act1)
            self.futures.atr.set_value(sym,'signals_PREVIOUS_2D',act2)
            self.futures.atr.set_value(sym,'signals_ANTI_PREVIOUS_2D',-act2)
            self.futures.atr.set_value(sym,'signals_PREVIOUS_3D',act3)
            self.futures.atr.set_value(sym,'signals_ANTI_PREVIOUS_3D',-act3)            
            self.futures.atr.set_value(sym,'signals_PREVIOUS_5D',act5)
            self.futures.atr.set_value(sym,'signals_ANTI_PREVIOUS_5D',-act5)
            self.futures.atr.set_value(sym,'signals_PREVIOUS_10D',act10)
            self.futures.atr.set_value(sym,'signals_ANTI_PREVIOUS_10D',-act10)
            self.futures.atr.set_value(sym,'signals_PREVIOUS_20D',act20)
            self.futures.atr.set_value(sym,'signals_ANTI_PREVIOUS_20D',-act20)
        
    def add_seasonality_signals(self):
        print('adding Seasonality signals..')
        for sym in self.futures.data_dict:
            data = self.futures.data_dict[sym]
            #seasonality
            seasonality=Seasonality()
            signal_SEA, signal_ADJSEA, validation_start_date = seasonality\
                                                        .get_signals(sym, data)
            #seasonality.show_chart()
            
            self.futures.atr.set_value(sym,'signals_SEASONALITY',signal_SEA)
            self.futures.atr.set_value(sym,'signals_ANTI_SEASONALITY',-signal_SEA)
            self.futures.atr.set_value(sym,'signals_ADJSEASONALITY',signal_ADJSEA)
            self.futures.atr.set_value(sym,'signals_ANTI_ADJSEASONALITY',-signal_ADJSEA)
            
    def train_and_predict(self):
        '''trains and predicts ml models'''
        '''
        self.ML=ML(self.futures)
        self.ML.prepare() #prepares training data
        self.ML.train() #trains models
        self.ML.predict() #appends signals to future.atr
        for sym in self.futures.data_dict:
            self.futures.atr.set_value(sym,'signals_ANTI_ADJSEASONALITY',-signal_ADJSEA)
        '''
        
    def generate_ml_signals(self, lookback, min_pv, min_return_std):
        self.training_signals[self.last_date][lookback]={}
        
        for sym in self.futures.data_dict:
            signals=pd.DataFrame()
            data = self.futures.data_dict[sym][-lookback:]
            
            self.zigzag_long=zigzag(sym, data, zigzag_stdev=self.START_ZIGZAG_STD)
            trend, mode =\
                    self.zigzag_long.get_peaks_valleys(min_peaks_valleys=min_pv, 
                                               return_threshold_stdev=min_return_std,
                                               verbose=False, show_plots=False)
    
            signals['signals_SWING_ALL_NZ_{}D_{}S_{}PV'.format(lookback,
                 min_return_std, min_pv)]=\
                pd.Series(data=self.zigzag_long.modes_no_zeros, 
                      index=data.index).shift(-1).fillna(0)
            signals['signals_SWING_SMALL_WZ_{}D_{}S_{}PV'.format(lookback,
                 min_return_std, min_pv)]=\
                pd.Series(data=self.zigzag_long.modes_with_small_swings, 
                      index=data.index).shift(-1).fillna(0)
            signals['signals_SWING_BIG_WZ_{}D_{}S_{}PV'.format(lookback,
                 min_return_std, min_pv)]=\
                pd.Series(data=self.zigzag_long.modes_with_big_swings, 
                      index=data.index).shift(-1).fillna(0)
                
                
            #gain_ahead signals
            close = data.Close.copy()
            gain_ahead=close.pct_change().shift(-1).fillna(0)
            #find where gain ahead was 0, other than the last index
            zero_gains_index=gain_ahead.reset_index()\
                    [(gain_ahead==0).values].index.tolist()[:-1]
            #if gain ahead was 0, then set it to the next value
            for i in zero_gains_index:
                gain_ahead.set_value(gain_ahead.index[i], gain_ahead[i+1])
                #print(data.iloc[i])
                #print(data.iloc[i+1])
            
            ga_signals=to_signals(gain_ahead)
            ga_lv_signals, ga_hv_signals = get_ga_hv_lv(gain_ahead,
                                                        self.GA_VOLATILITY_CUTOFF)
            
            #print('ga_hv_signals\n', ga_hv_signals[:-1].value_counts())
            #print('ga_lv_signals\n', ga_lv_signals[:-1].value_counts())
            signals['signals_SCALP_ALL_GA_{}D'.format(lookback)]=ga_signals
            signals['signals_SCALP_SMALL_GA_LV_{}D'.format(lookback)]=ga_lv_signals
            signals['signals_SCALP_BIG_GA_HV_{}D'.format(lookback)]=ga_hv_signals
            
            #add to training df
            self.training_signals[self.last_date][lookback][sym]=signals.copy()
    
            #add to atr
            self.futures.atr.set_value(sym,
                       'signals_ZZTREND_{}D'.format(lookback),trend)
            self.futures.atr.set_value(sym,
                       'signals_ANTI_ZZTREND_{}D'.format(lookback),-trend)
            self.futures.atr.set_value(sym,
                       'signals_ZZMODE_{}D'.format(lookback),mode)
            self.futures.atr.set_value(sym,
                       'signals_ANTI_ZZMODE_{}D'.format(lookback),-mode)
                    
    
    def add_ml_signals(self):
        '''generate signals for ml training, append state signals'''
        print('adding ml signals..')
        self.last_date=self.futures.last_date
        self.training_signals[self.last_date]={}
    
        #zigzag params
        #ZZM_120D_3S_7PV -1 35% 0 19% 1 46%
        #ZZM_120D_4S_4PV -1 38% 0 15% 1 47%
        #ZZM_120D_4S_5PV -1 35% 0 19% 1 45%
        #ZZM_120D_4S_6PV -1 32% 0 25% 1 43%
        #*ZZM_120D_4S_7PV -1 30% 0 31% 1 39%
        #ZZM_120D_7S_7PV -1 12% 0 65% 1 21%
        min_pv=7
        min_return_std=4
        lookback=self.LONG_LOOKBACK
        self.generate_ml_signals(lookback, min_pv, min_return_std)


        #*ZZM_90D_3S_8PV -1 34%, 0 30% 1 35%
        #ZZM_90D_4S_6PV -1 29%, 0 38% 1 32%
        #ZZM_90D_4S_5PV -1 33%, 0 30% 1 37%
        min_pv=8
        min_return_std=3
        lookback=self.MID_LOOKBACK
        self.generate_ml_signals(lookback, min_pv, min_return_std)    
        
        
        #zigzag params
        #ZZM_60D_3S_5PV -1 28%, 0 32% 1 40%
        #ZZM_60D_3S_6PV -1 25%, 0 37% 1 37%
        #*ZZM_60D_2S_9PV -1 32%, 0 30% 1 37%
        #ZZM_60D_2S_5PV -1 38%, 0 14% 1 47%
        #ZZM_60D_1S_5PV -1 46%, 0 2% 1 51%
        #ZZM_60D_1S_7PV -1 46%, 0 4% 1 50%
        min_pv=9
        min_return_std=2
        lookback=self.SHORT_LOOKBACK
        self.generate_ml_signals(lookback, min_pv, min_return_std)

        
        
