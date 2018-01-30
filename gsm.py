#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 12:44:23 2018

@author: hidemiasakura
Python 3.6

Nimbus System
one stop shop to get data.
the backbone of the data processing pipline

inputs: account config, board config
output: chart data

"""

import pandas as pd
import numpy as np
import calendar
import time
from datetime import datetime as dt
from nimbus.portfolio import Portfolio
from nimbus.csidata import Futures
#system files
#dic=pd.read_csv('./nimbus/data/system/dictionary.csv', index_col=0)
#dicfx=pd.read_csv('./nimbus/data/system/dictionary_fx.csv', index_col=0)




#params
params=  dict(
        account_value=5000,
        margin_percent=0.5,
        benchmark_sym='ES',
        correlation_cutoff=0.7,
        recreate_if_margin_call=False,
        increment=250,
        board_config=[
                      {
                        "color": "pink",
                        "position": "left",
                        "display": "Previous (1 day)",
                        "id": "PREVIOUS_1D"
                      },
                      {
                        "color": "indigo",
                        "position": "left",
                        "display": "Anti-Previous (1 day)",
                        "id": "ANTI_PREVIOUS_1D"
                      },
                      {
                        "color": "yellow",
                        "position": "right",
                        "display": "Previous (5 days)",
                        "id": "PREVIOUS_5D"
                      },
                      {
                        "color": "black",
                        "position": "top",
                        "display": "Excess Liquidity",
                        "id": "EXCESS"
                      },
                      {
                        "color": "red",
                        "position": "top",
                        "display": "Risk On",
                        "id": "RISKON"
                      },
                      {
                        "color": "#f8cd80 ",
                        "position": "bottom",
                        "display": "TREND 120D",
                        "id": "ZZTREND_120D"
                      },
                      {
                        "color": "#0049c1 ",
                        "position": "bottom",
                        "display": "MODE 60D",
                        "id": "ZZMODE_60D"
                      },
                      {
                        "color": "#c25de3 ",
                        "position": "bottom",
                        "display": "Anti-Adjusted Seasonality",
                        "id": "ANTI_ADJSEASONALITY"
                      },
                      {
                        "color": "#8ec54e ",
                        "position": "bottom",
                        "display": "Anti-TREND 60D",
                        "id": "ANTI_ZZTREND_60D"
                      },
                      {
                        "color": "#f49535 ",
                        "position": "bottom",
                        "display": "Seasonality",
                        "id": "SEASONALITY"
                      },
                      {
                        "color": "#3fa3e7 ",
                        "position": "bottom",
                        "display": "Anti-Seasonality",
                        "id": "ANTI_SEASONALITY"
                      },
                      {
                        "color": "transparent",
                        "position": "",
                        "id": "BLANK"
                      }
                    ],
        #strategy='signals_ANTI_EXCESS',
        #strategy='signals_EXCESS',
        #strategy='signals_SEASONALITY',
        #strategy='signals_ANTI_SEASONALITY',
        #strategy='signals_ADJSEASONALITY',
        #strategy='signals_ANTI_ADJSEASONALITY',
        #strategy='signals_RISKOFF',
        #strategy='signals_RISKON',
        #strategy='signals_PREVIOUS_1D',
        #strategy='signals_ANTI_PREVIOUS_1D',
        #strategy='signals_PREVIOUS_5D',
        #strategy='signals_ANTI_PREVIOUS_5D',
        #strategy='signals_ZZTREND',
        #strategy='signals_ANTI_ZZTREND',
        strategy='signals_ZZMODE_60D',
        #strategy='signals_ANTI_ZZMODE',
      )

class Service(object):
    global params
    
    def __init__(self):
        self.params=params
        #create futures object
        self.futures=Futures()
        
        #create portfolio object
        self.portfolio = Portfolio()
        
    def create(self, days, params=params):
        #create history using params for the portfolio for X days
        self.futures.create(self.portfolio, params, days)
    

if __name__=='__main__':
    start_time = time.time()
    gsm = Service()
    gsm.create(2)
    print(gsm.portfolio.history)
    print(gsm.portfolio.ranking(parents_only=True))
    print('Elapsed time: {} minutes. {}'\
          .format(round(((time.time() - start_time) / 60), 2), dt.now()))

'''
#portfolio.save()

filename='./nimbus/data/portfolios/50K_0_1516354961.json'
p2=Portfolio()
p2.load(filename)
p2.update(futures, futures.atr[system])
'''
