#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 12:44:23 2018

@author: hidemiasakura

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

#the input data
accounts = [
  {
    'id': "25k-chip",
    'display': "25K",
    'amount': 25000
  },
  {
    'id': "50k-chip",
    'display': "50K",
    'amount': 50000
  },
  {
    'id': "100k-chip",
    'display': "100K",
    'amount': 100000
  },
  {
    'id': "150k-chip",
    'display': "150K",
    'amount': 150000
  }
]

board_config= [
  {
    "color": "pink",
    "position": "left",
    "display": "Previous (1 day)",
    "id": "PREVIOUS_1_DAY"
  },
  {
    "color": "indigo",
    "position": "left",
    "display": "Anti-Previous (1 day)",
    "id": "ANTI_PREVIOUS_1_DAY"
  },
  {
    "color": "yellow",
    "position": "right",
    "display": "Previous (5 days)",
    "id": "PREVIOUS_5_DAYS"
  },
  {
    "color": "black",
    "position": "top",
    "display": "Risk Off",
    "id": "RISK_OFF"
  },
  {
    "color": "red",
    "position": "top",
    "display": "Risk On",
    "id": "RISK_ON"
  },
  {
    "color": "#f8cd80 ",
    "position": "bottom",
    "display": "Lowest Eq.",
    "id": "LOWEST_EQ"
  },
  {
    "color": "#0049c1 ",
    "position": "bottom",
    "display": "Highest Eq.",
    "id": "HIGHEST_EQ"
  },
  {
    "color": "#c25de3 ",
    "position": "bottom",
    "display": "Anti-HE",
    "id": "ANTI_HE"
  },
  {
    "color": "#8ec54e ",
    "position": "bottom",
    "display": "Anti-50",
    "id": "ANTI_50"
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
    "display": "Anti-Sea",
    "id": "ANTI_SEA"
  },
  {
    "color": "transparent",
    "position": "",
    "id": "BLANK"
  }
]

#fx=FX()
#fx.create()
start_time = time.time()
#create futures object
futures=Futures()

#create portfolio object
portfolio = Portfolio()

#params
system='signals_EXCESS'
params=  dict(
        account_value=500000,
        margin_percent=0.5,
        benchmark_sym='ES',
        correlation_cutoff=0.7,
        recreate_if_margin_call=False,
        increment=250,    
      )
#create history using params for the portfolio for X days
futures.create(portfolio, system, params, 30)

#update current betting selection
system='signals_RISKON'
portfolio.update(futures, futures.atr[system], **params)
print(portfolio.history)
print('Elapsed time: {} minutes. {}'\
      .format(round(((time.time() - start_time) / 60), 2), dt.now()))

#creates portfolio & target
#portfolio.create(fut, **params)

#update volatility target & margin calc
#params['account_value']=490000
#portfolio.update(fut, fut.atr.signals_Excess, **params)

#portfolio.account_value=portfolio.account_value/2
#portfolio.update_margin(fut, fut.atr.signals_EXCESS)
#portfolio.save()
'''
filename='./nimbus/data/portfolios/50K_0_1516354961.json'
p2=Portfolio()
p2.load(filename)
p2.update(futures, futures.atr[system])
'''
