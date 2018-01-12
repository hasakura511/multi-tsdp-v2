#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 14:23:27 2018

inputs:
    account value
output:
    portfolio
    target
    
@author: hidemiasakura
"""

import pandas as pd
import random
account_value = 500000
order=['debt', 'index', 'energy', 'grain', 'metal','meat', 'currency']

#load dictionary file with margin info
dic=pd.read_csv('./data/systems/dictionary.csv', index_col=0)
ordered_dic=dic.sort_values(by='max_margin', ascending=True).copy()
num_markets={}
portfolio=[]
total_margin=0
#while there is margin left
while total_margin<account_value/2 and len(portfolio)<dic.shape[0]:
    print len(portfolio), len(portfolio)<=dic.shape[0]
    for g in order:
        group=ordered_dic[ordered_dic['Group']==g].copy()
        if ordered_dic.shape[0]<1:
            break
        #num_markets[g]=group.shape[0]
        rounded_margins = group.max_margin.round(-3)
        print g, rounded_margins
        #for m in rounded_margins.unique()[0]:
        if len(rounded_margins.unique())>0:
            m=rounded_margins.unique()[0]
            markets=rounded_margins.index[rounded_margins==m].values       
            choice=random.choice(markets)
            portfolio.append(choice)
            margin=dic.ix[choice].max_margin
            total_margin +=margin
            ordered_dic=ordered_dic.drop([choice])
            print m, markets, choice, margin, total_margin, total_margin<account_value/2
        #if total_margin>account_value/2:
        #    break
            
        if total_margin>account_value/2:
            break
        
    
print len(portfolio), portfolio