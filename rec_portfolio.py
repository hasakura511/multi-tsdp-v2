#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 14:23:27 2018

inputs:
    account value
output:
    portfolio
    target
    max daily commissions
    
@author: hidemiasakura
"""

import pandas as pd
import random
import numpy

account_value = 500000
max_acct_margin =0.5*account_value

#order=['debt', 'index', 'energy', 'grain', 'metal','meat', 'currency']

def ib2csi(dic, ib_symlist):
    return dic.reset_index().set_index('ibsym').ix[ib_symlist].CSIsym.values.tolist()

def csi2ib(dic, csi_symlist):
    return dic.reset_index().set_index('CSIsym').ix[csi_symlist].ibsym.values.tolist()


#load dictionary file with margin info
dic=pd.read_csv('./data/systems/dictionary.csv', index_col=0)
ordered_dic=dic.sort_values(by='max_margin', ascending=True).copy()
order=ordered_dic.groupby(by='Group').first().sort_values(by='max_margin').index.tolist()
num_markets={}
portfolio=[]
total_margin=0
#while there is margin left, start adding markets
while total_margin<max_acct_margin and len(portfolio)<dic.shape[0]:
    print len(portfolio), len(portfolio)<=dic.shape[0]
    for g in order:
        group=ordered_dic[ordered_dic['Group']==g].copy()
        if ordered_dic.shape[0]<1:
            break
        #num_markets[g]=group.shape[0]
        #rounded_margins = group.max_margin.round(-3)
        rounded_margins=[x.round(-len(str(int(x)))+1) for x in group.max_margin]
        rounded_margins=pd.Series(index=group.index, data=rounded_margins)
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
            print m, markets, choice, margin, total_margin, total_margin<max_acct_margin
        #if total_margin>max_acct_margin:
        #    break
            
        if total_margin>max_acct_margin:
            break
        
print '\n\nportfolio selected'
print len(portfolio), portfolio, total_margin, account_value

while total_margin>max_acct_margin:
    print 'portfolio adjustment needed..'
    removed = portfolio.pop()
    total_margin=dic.ix[portfolio].max_margin.sum()
    print 'removed', removed, 'from portfolio'

print len(portfolio), ib2csi(dic, portfolio), total_margin, account_value

print
print
print 'adjusting target...'
increment=250
ordered_dic2=dic.sort_values(by='max_margin', ascending=True).ix[portfolio].copy()
#adjust the target
#load atr file
futuresDF=pd.read_csv('./data/futuresATR.csv', index_col=0)
csi_index=[x for x in dic.CSIsym if x in futuresDF.index]
futuresDF.index.name='CSIsym'
futuresDF2=futuresDF.ix[csi_index].copy()
futuresDF2.index=dic.index
futuresDF2.index.name='ibsym'
avg_atr=futuresDF2.ix[portfolio].usdATR.mean()
mult=1 if math.floor(avg_atr/250)<1 else math.floor(avg_atr/250)
target=250*mult
ordered_dic2['qty']=[int(math.ceil(x)) if x<1 else int(math.floor(x)) for x in target/futuresDF2.ix[portfolio].usdATR]
total_margin2=(ordered_dic2.qty*ordered_dic2.max_margin).sum()
print 'target', 'total margin'
print target, total_margin2
print ordered_dic2['qty'],'\n'


while total_margin2<max_acct_margin:
    target+=250
    ordered_dic2['qty']=[int(math.ceil(x)) if x<1 else int(math.floor(x)) for x in target/futuresDF2.ix[portfolio].usdATR]
    total_margin2=(ordered_dic2.qty*ordered_dic2.max_margin).sum()
    print target, total_margin2, ordered_dic2['qty']
    print
    
if total_margin2>max_acct_margin:
    target-=250
    ordered_dic2['qty']=[int(math.ceil(x)) if x<1 else int(math.floor(x)) for x in target/futuresDF2.ix[portfolio].usdATR]
    total_margin2=(ordered_dic2.qty*ordered_dic2.max_margin).sum()

print 'target',target, 'total margin',total_margin2, 'maxmargin',max_acct_margin, 'account value', account_value
print ordered_dic2['qty'].index.tolist()
print ordered_dic2['qty']
print 'max commissions',(ordered_dic2['qty']*2.7).sum()

#correlation - IB uses SPAN

corrDF = pd.read_html('./data/futures_3.html', index_col=0)[0]
cutoff=0.7
#signals=np.random.randint(-1,high=2, size=len(portfolio))
signals=np.random.choice([-1,1], len(portfolio))
positions =signals*ordered_dic2['qty']
positions.name = 'positions'


positions = positions[positions !=0]
csi_index=ib2csi(dic,positions.index)
positions.index=csi_index
corrDF2=corrDF.ix[csi_index][csi_index]
corrDF2=corrDF2[abs(corrDF2)>cutoff].fillna(0)
positions2=pd.concat([positions,dic.reset_index().set_index('CSIsym')['max_margin'].ix[csi_index]], axis=1)
positions2=positions2.sort_values(by=['max_margin'], ascending=False)


def is_long(n):
    if n>0:
        return True
    else:
        return False
exclude=[]
for sym in positions2.index:
    highcorr=corrDF2.ix[sym][corrDF2.ix[sym] !=0].index
    highcorr = highcorr.drop(sym)
    pos1=positions2.ix[sym].positions
    
    margin1=abs(positions2.ix[sym].positions)* positions2.ix[sym].max_margin
    positions2.set_value(sym, 'total_margin', abs(margin1))
    print '\n', sym, pos1, margin1, highcorr
    cdict={}
       
    for sym2 in highcorr:
        pos = positions2.ix[sym2].positions
        max_margin = positions2.ix[sym2].max_margin
        margin = abs(pos) * max_margin

        if not(is_long(pos1) and is_long(pos) or not is_long(pos1)\
                and not is_long(pos)):
        #    cdict[sym2]={'position':pos,'margin_impact':margin}
        #elif margin < margin1:
            cdict[sym2]={'position':pos,'margin':margin, 'corr': round(corrDF.ix[sym][sym2],2)}
            
        if sym2 in cdict and corrDF.ix[sym][sym2]>0 and sym2 not in exclude:
            print sym2, cdict[sym2]
            margin1-=margin
            print margin1
            #exclude.append(sym2)
            
    positions2.set_value(sym, 'hc', str(cdict))
    
    
    if sym not in exclude:
        positions2.set_value(sym, 'adj_margin', abs(margin1))
        exclude.append(sym)
        [exclude.append(s) for s in cdict.keys()]
        #print exclude
    else:
        positions2.set_value(sym, 'adj_margin', 0.0)
        
corrDF2=pd.concat([positions2,dic.reset_index().set_index('CSIsym')['max_margin'].ix[csi_index],corrDF2], axis=1)
corrDF2.index.name='CSIsym'
corrDF2.to_csv('./data/corrDF.csv', index=True)
total_margin=corrDF2.adj_margin.sum()

print 'Adjusted margin', total_margin,'Non-adjusted', corrDF2.total_margin.sum()


