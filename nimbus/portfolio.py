#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 14:23:27 2018

inputs:
    dictionary
    account value
    max account margin %
output:
    portfolio
    target
    max daily commissions
    
@author: hidemiasakura
"""
import sys
import pandas as pd
import numpy as np
import random
import numpy
import math
import json
import calendar
from datetime import datetime as dt
from os.path import isfile, join
import copy

def ib2csi(dic, ib_symlist):
    return dic.reset_index().set_index('ibsym').loc[ib_symlist].CSIsym.values.tolist()

def csi2ib(dic, csi_symlist):
    return dic.reset_index().set_index('CSIsym').loc[csi_symlist].ibsym.values.tolist()

def is_long(n):
    if n>0:
        return True
    else:
        return False
    
class Portfolio(object):
    
    def __init__(self):
        self.dic=pd.read_csv('./nimbus/data/system/dictionary.csv', index_col=0)\
                        .reset_index().set_index('CSIsym')
        '''
        account_data=  {
            'id': "5k-chip",
            'display': "5K",
            'amount': 5000,
          }
        '''
        self.starting_value=0
        self.account_value=0
        self.account_id=''
        self.display=''
        self.margin_percent=-0.5
        self.benchmark_sym='ES'
        self.max_acct_margin=0
        self.portfolio=[]
        self.qty={}
        self.atr = pd.DataFrame()
        self.atr_history={}
        self.correlations = pd.DataFrame()
        self.corr_history = {}
        self.total_margin=0
        self.target=0
        self.commission=2.7
        self.commissions=0
        self.max_commissions=0
        self.slippage=0
        self.portfolio_path='./nimbus/data/portfolios/'
        self.created=''
        self.updated=''
        self.increment=250
        self.recreate_if_margin_call = False
        self.correlation_cutoff=0.7
        self.history=pd.DataFrame()
        self.last_pnl=0
        self.last_date=None
        self.prev_date=None
        self.prev_selection='signals_OFF'
        self.last_selection='signals_OFF'
        
    def reduce(self):
        #removes last added from portfolio
        while self.total_margin>self.max_acct_margin:
            print('Portfolio adjustment needed..')
            removed = self.portfolio.pop()
            self.total_margin=self.dic.loc[self.portfolio].max_margin.sum()
            print('Removed', removed, 'from portfolio')
            
    def create(self, futures, signals, **kwargs):
        self.portfolio=[]
        self.total_margin=0
        self.account_value=kwargs.get('account_value', self.account_value)
        self.starting_value=self.account_value
        self.margin_percent=kwargs.get('margin_percent', self.margin_percent)
        self.max_acct_margin=self.margin_percent*self.account_value
        self.increment=kwargs.get('increment', self.increment)
        self.correlation_cutoff=kwargs.get('correlation_cutoff', self.correlation_cutoff)
        self.recreate_if_margin_call=kwargs.get('recreate_if_margin_call',\
                                                self.recreate_if_margin_call)
        self.benchmark_sym=kwargs.get('benchmark_sym', self.benchmark_sym)
        recreate=kwargs.get('recreate', False)
        
        ##sort by lowest margin markets
        ordered_dic=self.dic.sort_values(by='max_margin', ascending=True).copy()
        order=ordered_dic.groupby(by='Group').first()\
                        .sort_values(by='max_margin').index.tolist()
        print('\nCreating portfolio with account value {}'.format(self.account_value))
        print('kwargs', kwargs)
        #while there is margin left, start adding markets
        while self.total_margin<self.max_acct_margin\
                            and len(self.portfolio)<self.dic.shape[0]:
            #print(len(self.portfolio), len(self.portfolio)<=self.dic.shape[0])
            for g in order:
                group=ordered_dic[ordered_dic['Group']==g].copy()
                if ordered_dic.shape[0]<1:
                    break
                #num_markets[g]=group.shape[0]
                #rounded_margins = group.max_margin.round(-3)
                
                ##round the margins so we can group them
                rounded_margins=[x.round(-len(str(int(x)))+1) for x \
                                         in group.max_margin]
                rounded_margins=pd.Series(index=group.index, data=rounded_margins)
                #print g, rounded_margins
                
                ##start adding markets to portfolio by lowest margin
                #for m in rounded_margins.unique()[0]:
                if len(rounded_margins.unique())>0:
                    m=rounded_margins.unique()[0]
                    markets=rounded_margins.index[rounded_margins==m].values       
                    choice=random.choice(markets)
                    self.portfolio.append(choice)
                    margin=self.dic.loc[choice].max_margin
                    self.total_margin +=margin
                    ordered_dic=ordered_dic.drop([choice])
                    #print(m, markets, choice, margin, self.total_margin,\
                    #            self.total_margin<self.max_acct_margin)
                #if total_margin>max_acct_margin:
                #    break
                    
                if self.total_margin>self.max_acct_margin:
                    break
                
        #print('Portfolio selected:')
        #print(len(self.portfolio), self.portfolio, self.total_margin,\
        #        self.account_value)
        
        while self.total_margin>self.max_acct_margin:
            self.reduce()
            
        print('Portfolio successfully created..')
        print(len(self.portfolio), 'markets in portfolio')
        print(self.portfolio)
        if len(self.portfolio)>0:
            
            print('Total Margin', self.total_margin, 'account value',\
                  self.account_value)
            print('Total Margin/Account',\
                  round(self.total_margin/self.account_value,3),\
                  '< margin target', self.margin_percent)
        else:
            #account value insufficient. reset qty values
            print('Insufficient funds to create a futures portfolio!')
            self.qty={}
            
        if recreate:
            self.updated=int(calendar.timegm(dt.utcnow().utctimetuple()))
        else:
            self.created=self.updated=int(calendar.timegm(dt.utcnow().utctimetuple()))
        
        self.account_id=str(self.account_value)[:-3]+'K_0_'+str(self.created)
        self.display=str(self.account_value)[:-3]+'K'
        
        self.update_target(futures, signals)
        self.update_margin(futures, signals)
        self.atr_history[futures.last_date.strftime('%Y%m%d')]=self.atr.copy()
        self.history.set_value(futures.last_date, 'benchmark_value', self.starting_value)
        self.history.set_value(futures.last_date, 'benchmark_sym', self.benchmark_sym)

        self.history.index.name='Date'
        self.last_date=futures.last_date
        return self
            
    def recreate(self, futures, signals):
        print('\nRecreating Portfolio..')
        self.create(recreate=True)
        if len(self.portfolio)>0:
            self.update_target(futures, signals)
            self.update_margin(futures, signals)
        return self
        
    
    def update_target(self, futures, signals, **kwargs):
        '''
        adjust the target first by setting it to the average atr of the 
        portfolio then adjusting up/down
        '''
        self.account_value=kwargs.get('account_value', self.account_value)
        self.margin_percent=kwargs.get('margin_percent', self.margin_percent)
        self.increment=kwargs.get('increment', self.increment)
        self.max_acct_margin=self.margin_percent*self.account_value
        #signals=kwargs.get('signals', None)
        
        print('\nUpdating volatility target...')
        if len(self.portfolio)<1:
            print('create() first!')
            return
        
        def get_qty(target, portfolio, df):
            return [int(math.ceil(x)) if x<1 else int(math.floor(x))\
                    for x in target/df.loc[portfolio].usdATR]
            
        ordered_dic2=self.dic.sort_values(by='max_margin',\
                                          ascending=True).loc[self.portfolio].copy()
        #load atr file
        futuresDF=futures.atr.copy()
        markets=[x for x in self.dic.index if x in futuresDF.index]
        futuresDF2=futuresDF.loc[markets].copy()
        self.atr=futuresDF2
        
        #try to set the target to the average atr of the portfolio
        avg_atr=futuresDF2.loc[self.portfolio].usdATR.mean()
        mult=avg_atr/self.increment
        mult=1 if math.floor(mult)<1 else math.floor(mult)
        self.target=self.increment*mult
        ordered_dic2['qty']=get_qty(self.target, self.portfolio, futuresDF2)

        total_margin2=round((ordered_dic2.qty*ordered_dic2.max_margin).sum())
        print('target',self.target, 'avg_atr', avg_atr, 'total margin',total_margin2)
        print(ordered_dic2['qty'].to_dict(),'\n')
        
        #increase the target while there is enough margin
        while total_margin2<self.max_acct_margin:
            self.target+=self.increment
            ordered_dic2['qty']=get_qty(self.target, self.portfolio, futuresDF2)
            total_margin2=(ordered_dic2.qty*ordered_dic2.max_margin).sum()
            #print(self.target, total_margin2, ordered_dic2['qty'].to_dict())
            #print()
            
        #reduce if too high   
        while total_margin2>self.max_acct_margin and not self.target<=self.increment:
            self.target-=self.increment
            ordered_dic2['qty']=get_qty(self.target, self.portfolio, futuresDF2)
            total_margin2=(ordered_dic2.qty*ordered_dic2.max_margin).sum()
        
        #if total margin is still over, try setting all quantities to 1
        if total_margin2>self.max_acct_margin and self.target==self.increment:
            ordered_dic2['qty']=1
            total_margin2=(ordered_dic2.qty*ordered_dic2.max_margin).sum()
        
        #if total margin is still over, recreate portfolio
        if total_margin2>self.max_acct_margin:
            print('total_margin',total_margin2,'>max margin',self.max_acct_margin,\
                  'target',self.target,'cannot be reduced further than the increment of',\
                  self.increment)
            
            if self.recreate_if_margin_call:
                self.recreate(futures, signals)
                
            else:
                if len(self.portfolio)>=2:
                    #case two or more, returns 1 if 2
                    self.reduce()   
                    self.update_target(futures, signals)
                else:
                    #case if 1, reduce will return 0
                    self.recreate(futures, signals)
            
            return
        
        #update values
        self.total_margin=total_margin2
        self.max_commissions=(ordered_dic2['qty']*self.commission).sum()
        self.qty=ordered_dic2['qty'].to_dict()
        print('target finalized..')
        print(self.qty)
        print('target',self.target, 'total margin',self.total_margin,\
                'maxmargin',self.max_acct_margin, 'account value', self.account_value)
        #print ordered_dic2['qty'].index.tolist()
        print('max commissions',self.max_commissions)
        self.atr['target']=self.target
        self.atr['qty']=pd.Series(data=self.qty, index=futures.atr.index).fillna(0)
        self.history.set_value(futures.last_date, 'target', self.target)
        return self
               
    def update_margin(self, futures, signals, **kwargs): 
        #correlation - IB uses SPAN provided by the exchanges.
        #this is a lot more simpler
        if len(self.atr)<1 or len(self.portfolio)<1:
            print('update_target() first!')
            return
        self.account_value=kwargs.get('account_value', self.account_value)
        self.margin_percent=kwargs.get('margin_percent', self.margin_percent)
        self.increment=kwargs.get('increment', self.increment)
        self.max_acct_margin=self.margin_percent*self.account_value
        self.correlation_cutoff=kwargs.get('correlation_cutoff', self.correlation_cutoff)
        self.correlations =futures.correlations.copy()
        
        #check if index mismatch
        #print('\nUpdating Total Margin with account value of', self.account_value)
        #print(self.atr.index.tolist(), signals.index.tolist())
        if self.atr.index.tolist() != signals.index.tolist():
            print('Warning! mismatch in index')
            print(self.atr.index.tolist(), signals.index.tolist())
        
        qty=self.atr.qty
        positions =signals*qty
        positions.name = 'positions'
        
        #get non-zero positions
        positions = positions[positions !=0]
        #calc correlations for non-zero positions, prepare dataframe
        corrDF2=self.correlations.loc[positions.index][positions.index]
        corrDF2=corrDF2[abs(corrDF2)>self.correlation_cutoff].fillna(0)
        positions2=pd.concat([positions,\
                              self.dic['max_margin'].loc[positions.index]], axis=1)
        positions2=positions2.sort_values(by=['max_margin'], ascending=False)
        positions2['selection']=signals.name
        positions2['last_date']=futures.last_date

        #exclusion list for margins already added
        exclude=[]
        #print(positions2.index)
        for sym in positions2.index:
            highcorr=corrDF2.loc[sym][corrDF2.loc[sym] !=0].index
            #print(corrDF2)
            #print(highcorr)
            if len(highcorr)>0:
                highcorr = highcorr.drop(sym)
            pos1=positions2.loc[sym].positions
            
            margin1=abs(positions2.loc[sym].positions)* positions2.loc[sym].max_margin
            positions2.set_value(sym, 'total_margin', abs(margin1))
            #print('\n', sym, pos1, margin1, highcorr)
            cdict={}
               
            for sym2 in highcorr:
                pos = positions2.loc[sym2].positions
                max_margin = positions2.loc[sym2].max_margin
                margin = abs(pos) * max_margin
        
                if not(is_long(pos1) and is_long(pos) or not is_long(pos1)\
                        and not is_long(pos)):
                #    cdict[sym2]={'position':pos,'margin_impact':margin}
                #elif margin < margin1:
                    cdict[sym2]={'position':pos,'margin':margin,\
                         'corr': round(self.correlations.loc[sym][sym2],2)}
                    
                if sym2 in cdict and self.correlations.loc[sym][sym2]>0\
                            and sym2 not in exclude:
                    #print(sym2, cdict[sym2])
                    margin1-=margin
                    #print(margin1)
                    #exclude.append(sym2)
                    
            positions2.set_value(sym, 'hc', str(cdict))
            
            
            if sym not in exclude:
                positions2.set_value(sym, 'adj_margin', abs(margin1))
                exclude.append(sym)
                [exclude.append(s) for s in cdict.keys()]
                #print exclude
            else:
                positions2.set_value(sym, 'adj_margin', 0.0)
        
        
        corrDF2=pd.concat([positions2,self.dic['max_margin'].loc[positions.index],\
                           corrDF2], axis=1)
        corrDF2.index.name='CSIsym'
        #corrDF2.to_csv('./data/corrDF.csv', index=True)
        
        self.correlations = corrDF2
        self.corr_history[futures.last_date.strftime('%Y%m%d')]=corrDF2.copy()
        
        if signals.name not in self.atr:
            print('Adding', signals.name, 'to self.atr')
            self.atr[signals.name]=signals
        self.atr['positions']= signals*self.atr.qty
        self.atr['selection']= signals.name
        self.last_selection = signals.name
        self.history.set_value(futures.last_date, 'last_selection', self.last_selection)
        
        self.total_margin=round(corrDF2.adj_margin.sum())
        
        if len(self.atr_history)==0:
            #calc commissions from off
            self.commissions=self.atr.qty.sum()*self.commission
            self.account_value=self.starting_value-self.commissions
            
        else:
            #need up update account value again after current qty, pos calculated
            prev_positions=self.atr_history[self.prev_date.strftime('%Y%m%d')].positions
            contracts_traded=self.atr.positions-prev_positions
            #print(self.prev_date, prev_positions, futures.last_date,\
            #      self.atr.positions)
            #print(contracts_traded, abs(contracts_traded).sum())
            #print('contracts_traded',contracts_traded)
            self.commissions=abs(contracts_traded).sum()*self.commission
            self.account_value=self.history.account_value[-2]+\
                                self.last_pnl-self.commissions
            
        self.history.set_value(futures.last_date, 'account_value',\
                               self.account_value)
        self.display=str(self.account_value)[:-3]+'K'
        self.history.set_value(futures.last_date, 'prev_selection', self.prev_selection)
        self.history.set_value(futures.last_date, 'pnl', self.last_pnl)
        self.history.set_value(futures.last_date, 'commissions', self.commissions)
        #no slippage calc for csidata
        self.history.set_value(futures.last_date, 'slippage', self.slippage) 
        
        print('account_value',self.account_value,'Adjusted margin',\
                      self.total_margin,'Non-adjusted',\
                      round(self.correlations.total_margin.sum()))
        
        #adjust target if self.total_margin>self.max_acct_margin
        self.max_acct_margin=self.account_value*self.margin_percent
        #self.total_margin=self.max_acct_margin*2
        if self.total_margin>self.max_acct_margin:
            print('total margin',self.total_margin,'> max margin',\
                  self.max_acct_margin)
            self.update_target(futures, signals)
        self.history.set_value(futures.last_date, 'total_margin', self.total_margin)
        return self            
    
    def update_benchmark(self, futures):
        #replace all if sym has changed.

        benchmark_pc=futures.history[futures.history.index==self.benchmark_sym][\
                                                ['LastPctChg','last_date']]
        benchmark_pc=benchmark_pc.reset_index().set_index('last_date')
        #missing_dates = benchmark_pc.index>self.last_date
        #missing_dates=benchmark_pc.loc[missing_dates].index
        #num_dates_missing=len(missing_dates)
        history=pd.concat([self.history, benchmark_pc], axis=1)
        '''
        if num_dates_missing==1:
            benchmark_value=history.iloc[-2].benchmark_value*\
                                        (history.iloc[-1].LastPctChg+1)
            self.history.set_value(futures.last_date, 'benchmark_value', benchmark_value)
            self.history.set_value(futures.last_date, 'benchmark_sym', self.benchmark_sym)
            print(history)
        
        #if fix history if there's nans in the concat
        if num_dates_missing>1:
        
            print('more than one missing dates found,',missing_dates,\
                  'fixing benchmark history..')
        '''
        #update the benchmark values--change in sym recreates history
        history=pd.concat([self.history, benchmark_pc], axis=1)
        #print(history)
        history=history.copy()
        cp=(history.LastPctChg+1).cumprod()
        cp.iloc[0]=1
        history['benchmark_value']=history.iloc[0].benchmark_value*cp
        history['benchmark_sym']=self.benchmark_sym
        self.history.benchmark_value=history.benchmark_value
        self.history.benchmark_sym=history.benchmark_sym
        #print(history)
            
        
    def update_pnl(self, futures, signals):
        #create pnl rows in the current atr file

        
        prev_date=self.last_date.strftime('%Y%m%d')
        #last_date=futures.last_date.strftime('%Y%m%d')
        prev=self.atr_history[prev_date]
        last=futures.atr.copy()
        print(prev_date, 'atr_history equals self.atr check:', prev.equals(self.atr))
                    
        _signals=[x for x in prev.columns if 'signals_' in x]
        for s in _signals:
            pnl = prev.qty*prev[s]*prev.contract_value*last.LastPctChg
            #print(s, pnl, pnl.sum())
            futures.atr[s.replace('signals_','pnl_')]=pnl
             
            if s==signals.name:
                self.last_pnl=pnl.sum()
                self.history.set_value(futures.last_date, 'pnl', self.last_pnl)
                self.account_value=self.account_value+self.last_pnl
                
                
        self.update_benchmark(futures)

        
        #sys.exit()
        #self.account_value=kwargs.get('account_value', self.account_value)
        #self.display=str(self.account_value)[:-3]+'K'
        pass
        
    def update(self, futures, signals, **kwargs):
        print('Performing portfolio update..')
        print('kwargs', kwargs)
        self.benchmark_sym=kwargs.get('benchmark_sym', self.benchmark_sym)
        
        #update history
        if futures.last_date>self.last_date:
            print('Updating portfolio for..', futures.last_date,\
                  'last_date', self.last_date)
            self.prev_date=self.last_date
            self.prev_selection=self.last_selection
            print('Updating portfolio pnl..')
            #account value needs to be updated first here
            self.update_pnl(futures, signals)
            #update qty, this rewrites the self.atr
            print('Updating portfolio volatility target..')
            self.update_target(futures, signals, **kwargs)
            
            
        
        #update margins >= incase signal choice is changed.
        if futures.last_date>=self.last_date:
            #check and update total margins, doesn't rewrite self.atr
            print('Updating portfolio margin..')
            self.update_margin(futures, signals, **kwargs)
            self.atr_history[futures.last_date.strftime('%Y%m%d')]=self.atr
        
        #update history
        if futures.last_date>self.last_date:
            
            #only create() updates self.last_date=futures.last_date
            
            #self.update_history(futures)
            self.last_date=futures.last_date
            self.atr_history[futures.last_date.strftime('%Y%m%d')]=self.atr
        
            
        self.updated=int(calendar.timegm(dt.utcnow().utctimetuple()))
        
        return self
    
    def save(self):
        if len(self.atr)<1:
            print('update() first!')
            return

        
        export_dict={
            'account_id': str(self.account_id),
            'display': str(self.display),
            'starting_value': float(self.starting_value),
            'account_value': float(self.account_value),
            'portfolio':list(self.portfolio),
            'qty':str(self.qty),
            'target':int(self.target),
            'total_margin':float(self.total_margin),
            'max_acct_margin':float(self.max_acct_margin),
            'margin_percent':float(self.margin_percent),
            'commission':float(self.commission),
            'commissions':float(self.commissions),
            'slippage':float(self.slippage),
            'max_commissions':float(self.max_commissions),
            'benchmark_sym':str(self.benchmark_sym),
            'increment':int(self.increment),
            'recreate_if_margin_call':bool(self.recreate_if_margin_call),
            'correlation_cutoff':float(self.correlation_cutoff),
            'created':int(self.created),
            'updated':int(self.updated),
            'prev_date':self.prev_date.strftime('%Y%m%d'),
            'last_date':self.last_date.strftime('%Y%m%d'),
            'prev_selection':str(self.prev_selection),
            'last_selection':str(self.last_selection),
            'last_pnl':float(self.last_pnl),
            'history':self.history.to_json(),
            'atr':self.atr.to_json(),
            'atr_prev':self.atr_history[self.prev_date.strftime('%Y%m%d')].to_json(),
          }

        filename=self.portfolio_path+self.account_id+'.json'
        with open(filename, 'w') as f:
            json.dump(export_dict, f)
        print('\nSaved', filename)
        
        
    def load(self, filename):
        if isfile(filename):
            with open(filename, 'r') as f:
                portfolio_data = json.load(f)
        else:
            print(filename, 'not found')
            return
        
        self.account_id = portfolio_data['account_id']
        self.display = portfolio_data['display']
        self.starting_value = portfolio_data['starting_value']
        self.account_value = portfolio_data['account_value']
        self.portfolio = portfolio_data['portfolio']
        self.qty = eval(portfolio_data['qty'])
        self.target = portfolio_data['target']
        self.total_margin = portfolio_data['total_margin']
        self.max_acct_margin=portfolio_data['max_acct_margin']
        self.margin_percent=portfolio_data['margin_percent']
        self.commission=portfolio_data['commission']
        self.commissions=portfolio_data['commissions']
        self.slippage=portfolio_data['slippage']
        self.max_commissions = portfolio_data['max_commissions']
        self.benchmark_sym=portfolio_data['benchmark_sym']
        self.increment=portfolio_data['increment']
        self.recreate_if_margin_call = portfolio_data['recreate_if_margin_call']
        self.correlation_cutoff= portfolio_data['correlation_cutoff']
        self.created = portfolio_data['created']
        self.updated = portfolio_data['updated']
        self.prev_date = dt.strptime(portfolio_data['prev_date'], '%Y%m%d')
        self.last_date = dt.strptime(portfolio_data['last_date'], '%Y%m%d')
        self.prev_selection = portfolio_data['prev_selection']
        self.last_selection = portfolio_data['last_selection']
        self.last_pnl = portfolio_data['last_pnl']
        self.history = pd.read_json(portfolio_data['history'])
        self.atr = pd.read_json(portfolio_data['atr'])
        self.atr_history[portfolio_data['prev_date']] =\
                        pd.read_json(portfolio_data['atr_prev'])
        
        self.history.index.name='Date'
        
    def copy(self):
        return copy.deepcopy(self)
                    

        
    
