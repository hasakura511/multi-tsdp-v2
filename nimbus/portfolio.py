#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 14:23:27 2018

inputs:
    dictionary
    futures data
    signals
    params
output:
    portfolio
    target, qty
    atr master df
    margins
    history
    commissions
    
@author: hidemiasakura
"""
import sys
import pandas as pd
import numpy as np
import random
import math
import json
import calendar
from nimbus.board import Board
from nimbus.process.transform import to_signals, is_int
from datetime import datetime as dt
from os.path import isfile, join
import copy
#import warnings
#warnings.simplefilter('error')

def get_timestamp():
    return int(calendar.timegm(dt.utcnow().utctimetuple()))

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
        self.positions={}
        self.atr = pd.DataFrame()
        self.atr_dict={}
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
        self.pnl_pct=0.0
        self.pnl_cumpct=0.0
        self.last_date=None
        self.prev_date=None
        self.prev_selection='signals_OFF'
        self.last_selection='signals_OFF'
        self.comment=''
        self.insufficient_funds=False
        self.tier=0
        self.board_config={}
        self.board_dict={}
        self.last_signals=pd.Series()
        self.signals=''
        
    def reduce(self, recreate=False):
        #removes last added from portfolio
        removed_list=[]
        while self.total_margin>self.max_acct_margin:
            print('Portfolio adjustment needed..')
            removed = self.portfolio.pop()
            self.total_margin=self.dic.loc[self.portfolio].max_margin.sum()
            print('Removed', removed, 'from portfolio')
            removed_list.append(removed)
        if recreate == False:
            self.comment+='Removed {} from portfolio. '.format(str(removed_list))
        #self.history.set_value(futures.last_date, 'comment', self.comment)
            
    def append_child_signals(self, futures):
        #updates self.atr with new one from futures and adds child signals
        #should be done in the beginning of every iteration.
        self.atr=futures.atr.copy()
        if len(self.board_config)<1:
            message='board_config empty. set the board config.'
            print(message)
            sys.exit(message)
            
        board=Board()
        board.create(self.board_config)
        self.board_dict=board.board_dict
        for number in self.board_dict:
            self.atr['signals_'+str(number)]=to_signals(
                                    self.atr[self.board_dict[number]].sum(axis=1)
                                    )
  
                            
    def generate_portfolio(self):
        '''generates portfolio from scratch'''
        ##sort by lowest margin markets
        ordered_dic=self.dic.sort_values(by='max_margin', ascending=True).copy()
        order=ordered_dic.groupby(by='Group').first()\
                        .sort_values(by='max_margin').index.tolist()
        print('\nCreating portfolio with account value {}'.format(self.account_value))
        #print('kwargs', kwargs)
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
            self.reduce(recreate=True)
            
        print('Portfolio successfully created..')        
            
    def create(self, futures, **kwargs):
        self.portfolio=[]
        self.total_margin=0
        self.account_value=kwargs.get('account_value', self.account_value)
        self.board_config=kwargs.get('board_config', {})
        self.last_selection = kwargs.get('strategy', self.last_selection)
        self.margin_percent=kwargs.get('margin_percent', self.margin_percent)
        self.max_acct_margin=self.margin_percent*self.account_value
        self.increment=kwargs.get('increment', self.increment)
        self.correlation_cutoff=kwargs.get('correlation_cutoff', self.correlation_cutoff)
        self.recreate_if_margin_call=kwargs.get('recreate_if_margin_call',\
                                                self.recreate_if_margin_call)
        self.benchmark_sym=kwargs.get('benchmark_sym', self.benchmark_sym)
        recreate=kwargs.get('recreate', False)
        
        if len(self.history)==0:
            #first time run
            self.starting_value=self.account_value
            self.append_child_signals(futures)   
            self.last_signals=self.atr[self.last_selection]
            
        
        self.generate_portfolio()
 
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
            comment='Insufficient funds to create a futures portfolio! '
            print(comment)
            self.qty={}
            self.positions={}
            self.comment+=comment
            self.insufficient_funds=True
            self.last_selection='signals_OFF'
            
            self.history.set_value(futures.last_date, 'target', self.target)
            self.history.set_value(futures.last_date, 'qty', str(self.qty))
            self.history.set_value(futures.last_date, 'positions', str(self.positions))
            self.history.set_value(futures.last_date, 'last_selection', self.last_selection)
            self.history.set_value(futures.last_date, 'account_value',\
                                   self.account_value)
            self.history.set_value(futures.last_date, 'prev_selection', self.prev_selection)
            self.history.set_value(futures.last_date, 'pnl_ACCOUNT', self.last_pnl)
            if len(self.history)>=2:
                self.pnl_pct=self.account_value/self.history.account_value[-2]-1
                self.pnl_cumpct=self.account_value/self.starting_value-1
            #else:
            #    self.pnl_pct=0.0
            #    self.pnl_cumpct=0.0
            self.history.set_value(futures.last_date, 'account_pnl_pct', self.pnl_pct)
            self.history.set_value(futures.last_date, 'account_pnl_cumpct', self.pnl_cumpct)
            self.history.set_value(futures.last_date, 'commissions', self.commissions)
            self.history.set_value(futures.last_date, 'slippage', self.slippage)
            self.history.set_value(futures.last_date, 'total_margin', self.total_margin)
            self.atr['qty']=pd.Series(data=self.qty, index=futures.atr.index).fillna(0)
            self.atr['positions']= self.last_signals*self.atr.qty
            self.atr['selection']= self.last_signals.name

            
        if len(str(self.account_value))<=3:
            self.display=str(self.account_value/1000)+'K'
        else:
            self.display=str(self.account_value)[:-3]+'K'
            
        self.account_id=self.display+'_'+str(self.tier)+'_'+str(self.created)

        
        if not self.insufficient_funds:
            print('Sufficient funds in portfolio')
            self.update_target(futures)
            self.update_margin(futures)
        
        if recreate:
            self.updated=get_timestamp()
        else:
            #first time run
            self.created=self.updated=get_timestamp()
            self.history.set_value(futures.last_date, 'benchmark_value', self.starting_value)
            self.history.set_value(futures.last_date, 'benchmark_sym', self.benchmark_sym)
            self.history.set_value(futures.last_date, 'comment', self.comment)
                        
            _signals=[x for x in self.atr.columns if 'signals_' in x]
            for s in _signals:
                column_name=s.replace('signals_','pnl_')
                self.atr[column_name]=0.0
                self.history.set_value(futures.last_date, column_name, 0.0)
                
            self.atr_dict[futures.last_date.strftime('%Y%m%d')]=self.atr.copy()
            self.history.index.name='Date'
            self.last_date=futures.last_date
            
        return self
            
    def recreate(self, futures):
        print('\nRecreating Portfolio..')
        self.create(futures, recreate=True)
        if len(self.portfolio)>0:
            self.update_target(futures)
            self.update_margin(futures)
        return self
        
    
    def update_target(self, futures):
        '''
        adjust the target first by setting it to the average atr of the 
        portfolio then adjusting up/down
        '''
        #self.account_value=kwargs.get('account_value', self.account_value)
        #self.margin_percent=kwargs.get('margin_percent', self.margin_percent)
        #self.increment=kwargs.get('increment', self.increment)
        self.max_acct_margin=self.margin_percent*self.account_value
        #signals=kwargs.get('signals', None)
        
        
        if len(self.portfolio)<1:
            print('Empty portfolio. Recreating portfolio..')
            self.recreate(futures)
            return
        
        print('\nUpdating volatility target...')
        def get_qty(target, portfolio, df):
            return [int(math.ceil(x)) if x<1 else int(math.floor(x))\
                    for x in target/df.loc[portfolio].usdATR]
            
        ordered_dic2=self.dic.sort_values(by='max_margin',\
                                          ascending=True).loc[self.portfolio].copy()
        
        #markets=[x for x in self.dic.index if x in futuresDF.index]
        #futuresDF2=futuresDF.loc[markets].copy()
        #self.atr=futuresDF2
        
        #try to set the target to the average atr of the portfolio
        avg_atr=self.atr.loc[self.portfolio].usdATR.mean()
        mult=avg_atr/self.increment
        mult=1 if math.floor(mult)<1 else math.floor(mult)
        self.target=self.increment*mult
        ordered_dic2['qty']=get_qty(self.target, self.portfolio, self.atr)

        total_margin2=round((ordered_dic2.qty*ordered_dic2.max_margin).sum())
        print('target',self.target, 'avg_atr', avg_atr, 'total margin',total_margin2)
        #print(ordered_dic2['qty'].to_dict(),'\n')
        
        #increase the target while there is enough margin
        while total_margin2<self.max_acct_margin:
            self.target+=self.increment
            ordered_dic2['qty']=get_qty(self.target, self.portfolio, self.atr)
            total_margin2=(ordered_dic2.qty*ordered_dic2.max_margin).sum()
            #print(self.target, total_margin2, ordered_dic2['qty'].to_dict())
            #print()
            
        #reduce if too high   
        while total_margin2>self.max_acct_margin and not self.target<=self.increment:
            self.target-=self.increment
            ordered_dic2['qty']=get_qty(self.target, self.portfolio, self.atr)
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
                self.recreate(futures)
                self.comment+='recreate_if_margin_call set to True. Recreating portfolio. '
            else:
                if len(self.portfolio)>0:
                    #case 1 or more, if 1 it will go to recreate
                    self.reduce()   
                    self.update_target(futures)
                else:
                    '''
                    case if 0, recreate to see if there's 
                    any markets that meet margin requirements
                    '''
                    self.recreate(futures)
                    #self.comment+='1 market in portfolio. cannot reduce. recreating portfolio. '
            
            return
        
        #update values
        self.total_margin=total_margin2
        self.max_commissions=(ordered_dic2['qty']*self.commission).sum()
        if (pd.Series(data=self.qty)-ordered_dic2['qty']).sum() != 0:
            self.comment+='Adjusted qty '
            
        self.qty=ordered_dic2['qty'].to_dict()
        
        #print('target finalized..')
        print(self.qty)
        print('target',self.target, 'total margin',self.total_margin,\
                'maxmargin',self.max_acct_margin, 'account value', self.account_value)
        #print ordered_dic2['qty'].index.tolist()
        print('max commissions',self.max_commissions)
        self.atr['target']=self.target
        self.atr['qty']=pd.Series(data=self.qty, index=futures.atr.index).fillna(0)
        self.history.set_value(futures.last_date, 'target', self.target)
        self.history.set_value(futures.last_date, 'qty', str(self.qty))
        return self
    
    def calc_total_margin(self, futures):
        corrDF =futures.correlations.copy()
        positions =self.last_signals*self.atr.qty
        positions.name = 'positions'
        
        #get non-zero positions
        positions = positions[positions !=0]
        #calc correlations for non-zero positions, prepare dataframe
        corrDF=corrDF.loc[positions.index][positions.index]
        corrDF2=corrDF[abs(corrDF)>self.correlation_cutoff].fillna(0).copy()
        positions2=pd.concat([positions,\
                              self.dic['max_margin'].loc[positions.index]], axis=1)
        positions2=positions2.sort_values(by=['max_margin'], ascending=False)
        positions2['selection']=self.last_selection
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
                         'corr': round(futures.correlations.loc[sym][sym2],2)}
                    
                if sym2 in cdict and futures.correlations.loc[sym][sym2]>0\
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
                           corrDF], axis=1)
        corrDF2.index.name='CSIsym'
        #corrDF2.to_csv('./data/corrDF.csv', index=True)
        
        self.correlations = corrDF2
        self.corr_history[futures.last_date.strftime('%Y%m%d')]=corrDF2.copy()
        self.total_margin=round(corrDF2.adj_margin.sum())
        
    def update_margin(self, futures): 
        '''
        updates total margin, positions with the signals and futures correlations
        '''
        if len(self.atr)<1 or len(self.portfolio)<1:
            print('update_target() first!')
            return
        self.max_acct_margin=self.margin_percent*self.account_value
        self.last_signals=self.atr[self.last_selection]
        
        
        #check if index mismatch
        #print('\nUpdating Total Margin with account value of', self.account_value)
        #print(self.atr.index.tolist(), signals.index.tolist())
        #if self.atr.index.tolist() != self.last_signals.index.tolist():
        #    print('Warning! mismatch in index')
        #    print(self.atr.index.tolist(), signals.index.tolist())
        
        self.calc_total_margin(futures)
        
        #if signals.name not in self.atr:
        #    print('Adding', signals.name, 'to self.atr')
        #    self.atr[signals.name]=signals
        self.atr['positions']= self.last_signals*self.atr.qty
        self.positions=self.atr.positions[self.atr.positions != 0].astype(int).to_dict()
        self.atr['selection']= self.last_selection
        #self.last_selection = signals.name
        self.history.set_value(futures.last_date, 'positions', str(self.positions))
        self.history.set_value(futures.last_date, 'last_selection', self.last_selection)
        
        
        if len(self.atr_dict)==0:
            #calc commissions from off
            self.commissions=self.atr.qty.sum()*self.commission
            self.account_value=self.starting_value-self.commissions
            #self.history.set_value(futures.last_date, 'pnl_pct', self.pnl_pct)
            #self.history.set_value(futures.last_date, 'pnl_cumpct', self.pnl_cumpct)
            
        else:
            #need up update account value again after current qty, pos calculated
            prev_positions=self.atr_dict[self.prev_date.strftime('%Y%m%d')].positions
            contracts_traded=self.atr.positions-prev_positions
            #print(self.prev_date, prev_positions, futures.last_date,\
            #      self.atr.positions)
            #print(contracts_traded, abs(contracts_traded).sum())
            #print('contracts_traded',contracts_traded)
            self.commissions=abs(contracts_traded).sum()*self.commission
            self.account_value=self.history.account_value[-2]+\
                                self.last_pnl-self.commissions
            if len(self.atr_dict)==1:
                self.pnl_pct=self.account_value/self.starting_value-1
            else:
                self.pnl_pct=self.account_value/self.history.account_value[-2]-1
            self.pnl_cumpct=self.account_value/self.starting_value-1
        

        self.history.set_value(futures.last_date, 'account_value',\
                               self.account_value)
        self.display=str(self.account_value)[:-3]+'K'
        self.history.set_value(futures.last_date, 'prev_selection', self.prev_selection)
        self.history.set_value(futures.last_date, 'pnl_ACCOUNT', self.last_pnl)
        self.history.set_value(futures.last_date, 'account_pnl_pct', self.pnl_pct)
        self.history.set_value(futures.last_date, 'account_pnl_cumpct', self.pnl_cumpct)
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
            self.update_target(futures)
        self.history.set_value(futures.last_date, 'total_margin', self.total_margin)
        return self            
    
    def update_benchmark(self, futures):
        #replace all if sym has changed.

        benchmark_pc=futures.history[futures.history.index==self.benchmark_sym][\
                                                ['LastPctChg','last_date']].copy()
        benchmark_pc=benchmark_pc.reset_index().set_index('last_date')
        #missing_dates = benchmark_pc.index>self.last_date
        #missing_dates=benchmark_pc.loc[missing_dates].index
        #num_dates_missing=len(missing_dates)
        #history=pd.concat([self.history, benchmark_pc], axis=1)
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

        #print(benchmark_pc)
        benchmark_pc.set_value(benchmark_pc.index[0], 'LastPctChg',0.0)
        cp=(benchmark_pc.LastPctChg+1).cumprod()
        
        #add last_date to history
        self.history.benchmark_value=self.starting_value*cp
        self.history.benchmark_sym=self.benchmark_sym
        self.history['benchmark_pctchg']=benchmark_pc.LastPctChg
        self.history['benchmark_cumpct']=cp
        #print(history)
            
        
    def update_pnl(self, futures):
        #create pnl rows in the current atr file

        prev_date=self.last_date.strftime('%Y%m%d')
        #last_date=futures.last_date.strftime('%Y%m%d')
        prev=self.atr_dict[prev_date]
        
        #if not prev.equals(self.atr):
        #    print(prev_date, 'atr_history does not equal equals self.atr!')
                    
        _signals=[x for x in prev.columns if 'signals_' in x]
        for s in _signals:
            pnl = prev.qty\
                    *prev[s]\
                    *prev.contract_value\
                    *self.atr.LastPctChg
                    
            pnl_total=pnl.sum()
            #print(s, pnl, pnl_total)
            column_name=s.replace('signals_','pnl_')
            self.atr[column_name]=pnl
            
            if s==self.prev_selection:
                self.last_pnl=pnl_total
                self.history.set_value(futures.last_date, 'pnl_ACCOUNT', self.last_pnl)
                self.account_value=self.account_value+self.last_pnl
                
            self.history.set_value(futures.last_date, column_name, pnl_total)
                
        
        self.update_benchmark(futures)

        
        #sys.exit()
        #self.account_value=kwargs.get('account_value', self.account_value)
        #self.display=str(self.account_value)[:-3]+'K'
        pass
        
    def update(self, futures, **kwargs):
        print('Performing portfolio update..')
        print('updating kwargs', kwargs)
        self.account_value=kwargs.get('account_value', self.account_value)
        self.board_config=kwargs.get('board_config', self.board_config)
        self.margin_percent=kwargs.get('margin_percent', self.margin_percent)
        self.max_acct_margin=self.margin_percent*self.account_value
        self.increment=kwargs.get('increment', self.increment)
        self.correlation_cutoff=kwargs.get('correlation_cutoff', self.correlation_cutoff)
        self.recreate_if_margin_call=kwargs.get('recreate_if_margin_call',\
                                                self.recreate_if_margin_call)
        self.benchmark_sym=kwargs.get('benchmark_sym', self.benchmark_sym)        
        
        if futures.last_date>self.last_date:
            self.prev_selection=self.last_selection
            self.prev_date=self.last_date

        self.last_selection = kwargs.get('strategy', self.last_selection)
        #reset the comment
        self.comment=''
        
        #update history
        if futures.last_date>self.last_date:
            print('Updating portfolio for..', futures.last_date,\
                  'last_date', self.last_date)
            #add new date to history
            self.history.set_value(futures.last_date, 'comment', self.comment)
            #updates self.atr with new one from futures and adds child signals
            self.append_child_signals(futures)
            
            print('Updating portfolio pnl..')
            #account value needs to be updated first here
            self.update_pnl(futures)
            #update qty, this rewrites the self.atr
            print('Updating portfolio volatility target..')
            self.update_target(futures, **kwargs)
            
            
        
        #update margins >= incase signal choice is changed.
        if futures.last_date>=self.last_date:
            
            #check and update total margins, doesn't rewrite self.atr
            if not self.insufficient_funds:
                print('Updating portfolio margin..')
                self.update_margin(futures, **kwargs)
            self.atr_dict[futures.last_date.strftime('%Y%m%d')]=self.atr
            self.history.set_value(futures.last_date, 'comment', self.comment)
        
        #update history
        if futures.last_date>self.last_date:
            
            #only create() updates self.last_date=futures.last_date
            
            #self.update_history(futures)
            self.last_date=futures.last_date
            self.atr_dict[futures.last_date.strftime('%Y%m%d')]=self.atr
        
            
        self.updated=get_timestamp()
        
        return self
    
    def ranking(self, lookback=None, parents_only=False):
        #if not hasattr(self, 'atr_history_df'):
        #    print('run atr_history() first')
        #    return
        pnl_cols=[x for x in self.history.columns if 'pnl_'==x[:4]]
        ranking = self.history[pnl_cols].sum().sort_values(ascending=False)
        ranking = pd.DataFrame(ranking, columns=['total_pnl'])
        ranking['start_date']=self.history.index[0]
        ranking['end_date']=self.history.index[-1]
        
        if parents_only:
            parents = [i for i in ranking.index if not is_int(i.split('_')[1])]
            return ranking.loc[parents]
        else:
            return ranking

        
    def atr_history(self):
        self.atr_history_df=pd.DataFrame()
        dates=sorted(self.atr_dict.keys())
        for date in dates:
            self.atr_history_df=self.atr_history_df.append(self.atr_dict[date])
        print('self.atr_history_df created')
        return self.atr_history_df
    
        
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
            'positions':str(self.positions),
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
            'pnl_pct':float(self.pnl_pct),
            'pnl_cumpct':float(self.pnl_cumpct),
            'insufficient_funds':bool(self.insufficient_funds),
            'tier':int(self.tier),
            'board_config':str(self.board_config),
            'board_dict':str(self.board_dict),
            'history':self.history.to_json(),
            'atr':self.atr.to_json(),
            #only stores prev atr file in dict, hopefully there is no need to look
            #back further than that
            'atr_prev':self.atr_dict[self.prev_date.strftime('%Y%m%d')].to_json(),
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
        self.positions = eval(portfolio_data['positions'])
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
        self.pnl_pct = portfolio_data['pnl_pct']
        self.pnl_cumpct = portfolio_data['pnl_cumpct']
        self.insufficient_funds = portfolio_data['insufficient_funds']
        self.tier = portfolio_data['tier']
        self.board_config = eval(portfolio_data['board_config'])
        self.board_dict = eval(portfolio_data['board_dict'])
        self.history = pd.read_json(portfolio_data['history'])
        self.atr = pd.read_json(portfolio_data['atr'])
        self.atr_dict[portfolio_data['prev_date']] =\
                        pd.read_json(portfolio_data['atr_prev'])
        
        self.history.index.name='Date'
        
    def copy(self):
        return copy.deepcopy(self)
                    

        
    
