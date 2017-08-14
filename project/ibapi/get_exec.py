from wrapper_v4 import IBWrapper, IBclient
import time
import pandas as pd
from time import gmtime, strftime, time, localtime, sleep
import json
from pandas.io.json import json_normalize
import os


def get_exec():
    callback = IBWrapper()
    client=IBclient(callback)
    execlist=client.get_executions()
    return execlist;

def get_exec_open():
    callback = IBWrapper()
    client=IBclient(callback)    
    (account_value, portfolio_data)=client.get_IB_account_data()    
    return (account_value, portfolio_data)

def get_ibpos(**kwargs):
    portfolioPath='./data/portfolio/'
    (account_value, portfolio_data)=get_exec_open()
    data=pd.DataFrame(portfolio_data,columns=['sym','exp','qty','price','value','avg_cost','unr_pnl','real_pnl','accountid','currency'])
    dataSet=pd.DataFrame(data)
    #dataSet=dataSet.sort_values(by='times')
    dataSet['symbol']=dataSet['sym'] + dataSet['currency'] 
    dataSet=dataSet.set_index(['symbol'])
    dataSet.to_csv(portfolioPath+'ib_portfolio.csv')
    accountSet=pd.DataFrame(account_value)
    accountSet.to_csv(portfolioPath+'ib_account_value.csv', index=False)
    #
    return dataSet

def get_iblivepos():
    return get_ibpos()

def get_ibpos_from_csv():
    dataSet = pd.read_csv('./data/portfolio/ib_portfolio.csv', index_col=['sym','currency'])
   #
    return dataSet

def get_ib_sym_pos(portfolio_data, symbol, currency):
    portfolio_data=portfolio_data.reset_index()
    portfolio_data['symbol']=portfolio_data['sym'] + portfolio_data['currency']
    sym_cur=symbol + currency
    if sym_cur not in portfolio_data['symbol'].values:
        portfolio_data=portfolio_data.append(pd.DataFrame([[sym_cur,symbol,0,currency]], \
                              columns=['symbol','sym','qty','currency']))
    dataSet=portfolio_data
    #dataSet=dataSet.sort_values(by='times')
    dataSet=dataSet.set_index(['sym','currency'])
    #dataSet.to_csv('./data/portfolio/ib_portfolio.csv')
    #accountSet=pd.DataFrame(account_value)
    #accountSet.to_csv('./data/portfolio/ib_account_value.csv', index=False)
    #
    return dataSet

def get_ib_portfolio():
    filename='./data/portfolio/ib_portfolio.csv'
    
    if os.path.isfile(filename):
        dataSet = pd.read_csv(filename, index_col='symbol')
        if 'PurePL' not in dataSet:
            dataSet['PurePL']=0
        dataSet=dataSet.reset_index()
        dataSet['symbol']=dataSet['sym'] + dataSet['currency'] 
        dataSet=dataSet.set_index('symbol')
        return dataSet
    else:

        dataSet=pd.DataFrame({}, columns=['sym','exp','qty','openqty','price','openprice','value','avg_cost','unr_pnl','real_pnl','PurePL','accountid','currency'])
        dataSet['symbol']=dataSet['sym'] + dataSet['currency']        
        dataSet=dataSet.set_index('symbol')
        dataSet.to_csv(filename)
        return dataSet
   
def get_ib_pos(symbol, currency):
    portfolio_data=get_ib_portfolio()
    portfolio_data=portfolio_data.reset_index()
    portfolio_data['symbol']=portfolio_data['sym'] + portfolio_data['currency']
    sym_cur=symbol + currency
    portfolio_data=portfolio_data.set_index('symbol')
    if sym_cur not in portfolio_data.index.values:
       return 0
    else:
        ib_pos=portfolio_data.loc[sym_cur]
        ib_pos_qty=ib_pos['qty']
        return ib_pos_qty
#get_exec();
