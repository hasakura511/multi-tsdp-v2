# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 21:26:44 2016

@author: Hidemi
"""

import re
import ib
from ib.ext.Contract import Contract
from ib.opt import ibConnection, message
from time import sleep
from ib.ext.Contract import Contract
from ib.ext.ContractDetails import ContractDetails
from ib.opt import ibConnection, message
import time

def watcher(msg):
    print msg

contracts = [] # to store all the contracts
def contractDetailsHandler(msg):
    contracts.append(msg.contractDetails.m_summary)

con = ibConnection()
con.registerAll(watcher)
con.register(contractDetailsHandler, 'ContractDetails')
con.connect()

contract = Contract()
contract.m_symbol = "ES"
contract.m_exchange = "GLOBEX"
contract.m_currency = "USD"
contract.m_secType = "FUT"

con.reqContractDetails(1, contract)

time.sleep(2)

con.disconnect()

class Downloader(object):
    field4price = ''

    def __init__(self):
        self.tws = ibConnection()
        self.tws.register(self.tickPriceHandler, 'TickPrice')
        self.tws.connect()
        self._reqId = 1 # current request id

    def tickPriceHandler(self,msg):
        if msg.field == 4:
            self.field4price = msg.price
            #print '[debug]', msg

    def requestData(self,contract): 
        self.tws.reqMktData(self._reqId, contract, '', 1)
        self._reqId+=1

if __name__=='__main__':
    dl = Downloader()
    c = Contract()
    c.m_symbol = 'SPY'
    c.m_secType = 'STK'
    c.m_exchange = 'SMART'
    c.m_currency = 'USD'
    dl.requestData(c)
    sleep(3)
    print 'Price - field 4: ', dl.field4price