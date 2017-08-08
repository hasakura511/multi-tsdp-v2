#! /usr/bin/env python
# -*- coding: utf-8 -*-

from ib.ext.Contract import Contract
from ib.opt import ibConnection, message
from time import sleep
import time

#class Datacol:
#def __init__(self, contract, date, t, duration, outfile):
#outfile = outfile
#contract = contract
#date = date
#t = t


def close(self):
    while not data_received:
      pass
    con.cancelHistoricalData(tick_id)
    time.sleep(1)
    con.disconnect()
    time.sleep(1)

def process_data(self, msg):
    if msg.open != -1:
      print>>outfile, msg.date, msg.open, msg.high, msg.low, msg.close, msg.volume, msg.count, msg.WAP, msg.hasGaps
    else:
      data_received = True
# print all messages from TWS
def watcher(msg):
    print msg

# show Bid and Ask quotes
def my_BidAsk(msg):
    if msg.field == 1:
        print '%s:%s: bid: %s' % (contractTuple[0],
                       contractTuple[6], msg.price)
    elif msg.field == 2:
        print '%s:%s: ask: %s' % (contractTuple[0], contractTuple[6], msg.price)

def makeStkContract(contractTuple):
    newContract = Contract()
    newContract.m_symbol = contractTuple[0]
    newContract.m_secType = contractTuple[1]
    newContract.m_exchange = contractTuple[2]
    newContract.m_currency = contractTuple[3]
    newContract.m_expiry = contractTuple[4]
    newContract.m_strike = contractTuple[5]
    newContract.m_right = contractTuple[6]
    print 'Contract Values:%s,%s,%s,%s,%s,%s,%s:' % contractTuple
    return newContract

if __name__ == '__main__':
    con = ibConnection()
    con.registerAll(watcher)
    showBidAskOnly = False  # set False to see the raw messages
    if showBidAskOnly:
        con.unregister(watcher, message.tickSize, message.tickPrice,
                       message.tickString, message.tickOptionComputation)
        con.register(my_BidAsk, message.tickPrice)
    con.connect()
    sleep(1)
    tickId = 1

    # Note: Option quotes will give an error if they aren't shown in TWS
    contractTuple = ('SPY', 'STK', 'SMART', 'USD', '', 0.0, '')
    #contractTuple = ('QQQQ', 'OPT', 'SMART', 'USD', '20070921', 47.0, 'CALL')
    #contractTuple = ('ES', 'FUT', 'GLOBEX', 'USD', '20160318', 0.0, '')
    #contractTuple = ('N225M', 'FUT', 'OSE.JPN', 'JPY', '20160311', 0.0, '')
    #contractTuple = ('ES', 'FOP', 'GLOBEX', 'USD', '20070920', 1460.0, 'CALL')
    #contractTuple = ('EUR', 'CASH', 'IDEALPRO', 'USD', '', 0.0, '')
    stkContract = makeStkContract(contractTuple)
    print '* * * * REQUESTING MARKET DATA * * * *'
    con.reqMktData(tickId, stkContract, '', False)
    duration = 28800
    duration = str(duration) + ' S'
    t = '00:00:00'
    #tick_id = 1
    #con = ibConnection()
    con.register(process_data, message.historicalData)
    #con.connect()
    time.sleep(1)
    date = '20160219'
    end_datetime = ('%s %s US/Eastern' % (date, t))
    bar = con.reqHistoricalData(tickerId=tickId, contract=stkContract, endDateTime=end_datetime, durationStr=duration, barSizeSetting='8 hours', whatToShow='TRADES', useRTH=0, formatDate=1)
    data_received = False
    sleep(15)
    print '* * * * CANCELING MARKET DATA * * * *'
    con.cancelMktData(tickId)
    sleep(1)
    con.disconnect()
    sleep(1)
