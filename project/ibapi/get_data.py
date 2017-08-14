
import sys

import pandas as pd

#import datacol
from ib.ext.Contract import Contract

from ib.opt import ibConnection, message
import time


class Datacol:
  def __init__(self, contract, date, t, duration, outfile):
    self.outfile = outfile
    self.contract = contract
    self.date = date
    self.t = t
    self.duration = str(duration) + ' S'
    self.tick_id = 1
    self.con = ibConnection()
    self.con.register(self.process_data, message.historicalData)
    self.con.connect()
    time.sleep(1)
    end_datetime = ('%s %s US/Eastern' % (self.date, self.t))
    self.con.reqHistoricalData(tickerId=self.tick_id, contract=self.contract, endDateTime=end_datetime, durationStr=self.duration, barSizeSetting='5 secs', whatToShow='TRADES', useRTH=0, formatDate=1)
    self.data_received = False

  def close(self):
    while not self.data_received:
      pass
    self.con.cancelHistoricalData(self.tick_id)
    time.sleep(1)
    self.con.disconnect()
    time.sleep(1)

  def process_data(self, msg):
    if msg.open != -1:
      print>>self.outfile, msg.date, msg.open, msg.high, msg.low, msg.close, msg.volume, msg.count, msg.WAP, msg.hasGaps
    else:
      self.data_received = True

#def main():
#contractTuple = ('QQQQ', 'STK', 'SMART', 'USD', '', 0.0, '')
#contractTuple = ('QQQQ', 'OPT', 'SMART', 'USD', '20070921', 47.0, 'CALL')
contractTuple = ('ES', 'FUT', 'GLOBEX', 'USD', '201609', 0.0, '')
#contractTuple = ('ES', 'FOP', 'GLOBEX', 'USD', '20070920', 1460.0, 'CALL')
#contractTuple = ('EUR', 'CASH', 'IDEALPRO', 'USD', '', 0.0, '')

#symbol = sys.argv[1]
#exch = sys.argv[2]
#expiry = sys.argv[3]
#date = sys.argv[4]
symbol = contractTuple[0]
exch = contractTuple[2]
expiry = contractTuple[4]
date = '20160219'
d = pd.to_datetime(date) - pd.DateOffset(1)
prev_date = '%04d%02d%02d' % (d.year, d.month, d.day)

contract = Contract()
contract.m_symbol = symbol
contract.m_secType = 'FUT'
contract.m_exchange = exch
contract.m_currency = 'USD'
contract.m_expiry = expiry
print 'Collecting', date, 'data for', contract.m_symbol, 'expiration', contract.m_expiry

outfile = open(date+'.bars', 'w')

for h in xrange(20,24,2):
    broker = datacol.Datacol(contract, prev_date, ('%02d:00:00' % h), 7200, outfile)
    broker.close()

for h in xrange(0,18,2):
    broker = datacol.Datacol(contract, date, ('%02d:00:00' % h), 7200, outfile)
    broker.close()

broker = datacol.Datacol(contract, date, '17:15:00', 4500, outfile)
broker.close()

outfile.close()


#if __name__ == "__main__":
#  main()