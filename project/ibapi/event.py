#!/usr/bin/python
# -*- coding: utf-8 -*-

class Event(object):
 #interface for all other events
 pass

class MarketEvent(Event):
 #Handles event of receiving a new market update(bars) from DataHandler
 def __init__(self):
        self.type = 'MARKET'

class SignalEvent(Event):
 #used by the Portfolio object, send from Strategy object
 def __init__(self, strategy_id, symbol, datetime, signal_type, strength):
  self.type = 'SIGNAL'
  self.strategy_id = strategy_id
  self.symbol = symbol
  self.datetime = datetime
  self.signal_type = signal_type
  self.strength = strength # useful for mean reversion strategies

class OrderEvent(Event):
 #handles the event of sending an order to an execution system
 def __init__(self, symbol, order_type, quantity, direction):
  self.type = 'ORDER'
  self.symbol = symbol #instrument ticker
  self.order_type = order_type # MKT/LMT
  self.quantity = quantity  #non-negatie integer
  self.direction = direction # BUY/SELL

 def print_order(self):
  #outputs the values within order
  print "Order: Symbol=%s, Type=%s, Quantity=%s, Direction=%s" % (self.symbol, self.order_type, self.quantity, self.direction)

class FillEvent(Event):
 #contains fill data returned from brokerage
 def __init__(self, timeindex,symbol,exchange,quantity, direction, fill_cost, commission = None):
  #if commission is not provided, the fill object will calculate it
  self.type = 'FILL'
  self.timeindex = timeindex #bar resolution when order was filled
  self.symbol = symbol
  self.exchange = exchange
  self.quantity = quantity
  self.direction = direction # BUY/SELL
  self.fill_cost = fill_cost

  #calc commission
  if commission is None:
   self.commission = self.calculate_ib_commission()
  else:
   self.commission = commission

 def calculate_ib_commission(self):
  #usd, need to add exchange/ecn fees
  full_cost = 1.3
  if self.quantity <= 500:
   full_cost = max(1.3, 0.013 * self.quantity)
  else:
   full_cost = max(1.3, 0.008* self.quantity)
  return full_cost

