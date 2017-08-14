#!/usr/bin/python
#-*- coding: utf-8 -*-

import datetime
import time

from ib.ext.Contract import Contract
from ib.ext.Order import Order
from ib.opt import ibConnection, message

from event import FillEvent, OrderEvent
from execution import ExecutionHandler

class IBExecutionHandler(ExecutionHandler):
    # Handles execution via IB API for live trading
    
    def __init__(self, events, order_routing="SMART", currency="USD"):
        self.events = events
        self.order_routing = order_routing
        self.currency = currency
        self.fill_dict = {}
        
        self.tws_conn = self.create_tws_connection()
        self.order_id = self.create_initial_order_id()
        self.register_handlers()
        
    def _error_handler(self, msg):
        # Handles the capture of error messages
        # Insert error handling here
        print "Server Error: %s" % msg
    
    def _reply_handler(self, msg):
        # Determines if a Fill Event needs to be created
        
        # Handles openOrder messages
        if (msg.typeName == 'openOrder' and msg.orderId == self.order_id and 
                            not self.fill_dict.has_key(msg.orderId)):
            self.create_fill_dict_entry(msg)
        # Handles orderStatus Filled messages
        if (msg.typeName == 'orderStatus' and msg.status == 'Filled' and 
                            not self.fill_dict[msg.orderId]['filled']):
            self.create_fill(msg)
        print 'Server Response: %s, %s\n' % (msg.typeName, msg)
    
    def create_tws_connection(self):
        """ Connect to the TWs running on port 7496, with a clientId of 10.
        Need a separate ID for market data connection, if used elsewhere.
        """
        tws_conn = ibConnection()
        tws_conn.connect()
        return tws_conn
    
    def create_initial_order_id(self):
        """ Creates the initial order ID.  '1' as default.  Query the IB for 
        latest available ID and use that. Reset the order ID via TWS > Global
        Configuration > API Setting
        """
        return 1
    
    def register_handlers(self):
        """ Register the error and server reply message functions
        """
        # Assign the error handling function to the TWS connection
        self.tws_conn.register(self._error_handler,'Error')
        
        # Assign all of the server reply messages to the reply_handler
        self.tws-conn.registerAll(self._reply_handler)
    
    def create_contract(self, symbol, sec_type, exch, prim_exch, curr):
        """ Create a contract object defining purchase specifications. To 
        transact a trade it is necessary to create an IbPy Contract instance 
        and pair it with an IbPy Order instance to send to the IB API.
        """
        contract = Contract()
        contract.m_symbol = symbol # ticker
        contract.m_secType = sec_type # security type ('STK' = stock)
        contract.m_exchange = exch # exchange
        contract.m_primaryExch = prim_exch # primary exchange 
        contract.m_currency = curr # currency in which to purchase asset
        return contract
    
    def create_order(self, order_type, quantity, action):
        """ Create an Order Object to pair with the Contract Object
        """
        order = Order()
        order.m_orderType = order_type # 'MKT'/'LMT'
        order.m_totalQuantity = quantity # Integer
        order.m_action = action # 'BUY'/'SELL'
        return order
    
    def create_fill_dict_entry(self, msg):
        """ Creates an entry in the fill_dict to avoid duplicating FillEvent
        instances for a particular order ID. When a fill has been generated 
        the 'filled' key of an entry for the order is set to True. If IB 
        duplicates the message, it will not lead to a new fill.
        """
        self.fill_dict[msg.orderId] = {
            "symbol":msg.contract.m_symbol,
            "exchange": msg.contract.m_exchange,
            "direction": msg.order.m_action,
            "filled": False
        }
    
    def create_fill(self, msg):
        """ Creates the FillEvent and places it onto the events queue, 
        after order is filled.
        """
        fd = self.fill_dict[msg.orderId]
        
        # Prepare fill data
        symbol = fd["symbol"]
        exchange = fd["exchange"]
        filled = msg.filled
        direction = fd["direction"]
        fill_cost = msg.avgFillPrice
        
        # Create a fill event object
        fill = FillEvent(
            datetime.datetime.utcnow(), symbol, exchange, filled, direction,
            fill_cost
            )
        
        # Makes sure multiple messages don't create additional fills
        self.fill_dict[msg.orderId]['filled'] = True
        
        # Place fill event into queue
        self.events.put(fill_event)
    
    def execution_order(self, event):
        """ Creates the IB order objects and sumbits it via IB API. 
        """
        
        if event.type == 'ORDER':
            # Prepare parameters for the order
            asset = event.symbol
            asset_type = "STK"
            order_type = event.order_type
            quantity = event.quantity
            direction = event.direction
            
            # Create IB contract
            ib_contract = self.create_contract( asset, asset_type,
                            self.order_routing, self.order_routing, 
                            self.currency
                            )
            
            # Create the IB order via the passed Order event
            ib_order = self.create_order(order_type, quantity, direction)
            
            # Send the order to IB
            self.tws_conn.placeOrder(self.order_id, ib_contract, ib_order)
            
            # Removes inconsistent behavior of the API
            time.sleep(1)
            
            # Increment the order ID
            self.order_id += 1
