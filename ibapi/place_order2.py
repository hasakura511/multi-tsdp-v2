import sys
import random
import time
from threading import Event
import logging

from swigibpy import (EWrapper, EPosixClientSocket, Contract, Order, TagValue,
                      TagValueList)


WAIT_TIME = 10.0


try:
    # Python 2 compatibility
    input = raw_input
    from Queue import Queue
except:
    from queue import Queue

###


class PlaceOrderExample(EWrapper):
    '''Callback object passed to TWS, these functions will be called directly
    by TWS.
    '''

    def __init__(self):
        super(PlaceOrderExample, self).__init__()
        self.order_filled = Event()
        self.order_ids = Queue()

    def openOrderEnd(self):
        '''Not relevant for our example'''
        pass

    def execDetails(self, id, contract, execution):
        '''Not relevant for our example'''
        pass

    def managedAccounts(self, openOrderEnd):
        '''Not relevant for our example'''
        pass

    ###############

    def nextValidId(self, validOrderId):
        '''Capture the next order id'''
        self.order_ids.put(validOrderId)

    def orderStatus(self, id, status, filled, remaining, avgFillPrice, permId,
                    parentId, lastFilledPrice, clientId, whyHeld):

        logging.info(("Order #%s - %s (filled %d, remaining %d, avgFillPrice %f,"
               "last fill price %f)") %
              (id, status, filled, remaining, avgFillPrice, lastFilledPrice))
        if remaining <= 0:
            self.order_filled.set()

    def openOrder(self, orderID, contract, order, orderState):

        logging.info("Order opened for %s" % contract.symbol)

    def commissionReport(self, commissionReport):
        logging.info('Commission %s %s P&L: %s' % (commissionReport.currency,
                                            commissionReport.commission,
                                            commissionReport.realizedPNL))

def place_orders(execDict, clientid):
    # Instantiate our callback object
    callback = PlaceOrderExample()

    # Instantiate a socket object, allowing us to call TWS directly. Pass our
    # callback object so TWS can respond.
    tws = EPosixClientSocket(callback)
    
    # Connect to tws running on localhost
    #tid=random.randint(1,10000)

    if not tws.eConnect("", 7496, clientid):
        raise RuntimeError('Failed to connect to TWS')
        
    for sym in execDict.keys():
        action, quant, contract = execDict[sym]

        if action == 'BOT':
            action = 'BUY'
        elif action == 'SLD':
            action='SELL'
        else:
            logging.info('skipping'+sym)
            continue     

        #prompt = input("WARNING: This example will place an order on your IB "
        #               "account, are you sure? (Type yes to continue): ")
        #if prompt.lower() != 'yes':
        #    sys.exit()
        

        
        # Simple contract for GOOG
        #contract = Contract()
        #contract.symbol = sym
        #contract.secType = type
        #contract.exchange = exchange
        #contract.currency = currency
        #contract.localSymbol=iblocalsym

        logging.info('Waiting for valid order id')
        if callback.order_ids.empty():
            tws.reqIds(0)

        order_id = callback.order_ids.get(timeout=WAIT_TIME)
        if not order_id:
            raise RuntimeError('Failed to receive order id after %ds' % WAIT_TIME)
        
        # Order details
        algoParams = TagValueList()
        #algoParams.append(TagValue("componentSize", "3"))
        #algoParams.append(TagValue("timeBetweenOrders", "60"))
        #algoParams.append(TagValue("randomizeTime20", "1"))
        #algoParams.append(TagValue("randomizeSize55", "1"))
        #algoParams.append(TagValue("giveUp", "1"))
        #algoParams.append(TagValue("catchUp", "1"))
        algoParams.append(TagValue("waitForFill", "1"))
        #algoParams.append(TagValue("startTime", "20110302-14:30:00 GMT"))
        #algoParams.append(TagValue("endTime", "20110302-21:00:00 GMT"))

        order = Order()
        order.action = action
        #order.lmtPrice = 140
        order.orderType = 'MKT'
        order.totalQuantity = int(quant)
        #order.algoStrategy = "AD"
        order.tif = 'DAY'
        #order.algoParams = algoParams
        #order.transmit = False

        
        logging.info("Placing order for %d %s's (id: %d)" % (order.totalQuantity,
                                                      contract.symbol, order_id))
        
        # Place the order
        tws.placeOrder(
            order_id,                                   # orderId,
            contract,                                   # contract,
            order                                       # order
        )
        
        logging.info("\n====================================================================")
        logging.info(" Order placed, waiting %ds for TWS responses" % WAIT_TIME)
        logging.info("====================================================================\n")
        
        
        logging.info("Waiting for order to be filled..")
        
        try:
            callback.order_filled.wait(WAIT_TIME)
        except KeyboardInterrupt:
            pass
        finally:
            if not callback.order_filled.is_set():
                logging.info('Failed to fill order')
        time.sleep(2)
        
    logging.info("\nDisconnecting...")
    tws.eDisconnect()
#place_order("BUY", 1, "EUR", "CASH", "USD", "IDEALPRO");
