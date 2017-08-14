import pandas as pd
import numpy as np
import sys
import datetime
from datetime import datetime as dt
from threading import Event
from pytz import timezone
from swigibpy import EWrapper, EPosixClientSocket, Contract

#saveSignals= False
WAIT_TIME = 60.0



class HistoricalDataExample(EWrapper):
    '''Callback object passed to TWS, these functions will be called directly
    by TWS.
    '''

    def __init__(self):
        super(HistoricalDataExample, self).__init__()
        self.got_history = Event()
        self.data = pd.DataFrame(columns = ['Open','High','Low','Close','Volume'])

    def orderStatus(self, id, status, filled, remaining, avgFillPrice, permId,
                    parentId, lastFilledPrice, clientId, whyHeld):
        pass

    def openOrder(self, orderID, contract, order, orderState):
        pass

    def nextValidId(self, orderId):
        '''Always called by TWS but not relevant for our example'''
        pass

    def openOrderEnd(self):
        '''Always called by TWS but not relevant for our example'''
        pass

    def managedAccounts(self, openOrderEnd):
        '''Called by TWS but not relevant for our example'''
        pass

    def getData(self):
        return self.data;
        
    def historicalData(self, reqId, date, open, high,
                       low, close, volume,
                       barCount, WAP, hasGaps):

        if date[:8] == 'finished':
            print("History request complete")
            self.got_history.set()
        else:
            self.data.loc[date] = [open,high,low,close,volume]
            #print "History %s - Open: %s, High: %s, Low: %s, Close: %s, Volume: %d"\
            #          % (date, open, high, low, close, volume)

            #print(("History %s - Open: %s, High: %s, Low: %s, Close: "
            #       "%s, Volume: %d, Change: %s, Net: %s") % (date, open, high, low, close, volume, chgpt, chg));

        #return self.data


def getDataFromIB(brokerData,endDateTime):
    #data_cons = pd.DataFrame()
    # Instantiate our callback object
    callback = HistoricalDataExample()

    # Instantiate a socket object, allowing us to call TWS directly. Pass our
    # callback object so TWS can respond.
    tws = EPosixClientSocket(callback)
    #tws = EPosixClientSocket(callback, reconnect_auto=True)
    # Connect to tws running on localhost
    if not tws.eConnect("", brokerData['port'], brokerData['client_id']):
        raise RuntimeError('Failed to connect to TWS')

    # Simple contract for GOOG
    contract = Contract()
    contract.exchange = brokerData['exchange']
    contract.symbol = brokerData['symbol']
    contract.secType = brokerData['secType']
    contract.currency = brokerData['currency']
    ticker = contract.symbol+contract.currency
    #today = dt.today()

    print("\nRequesting historical data for %s" % ticker)

    # Request some historical data.

    #for endDateTime in getHistLoop:
    tws.reqHistoricalData(
        brokerData['tickerId'],                                         # tickerId,
        contract,                                   # contract,
        endDateTime,                            #endDateTime
        brokerData['durationStr'],                                      # durationStr,
        brokerData['barSizeSetting'],                                    # barSizeSetting,
        brokerData['whatToShow'],                                   # whatToShow,
        brokerData['useRTH'],                                          # useRTH,
        brokerData['formatDate']                                          # formatDate
        )


    print("====================================================================")
    print(" %s History requested, waiting %ds for TWS responses" % (endDateTime, WAIT_TIME))
    print("====================================================================")


    try:
        callback.got_history.wait(timeout=WAIT_TIME)
    except KeyboardInterrupt:
        pass
    finally:
        if not callback.got_history.is_set():
            print('Failed to get history within %d seconds' % WAIT_TIME)
    
    #data_cons = pd.concat([data_cons,callback.data],axis=0)
             
    print("Disconnecting...")
    tws.eDisconnect()
        
    return callback.data
