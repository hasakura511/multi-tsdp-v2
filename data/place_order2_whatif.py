import sys
import random
import time
from threading import Event
import logging
from ibapi.wrapper_v5 import IBWrapper, IBclient
from swigibpy import (EWrapper, EPosixClientSocket, Contract, Order, TagValue,
                      TagValueList)
import pandas as pd

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
        self.whatif_filled = Event()
        self.order_ids = Queue()
        self.orderstate={}

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

        print(("Order #%s - %s (filled %d, remaining %d, avgFillPrice %f,"
               "last fill price %f)") %
              (id, status, filled, remaining, avgFillPrice, lastFilledPrice))
        if remaining <= 0:
            self.order_filled.set()

    def openOrder(self, orderID, contract, order, orderState):

        print("Order opened for %s %s" % (contract.symbol, orderID))
        print('status %s, initMargin %s, maintMargin %s, equityWithLoan %s,\
                    commission %s, minCommission %s, maxCommission %s,\
                    commissionCurrency %s, warningText %s' % (orderState.status,
                          orderState.initMargin, orderState.maintMargin, orderState.equityWithLoan,
                    orderState.commission, orderState.minCommission, orderState.maxCommission,
                    orderState.commissionCurrency, orderState.warningText))
        self.orderstate['status']=orderState.status
        self.orderstate['initMargin']=orderState.initMargin
        self.orderstate['maintMargin']=orderState.maintMargin
        self.orderstate['equityWithLoan']=orderState.equityWithLoan
        self.orderstate['commission']=orderState.commission
        self.orderstate['minCommission']=orderState.minCommission
        self.orderstate['maxCommission']=orderState.maxCommission
        self.orderstate['commissionCurrency']=orderState.commissionCurrency
        self.orderstate['warningText']=orderState.warningText
        self.whatif_filled.set()
        
    def commissionReport(self, commissionReport):
        print('Commission %s %s P&L: %s' % (commissionReport.currency,
                                            commissionReport.commission,
                                            commissionReport.realizedPNL))
    '''
    def orderState(self, status, initMargin, maintMargin, equityWithLoan,
                    commission, minCommission, maxCommission,
                    commissionCurrency, warningText):
        print('status %s, initMargin %s, maintMargin %s, equityWithLoan %s,\
                    commission %s, minCommission %s, maxCommission %s,\
                    commissionCurrency %s, warningText %s' % (status, initMargin, maintMargin, equityWithLoan,
                    commission, minCommission, maxCommission,
                    commissionCurrency, warningText))
    '''
def place_orders(execDict, clientid, whatIf = False):
    # Instantiate our callback object
    callback = PlaceOrderExample()

    orderstates=pd.DataFrame()
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
            print('skipping'+sym)
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

        print('Waiting for valid order id')
        if callback.order_ids.empty():
            tws.reqIds(0)
            order_id = callback.order_ids.get(timeout=WAIT_TIME)
        else: 
            order_id+=1

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
        order.whatIf = whatIf
        #order.algoParams = algoParams
        #order.transmit = False

        
        print("Placing order for %d %s's (id: %d)" % (order.totalQuantity,
                                                      contract.symbol, order_id))
        
        # Place the order
        tws.placeOrder(
            order_id,                                   # orderId,
            contract,                                   # contract,
            order                                       # order
        )
        

        if not whatIf:
            print("\n====================================================================")
            print(" Order placed, waiting %ds for TWS responses" % WAIT_TIME)
            print("====================================================================\n")
            
            print("Waiting for order to be filled..")
            
            try:
                callback.order_filled.wait(WAIT_TIME)
            except KeyboardInterrupt:
                pass
            finally:
                if not callback.order_filled.is_set():
                    print('Failed to fill order')
            time.sleep(2)
        else:
            print("Waiting for whatif response..")
            try:
                callback.whatif_filled.wait(WAIT_TIME)
            except KeyboardInterrupt:
                pass
            finally:
                if not callback.whatif_filled.is_set():
                    print('Failed to fill order')
                else:
                    orderstates=orderstates.append(pd.Series(name=contract.symbol,data=callback.orderstate))
                    callback.whatif_filled.clear()
            #time.sleep(2)
            
        
        
    print("\nDisconnecting...")
    tws.eDisconnect()
    if whatIf:
        return orderstates
        
def place_orders2(contract, order, clientid):
    # Instantiate our callback object
    callback = PlaceOrderExample()

    # Instantiate a socket object, allowing us to call TWS directly. Pass our
    # callback object so TWS can respond.
    tws = EPosixClientSocket(callback)
    
    # Connect to tws running on localhost
    #tid=random.randint(1,10000)

    if not tws.eConnect("", 7496, clientid):
        raise RuntimeError('Failed to connect to TWS')
        
    #for sym in execDict.keys():
    #    action, quant, contract = execDict[sym]

    #    if action == 'BOT':
    #        action = 'BUY'
    #    elif action == 'SLD':
    #        action='SELL'
    #    else:
    #        print('skipping'+sym)
    #        continue     

        #prompt = input("WARNING: This example will place an order on your IB "
        #               "account, are you sure? (Type yes to continue): ")
        #if prompt.lower() != 'yes':
        #    sys.exit()
        

        
        # Simple contract for GOOG
        #contract = Contract()
        #contract.symbol = "GC"
        #contract.secType = "FUT"
        #contract.exchange = "GLOBEX"
        #contract.currency = "USD"
        #contract.localSymbol=iblocalsym

    print('Waiting for valid order id')
    if callback.order_ids.empty():
        tws.reqIds(0)

    order_id = callback.order_ids.get(timeout=WAIT_TIME)
    if not order_id:
        raise RuntimeError('Failed to receive order id after %ds' % WAIT_TIME)
    
    # Order details
    #algoParams = TagValueList()
    #algoParams.append(TagValue("componentSize", "3"))
    #algoParams.append(TagValue("timeBetweenOrders", "60"))
    #algoParams.append(TagValue("randomizeTime20", "1"))
    #algoParams.append(TagValue("randomizeSize55", "1"))
    #algoParams.append(TagValue("giveUp", "1"))
    #algoParams.append(TagValue("catchUp", "1"))
    #algoParams.append(TagValue("waitForFill", "1"))
    #algoParams.append(TagValue("startTime", "20110302-14:30:00 GMT"))
    #algoParams.append(TagValue("endTime", "20110302-21:00:00 GMT"))

    #order = Order()
    #order.action = 'BUY'
    #order.lmtPrice = 140
    #order.orderType = 'MKT'
    #order.totalQuantity = 1
    #order.algoStrategy = "AD"
    #order.tif = 'DAY'
    #order.whatIf = True
    #order.algoParams = algoParams
    #order.transmit = False

    
    print("Placing order for %d %s's (id: %d)" % (order.totalQuantity,
                                                  contract.symbol, order_id))
    
    # Place the order
    tws.placeOrder(
        order_id,                                   # orderId,
        contract,                                   # contract,
        order                                       # order
    )
    
    print("\n====================================================================")
    print(" Order placed, waiting %ds for TWS responses" % WAIT_TIME)
    print("====================================================================\n")
    
    
    print("Waiting for order to be filled..")
    
    try:
        callback.order_filled.wait(WAIT_TIME)
    except KeyboardInterrupt:
        pass
    finally:
        if not callback.order_filled.is_set():
            print('Failed to fill order')
    time.sleep(2)
        
    print("\nDisconnecting...")
    tws.eDisconnect()
#place_order("BUY", 1, "EUR", "CASH", "USD", "IDEALPRO");

def create_execDict(feeddata, systemdata):
    global debug
    global client
    #global csidate
    #global ttdate
    def getContractDate(c2sym, systemdata):
        currentcontract = [x for x in systemdata.c2sym if x[:-2] == c2sym]
        if len(currentcontract)==1:
            #Z6
            ccontract = currentcontract[0][-2:]
            ccontract = 201000+int(ccontract[-1])*100+months[ccontract[0]]
            return str(ccontract)
        else:
            return ''    
    #after 1AM EST server reset
    #downloadtt = not ttdate==int(ib_server_reset_date)
    #print 'ttdate',ttdate, 'ib_server_reset_date',ib_server_reset_date, 'downloadtt', downloadtt,'refresh_timetable',refresh_timetable
    execDict=dict()
    #need systemdata for the contract expiry
    #systemdata=pd.read_csv(systemfile)

    #openPositions=get_ibfutpositions(portfolioPath)
    #print feeddata.columns
    #if True:
    #if downloadtt or refresh_timetable:
    downloadtt=True
    print 'new timetable due, geting new contract details from IB'
    contractsDF=pd.DataFrame()
    #else:
    #    print 'loading contract details from file'
    #    contractsDF=pd.read_csv(systemPath+'ib_contracts.csv', index_col='ibsym')
        
    feeddata=feeddata.reset_index()
    for i in feeddata.index:
        
        #print 'Read: ',i
        system=feeddata.ix[i]
        c2sym=system.c2sym
        ibsym=system.ibsym   
        index = systemdata[systemdata.c2sym2==c2sym].index[0]
        #find the current contract

        #print system
        contract = Contract()
        
        if system['ibtype'] == 'CASH':
            #fx
            symbol=system['ibsym']+system['ibcur']
            contract.symbol=system['ibsym']
            ccontract = ''
        else:
            #futures
            contract.expiry = getContractDate(system.c2sym, systemdata)
            symbol=contract.symbol= system['ibsym']
            contract.multiplier = str(system['multiplier'])

        #contract.symbol = system['ibsym']
        contract.secType = system['ibtype']
        contract.exchange = system['ibexch']
        contract.currency = system['ibcur']
        
        print i+1, contract.symbol, contract.expiry
        if downloadtt:
            contractInfo=client.get_contract_details(contract)
            #print contractInfo
            contractsDF=contractsDF.append(contractInfo)
            execDict[symbol+contractInfo.expiry[0]]=['BOT', 1, contract]
            systemdata.set_value(index, 'ibcontract', symbol+contractInfo.expiry[0])
        #else:
        #    execDict[contractsDF.ix[symbol].contracts]=['PASS', 0, contract]
        #    systemdata.set_value(index, 'ibcontract', contractsDF.ix[symbol].contracts)
            
        #update system file with correct ibsym and contract expiry
        #print index, ibsym, contract.expiry, systemdata.columns
        systemdata.set_value(index, 'ibsym', ibsym)
        systemdata.set_value(index, 'ibcontractmonth', contract.expiry)
        
    '''
        #print c2sym, ibsym, systemdata.ix[index].ibsym.values, systemdata.ix[index].c2sym.values, contract.expiry
    
    #systemdata.to_csv(systemfile, index=False)
    #systemdata.to_sql(name='v4futures_moc_live', if_exists='replace', con=writeConn, index=False)
    #print '\nsaved v4futures_moc_live to', dbPath
    
    if downloadtt:
        feeddata=feeddata.set_index('ibsym')
        contractsDF=contractsDF.set_index('symbol')
        contractsDF.index.name = 'ibsym'
        contractsDF['contracts']=[x+contractsDF.ix[x].expiry for x in contractsDF.index]
        #contractsDF['Date']=csidate
        #contractsDF['timestamp']=int(calendar.timegm(dt.utcnow().utctimetuple()))
        #print contractsDF.index
        #print feeddata.ix[contractsDF.index].drop(['ibexch','ibtype','ibcur'],axis=1).head()
        contractsDF = pd.concat([ feeddata.ix[contractsDF.index].drop(['ibexch','ibtype','ibcur','Date','timestamp'],axis=1),contractsDF], axis=1)
        #try:
        #    contractsDF.to_sql(name='ib_contracts', con=writeConn, index=True, if_exists='replace', index_label='ibsym')
        #    print '\nsaved ib_contracts to',dbPath
        #except Exception as e:
            #print e
        #    traceback.print_exc()
        #if not debug:
        #    contractsDF.to_csv(systemPath+'ib_contracts.csv', index=True)
        #    print 'saved', systemPath+'ib_contracts.csv'
    '''
    print '\nCreated exec dict with', len(execDict.keys()), 'contracts:'
    print execDict.keys()
    return execDict,contractsDF,systemdata

months = {
                'F':1,
                'G':2,
                'H':3,
                'J':4,
                'K':5,
                'M':6,
                'N':7,
                'Q':8,
                'U':9,
                'V':10,
                'X':11,
                'Z':12
                }
feedfile='./data/systems/system_ibfeed.csv'
feeddata=pd.read_csv(feedfile,index_col='ibsym')
futuresDF_results=pd.read_csv('./data/futuresATR.csv',  index_col=0)
futuresDF_results.index.name='CSIsym'
futuresDF = futuresDF_results.ix[feeddata.CSIsym.tolist()].copy()
futuresDF = futuresDF.reset_index()
futuresDF.rename(columns=lambda x: x.replace('Contract', 'c2sym'), inplace=True)
futuresDF['c2sym2']=[x[:-2] for x in futuresDF.c2sym]

callback = IBWrapper()
client=IBclient(callback, port=7496, clientid=0)
execDict,contractsDF,systemdata= create_execDict(feeddata, futuresDF)
(account_value, portfolio_data)=client.get_IB_account_data()
client.disconnect()

orderstates=place_orders(execDict, 1, whatIf = True)

accountSet=pd.DataFrame(account_value,columns=['desc','value','currency','account_id'])
accountSet=accountSet.set_index(['desc']).transpose()
InitMarginReq=float(accountSet['InitMarginReq'].value)
InitMarginReqC=float(accountSet['InitMarginReq-C'].value)
InitMarginReqS=float(accountSet['InitMarginReq-S'].value)
MaintMarginReq=float(accountSet['MaintMarginReq'].value)
MaintMarginReqC=float(accountSet['MaintMarginReq-C'].value)
MaintMarginReqS=float(accountSet['MaintMarginReq-S'].value)

FullMaintMarginReq=float(accountSet['FullMaintMarginReq'].value)
FullMaintMarginReqC=float(accountSet['FullMaintMarginReq-C'].value)
FullMaintMarginReqS=float(accountSet['FullMaintMarginReq-S'].value)
AvailableFunds=float(accountSet['AvailableFunds'].value)
AvailableFundsC=float(accountSet['AvailableFunds-C'].value)
AvailableFundsS=float(accountSet['AvailableFunds-S'].value)
NetLiquidation=float(accountSet['NetLiquidation'].value)
NetLiquidationC=float(accountSet['NetLiquidation-C'].value)
NetLiquidationS=float(accountSet['NetLiquidation-S'].value)
im=abs(orderstates.initMargin.astype(float)-InitMarginReq)
mm=abs(orderstates.maintMargin.astype(float)-MaintMarginReq)
print 'InitMarginReq',InitMarginReq,'MaintMarginReq',MaintMarginReq
print im,
print mm
dic=pd.read_csv('\Users\Hidemi\Documents\GitHub\multi-tsdp\data\systems\dictionary.csv', index_col='ibsym')
dic['InitMarginReq_IB']=im
dic['MaintMarginReq_IB']=mm
dic.reset_index().to_csv('dictionary.csv', index=False)

'''
contract = Contract()
contract.symbol = "ES"
contract.secType = "FUT"
contract.exchange = "GLOBEX"
contract.currency = "USD"
contract.expiry = "201803"

order = Order()
order.action = 'BUY'
#order.lmtPrice = 140
order.orderType = 'MKT'
order.totalQuantity = 1
#order.algoStrategy = "AD"
order.tif = 'DAY'
order.whatIf = True

callback = IBWrapper()
client=IBclient(callback, port=7496, clientid=0)
(account_value, portfolio_data)=client.get_IB_account_data()
client.disconnect()
accountSet=pd.DataFrame(account_value,columns=['desc','value','currency','account_id'])
accountSet=accountSet.set_index(['desc']).transpose()
InitMarginReq=float(accountSet['InitMarginReq'].value)
InitMarginReqC=float(accountSet['InitMarginReq-C'].value)
InitMarginReqS=float(accountSet['InitMarginReq-S'].value)
FullMaintMarginReq=float(accountSet['FullMaintMarginReq'].value)
FullMaintMarginReqC=float(accountSet['FullMaintMarginReq-C'].value)
FullMaintMarginReqS=float(accountSet['FullMaintMarginReq-S'].value)

place_orders(contract, order, 0)
'''