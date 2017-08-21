"""IB Wrapper/Client"""
import time
import logging
import swigibpy

from django.conf import settings

import .ibconfig
import .errors


logger = logging.getLogger(__name__)


class IB_wrapper(swigibpy.EWrapper):
    """Callback object passed to TWS, these functions will be called directly by TWS."""
    def __init__(self, *args, **kwargs):
        self.cleanup()
        super(swigibpy.EWrapper, self).__init__(self, *args, **kwargs)

    def cleanup(self):
        """Clean errors"""
        self.flag_iserror = False
        self.error_msg = ''
        self.data_curent_time = 0
        self.data = None
    
    def set_error(self, message):
        """Raise an error"""
        self.flag_iserror = True
        self.error_msg = message

    def error(self, error_id, error_code, error_string):
        """Error handling"""
        """
        Here are some typical IB errors
        INFO: 2107, 2106
        WARNING 326 - can't connect as already connected
        CRITICAL: 502, 504 can't connect to TWS.
            200 no security definition found
            162 no trades
        """
        # Any errors not on this list we just treat as information
        ERRORS_TO_TRIGGER = [201, 103, 502, 504, 509, 200, 162, 420, 2105, 1100, 
                             478, 201, 399]

        if error_code in ERRORS_TO_TRIGGER:
            message = "IB error id {id} errorcode {error_code} string {error_string}".format(
                id=error_id, error_code=error_code, error_string=error_string)
            print(message)
            self.set_error(message)

    def currentTime(self, current_time):
        self.data_current_time = current_time

    ### stuff we don't use
    def nextValidId(self, orderId):
        pass

    def managedAccounts(self, openOrderEnd):
        pass

    def historicalData(self, req_id, date, open, high, low, close, volume,
                       bar_count, wap, has_gaps):
        data = (req_id, date, open, high, low, close, volume,
                bar_count, wap, has_gaps)
        prefix = "IB Wrapper -> historicalData()"
        logger.debug("{prefix} raw data: {data}.".format(prefix=prefix, data=data))
        if date[:8] == 'finished':
            logger.debug("{}: request complete".format(prefix))
        else:
            date = datetime.strptime(date, "%Y%m%d").strftime("%d %b %Y")
            logger.debug("{} {} - Open: {}, High: {}, Low: {}, Close: "
                        "{}, Volume: {}").format(
                            prefix, date, open, high, low, close, volume))
        self.data = data

class IB_client(object):
    """IB CLient"""
    def __init__(self, wrapper):
        if not (settings.IB_HOST and settings.IB_PORT and settings.IB_CLIENT_ID):
            raise ValueError('Interactive Broker was not configured properly. '
                             'Please check off IB_HOST, IB_PORT and IB_CLIENT_ID '
                             'values in the settings.')
        self.wrapper = wrapper
        self.tws = swigibpy.EPosixClientSocket(wrapper, reconnect_auto=True)
        self.tws.eConnect(settings.IB_HOST, settings.IB_PORT, settings.IB_CLIENT_ID)

    def get_historical_data(self, ticker_id, exchange, symbol, sec_type, currency, 
                            end_datetime, duration, bar_size_settings, what_to_show, 
                            user_rth=0, format_date=1, tag_value_list=None):
        """Get historical data"""
        # left for an instance:
        # self.get_historical_data(ticker_id, 'SMART', 'GOOG', 'STK', 'USD', 
        #                          datetime.today(), '1 W', '1 day', 'TRADES')
        contract = swigibpy.Contract()
        contract.exchange = exchange
        contract.symbol = symbol
        contract.secType = sec_type
        contract.currency = currency
        logger.debug('IB Client -> reqHistoricalData() was started with args {}.'.format(
            (ticker_id, contract, end_datetime.strftime("%Y%m%d %H:%M:%S %Z"),
             duration, bar_size_settings, what_to_show, user_rth, format_date, 
             tag_value_list)))
        self.tws.reqHistoricalData(ticker_id, contract, 
            end_datetime.strftime("%Y%m%d %H:%M:%S %Z"), duration, bar_size_settings, 
            what_to_show, user_rth, format_date, tag_value_list)
        self.loop()
        if not self.wrapper.data:
            logger.debug('IB Client -> Error: client got empty result')
        return self.wrapper.data

    def loop(self):
        """Main client's loop"""
        logger.debug("IB Client -> main loop was started")
        self.wrapper.cleanup()
        start_time = time.time()
        while True:
            if self.wrapper.data_current_time:
                # finished
                break
            if self.wrapper.flag_iserror:
                message = "IB CLient -> Error: %s" % self.wrapper.error_msg
                logger.debug(message)
                raise errors.IBClientError(message)
            if (time.time() - start_time) > ibconfig.MAX_WAIT_SECONDS:
                message = "IB CLient -> Error: timeout"
                logger.debug(message)
                raise errors.IBClientError(message)

