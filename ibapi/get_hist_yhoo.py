from yahoo_finance import Share
from pprint import pprint

yahoo = Share('ESH15.CME')
pprint(yahoo.get_historical('2015-04-25', '2016-02-29'))
