#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 19:43:52 2017

@author: hidemiasakura
"""

import calendar
import os
from os.path import isfile
import re
import time
import math
import json
import datetime
import numpy as np
from datetime import datetime as dt
from pytz import timezone
from tzlocal import get_localzone
import sqlite3
import pandas as pd
import sys 

dbPath = './data/systems/dictionary.sqlite3'



def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def getBackendDB():
    global dbPath
    readConn = sqlite3.connect(dbPath)
    return readConn
 

reload(sys)  
sys.setdefaultencoding('utf8')
db=getBackendDB()
dictionary = pd.read_csv('./data/systems/dictionary.csv')
dictionary.to_sql(name='dictionary', if_exists='replace', con=db, index=True, index_label='ID')
