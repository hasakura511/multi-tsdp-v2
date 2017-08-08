# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 07:20:50 2016

@author: Hidemi
"""

from ib.opt import Connection

def print_message_from_ib(msg):
    print(msg)
#    return msg

#def main():
conn = Connection.create(port=7496, clientId=100)
conn.registerAll(print_message_from_ib)
conn.connect()

#In future blog posts, this is where we'll write code that actually does
#something useful, like place orders, get real-time prices, etc.

import time
time.sleep(1) #Simply to give the program time to print messages sent from IB
conn.disconnect()

#if __name__ == "__main__": main()