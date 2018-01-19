"""
Copyright (C) 2018 Global Systems Management. All rights reserved.  
"""

""" Package implementing the Nimbus API for the GSM System """

VERSION = {
    'major': 0,
    'minor': 0,
    'micro': 0}


def get_version_string():
    version = '{major}.{minor}.{micro}'.format(**VERSION)
    return version

__version__ = get_version_string()
