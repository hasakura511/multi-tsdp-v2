#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 19:21:57 2018

board

inputs
board layout

output 
numbers and their parent systems

@author: hidemiasakura
"""
import numpy as np
old_board_config= [
  {
    "color": "pink",
    "position": "left",
    "display": "Previous (1 day)",
    "id": "PREVIOUS_1_DAY"
  },
  {
    "color": "indigo",
    "position": "left",
    "display": "Anti-Previous (1 day)",
    "id": "ANTI_PREVIOUS_1_DAY"
  },
  {
    "color": "yellow",
    "position": "right",
    "display": "Previous (5 days)",
    "id": "PREVIOUS_5_DAYS"
  },
  {
    "color": "black",
    "position": "top",
    "display": "Risk Off",
    "id": "RISK_OFF"
  },
  {
    "color": "red",
    "position": "top",
    "display": "Risk On",
    "id": "RISK_ON"
  },
  {
    "color": "#f8cd80 ",
    "position": "bottom",
    "display": "Lowest Eq.",
    "id": "LOWEST_EQ"
  },
  {
    "color": "#0049c1 ",
    "position": "bottom",
    "display": "Highest Eq.",
    "id": "HIGHEST_EQ"
  },
  {
    "color": "#c25de3 ",
    "position": "bottom",
    "display": "Anti-HE",
    "id": "ANTI_HE"
  },
  {
    "color": "#8ec54e ",
    "position": "bottom",
    "display": "Anti-50",
    "id": "ANTI_50"
  },
  {
    "color": "#f49535 ",
    "position": "bottom",
    "display": "Seasonality",
    "id": "SEASONALITY"
  },
  {
    "color": "#3fa3e7 ",
    "position": "bottom",
    "display": "Anti-Sea",
    "id": "ANTI_SEA"
  },
  {
    "color": "transparent",
    "position": "",
    "id": "BLANK"
  }
]

board_config= [
  {
    "color": "pink",
    "position": "left",
    "display": "Previous (1 day)",
    "id": "PREVIOUS_1"
  },
  {
    "color": "indigo",
    "position": "left",
    "display": "Anti-Previous (1 day)",
    "id": "ANTI_PREVIOUS_1"
  },
  {
    "color": "yellow",
    "position": "right",
    "display": "Previous (5 days)",
    "id": "PREVIOUS_5"
  },
  {
    "color": "black",
    "position": "top",
    "display": "Excess Liquidity",
    "id": "EXCESS"
  },
  {
    "color": "red",
    "position": "top",
    "display": "Risk On",
    "id": "RISKON"
  },
  {
    "color": "#f8cd80 ",
    "position": "bottom",
    "display": "TREND",
    "id": "ZZTREND"
  },
  {
    "color": "#0049c1 ",
    "position": "bottom",
    "display": "MODE",
    "id": "ZZMODE"
  },
  {
    "color": "#c25de3 ",
    "position": "bottom",
    "display": "Anti-Adjusted Seasonality",
    "id": "ANTI_ADJSEASONALITY"
  },
  {
    "color": "#8ec54e ",
    "position": "bottom",
    "display": "Anti-TREND",
    "id": "ANTI_ZZTREND"
  },
  {
    "color": "#f49535 ",
    "position": "bottom",
    "display": "Seasonality",
    "id": "SEASONALITY"
  },
  {
    "color": "#3fa3e7 ",
    "position": "bottom",
    "display": "Anti-Seasonality",
    "id": "ANTI_SEASONALITY"
  },
  {
    "color": "transparent",
    "position": "",
    "id": "BLANK"
  }
]
'''
top possible values are
2,3,4,6, where if it's 2 then the bottom must be 6. 2*6=12,
 since we are fixing the width to 12. so (2,6), where (top,bottom) where 2*6=12
possible (top,bottom) (1*12) (2*6), (3*4), (4*3), (6*2)

left/right max is 6
'''

class Board(object):
    def __init__(self):
        self.side_min=3
        self.side_max=6
        self.top_max=6
        self.bottom_max=12
        self.bottom=[]
        self.top=[]
        self.left=[]
        self.right=[]
        self.board_dict={}
        
    def create(self, config, verbose=False):
        
        for parent in config:
            #print(parent)
            if parent['position'] != "":
                position = getattr(self, parent['position'])
                position.append(parent['id'])
        
        self.len_bottom=len(self.bottom)
        self.len_top=len(self.top)
        self.len_left=len(self.left)
        self.len_right=len(self.right)
        
        self.height = max(self.len_left, self.len_right)+1
        if self.len_top*self.len_bottom==0:
            self.width = max(self.len_top, self.len_bottom)
            new_top=self.top
            new_bottom=self.len_bottom
        else:
            self.width = self.len_top*self.len_bottom
            new_top=self.top*self.len_bottom
            new_bottom=[]
            for b in range(self.len_bottom):
                for t in range(self.len_top):
                    new_bottom.append(self.bottom[b])
                    
        len_new_bottom=len(new_bottom)
        len_new_top=len(new_top)
        if verbose:
            print('bottom\n', self.bottom, 'top\n', self.top, 'left\n', self.left,
                  'right\n', self.right)
            print('top', len_new_top, new_top)
            print('bottom',len_new_bottom, new_bottom)
            print('height', self.height, 'width', self.width)
        self.num_tiles =self.height * self.width
        board=np.empty((self.height,self.width), dtype=np.object_)
        board.fill([])
        board = np.frompyfunc(list,1,1)(board)
        #self.board=board
        #append left and right parents
        for h in range(self.height):
            
            for w in range(self.width):
                if h<self.len_left:
                    board[h,w].append('signals_'+self.left[h])
                    #print(h,w,board[h,w])
                    
            for w in range(self.width):
                if h<self.len_right:
                    board[h,w].append('signals_'+self.right[h])
                    #print(h,w,board[h,w])
                
        #append top and bottom
        i=0
        for w in range(self.width):
            for h in range(self.height):
                    i+=1
                    if w<len_new_top:
                        board[h,w].append('signals_'+new_top[w])
                    if w<len_new_bottom:
                        board[h,w].append('signals_'+new_bottom[w])
                    if verbose:
                        print(i,h,w,board[h,w])
                    self.board_dict[i]=board[h,w]

        self.board=board
        return self
        
'''
b=Board()
b.create(board_config, verbose=True)
'''