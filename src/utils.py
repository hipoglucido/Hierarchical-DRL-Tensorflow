# -*- coding: utf-8 -*-
import os
import time

def get_timestamp():
      return time.strftime("%Ss%Hh%Mm%b%d")


class Logger:
    # Using the existing Logger library was throwing some issues when developing on Spyder
    def __init__(self):
        return
     
    def print(self, text, newline):
        if newline:
            pre = "\n"
        else:
            pre = ""
        output = get_timestamp() + " -" + self.mode +"- " +pre+ text
            
        print(output)
         
    def warn(self, text, newline = False):
        self.mode = 'WARNING'
        self.print(text, newline)
         
    def info(self, text, newline = False):
        self.mode = 'info'
        self.print(text, newline)
