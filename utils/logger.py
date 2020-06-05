# -*- coding: utf-8 -*-

import constants
#custom print fuction


def clear_log():
   open('logs.txt', 'w').close()
def log(message):
    
    if(constants.is_coare == 1):
        with open('logs.txt', 'a') as f:
            print(message, file=f)
    else:
        print(message)