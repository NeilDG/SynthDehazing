# -*- coding: utf-8 -*-

import constants
#custom print fuction

file_name = "logs_" + constants.VERSION + "_" + constants.ITERATION + ".txt"

def clear_log():
   open(file_name, 'w+').close()
   
def log(message):
    if(constants.is_coare == 1):
        with open(file_name, 'a+') as f:
            print(message, file=f)
    else:
        print(message)