# -*- coding: utf-8 -*-

import constants
#custom print fuction


def clear_log():
   file_name = "logs_" + constants.VERSION + "_" + constants.ITERATION + ".txt"
   open(file_name, 'w+').close()
   
def log(message):
    file_name = "logs_" + constants.VERSION + "_" + constants.ITERATION + ".txt"
    if(constants.is_coare == 1):
        with open(file_name, 'a+') as f:
            print(message, file=f)
    else:
        print(message)