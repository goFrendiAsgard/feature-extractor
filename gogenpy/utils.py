'''
Created on Nov 10, 2012

@author: gofrendi
'''

import random, math
from sys import stdout

randomizer = random.Random(10)

def sin(num):
    return math.sin(num)

def cos(num):
    return math.cos(num)

def mod(num):
    return math.fmod(num)

def factorial(num):
    return math.factorial(num)

def plus(num_1, num_2):
    return num_1 + num_2

def minus(num_1, num_2):
    return num_1 - num_2

def multiply(num_1, num_2):
    return num_1 * num_2

def divide(num_1, num_2):
    return num_1/num_2

def power(num_1, num_2):
    return num_1 ** num_2

def sqr(num):
    return num ** 2

def sqrt(num):
    return num ** 0.5

def write(text):
    # clear the line
    stdout.write("\r%s" % (' '*80))
    stdout.flush()
    # write new thing
    stdout.write("\r%s" % (text))
    stdout.flush()


def bin_to_dec(binary):
    '''
    decimal form of binary
    '''
    return int(binary,2)

def dec_to_bin(decimal):
    '''
    binary form of decimal
    '''
    return str(bin(decimal)[2:])

def bin_digit_needed(decimal):
    '''
    binary digit needed to represent decimal number
    '''
    return len(dec_to_bin(decimal))

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def execute(expr, record, variables):
    '''
    execute string as python program, give tuple(result, error) as return value
    '''
    result = 0
    error = False    
    # get result and error state
    try:
        sandbox={}
        exec ('from gogenpy.utils import *') in sandbox
        # initialize features
        for i in xrange(len(variables)):
            feature = variables[i]
            exec(feature+' = '+str(record[i])) in sandbox 
        # execute expr, and get the result       
        exec('__result = '+expr) in sandbox                      
        result = float(sandbox['__result'])
    except:
        error = True
    return result, error


