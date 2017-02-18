#================================================================
# This file contains a bunch of useful general tools/shortcuts
#================================================================
from time import sleep
from ggcm_exceptions import ggcm_exception
import My_Re
import os

from math import asin, atan, pi
from numpy.linalg import norm
def toSpherical(v):
    'Function to convert a 3d vector v into spherical coordinates.'
    r = norm(v)
    try:
        ph = asin(v[2]/r)
    except:
        print v, r
        raise
    if v[0] == 0:
        if v[1] == 0:
            th = 0
        elif v[1] > 0:
            th = pi/2
        else:
            th = 3*pi/2
    else:
        th = atan(v[1]/v[0])
        if v[0] > 0:
            if v[1] <= 0:
                th += 2*pi
        else: # if v[0] < 0
            th += pi
            
    return (r,th,ph)

def retryFunc(func, *args, **kwargs):
    '''
    Fuction that allows the function passed into it to be attempted multiple times.
    
    Keyword args:
    n_retries -- Number of times to retry. Default is 3.
    wait      -- Time to wait in between retries (in seconds). Default is 1 sec.
    ex_class  -- the class of error to wait for. Default is BaseException.
    ex_patt   -- a regular expression pattern searched for in the exception. 
                 If found, retry will occur, if not found, retry will not occur 
                 and exception will be raised.
    ex_func   -- a function to be run when exception occurs. Default is None
    '''
    defaults = {'n_retries':3, 'wait':1, 'ex_class':BaseException, 'ex_func':None, 'ex_patt':'.*'}
    values = {}
    for key, value in defaults.iteritems():
        if key in kwargs:
            values[key] = value
            del kwargs[key]
        else:
            values[key] = defaults[key]
    
    n_retries, wait, ex_class, ex_patt, ex_func = values.values()
        
    while n_retries + 1:
        try:
            ret = func(*args, **kwargs)
            return ret
        except ex_class, e:
            if My_Re.find(ex_patt, str(e)):
                n_retries -= 1
                ex_func()
                sleep(wait)
            else:
                raise
    else:
        raise ggcm_exception("After %d retries, %s failed to run without exception." % (n_retries, func.func_name), 'Retry Error')
