"""
    This is a simple module for doing arithmetic with log probabilities.
    This module depends on having NumPy installed and available.
    
    Only addition, subtraction, multiplication and division are implemented.
    Inputs and outputs for the arithmetic functions are log probabilities.
    Conversion functions to get to and from log/normal space are also available.
    
    Example:
    
    If p and q are probabilities, then x and y are representations in log space:
        x = log(p) and y = log(q)  <==>  p = e^x and q = e^y
    To multiply two probabilities, r = p*q, it's easy to stay in the log space:
        w = log(r) = log(p*q) = log(e^x * e^y) = log(e^(x + y)) = x + y
    
    So the function multiply_log_weights assumes you already have x and y, the
    log space representations of some probabilities, and returns the log space 
    result of multiplying the probabilities, i.e. adding the log space values.
    To get back to normal space, just use convert_logweight_to_standard(w).
    If you don't have x to start with, but only p, you gan get it using
    convert_standard_to_logweight(p).
"""


## functions for converting between log space and normal space

def convert_standard_to_logweight(standardweight):
    """
        Takes a normal probability and converts it to log space.
    """
    import numpy as np
    return np.log(standardweight)

def convert_logweight_to_standard(logweight):
    """
        Takes a log probability and converts it to a normal one.
    """
    import numpy as np
    return np.exp(logweight)


## functions for doing arithmetic while staying in log space

def multiply_log_weights(x,y):
    """
        Multiply two probabilities in log space.
    """
    return x + y

def divide_log_weights(x,y):
    """
        Divide the first probability by the second in log space.
    """
    return x - y

def add_log_weights(x,y):
    """
        Add two probabilities in log space.
        This is a more numerically stable version of log(e^x + e^y).
    """
    if y > x:
        y,x = x,y
    import numpy as np
    return x + np.log1p(np.exp(y-x))

def subtract_log_weights(x,y,warning=True):
    """
        Subtract the second probability from the first  in log space.
        This is a more numerically stable version of log(e^x - e^y).
        Warnings result when trying to represent negative numbers in log space.
        They are enabled by default.
    """
    if y > x:
        if warning is True: # sometimes for testing it's useful to get Nans out rather than an exception, but an exception is the default
            raise Exception('Negative probability')
            print 'You are subtracting a bigger number from a smaller one; trying to represent a negative number in log-space is an indeterminate problem'
    import numpy as np
    return x + np.log1p(-np.exp(y-x))
