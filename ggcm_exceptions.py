#================================================================
# This file contains the ggcm_exceptions. All exceptions should
# be derived from the ggcm_exception so that exceptions we raise
# can be distinguished from Python exceptions (e.g. syntax errors)
#================================================================

from datetime import datetime
from ggcm_logger import logging, LOG_DIR
import os
import sys

exception_logger = logging.getLogger("exceptions")
exception_handler = logging.FileHandler(LOG_DIR + "/exceptions.log")
exception_logger.addHandler(exception_handler)

if hasattr(sys, 'frozen'): #support for py2exe
    _srcfile = "logging%s__init__%s" % (os.sep, __file__[-4:])
elif str.lower(__file__[-4:]) in ['.pyc', '.pyo']:
    _srcfile = __file__[:-4] + '.py'
else:
    _srcfile = __file__
_srcfile = os.path.normcase(_srcfile)

# Purely esthetic. Used in the output to make boxes.
LONG_LINE = "\n=============================================================="

#================================================================
# GGCM_EXCEPTION: Base Class
#================================================================
class ggcm_exception(Exception):
    '''
    Simple exception class for a generic exception within the code.
    '''
    def __init__(self, description, err_type):
        Exception.__init__(self)
        exception_logger.exception(description)
        self.description = description
        self.type = err_type

    def __str__(self):
        string = LONG_LINE
        string += "\nEncountered %s at %s" % (self.type, str(datetime.now()))
        string += "\nDescription: %s" % self.description
        string += LONG_LINE
        return string
    
    def __findCaller(self):
        """
        Find the stack frame of the caller so that we can note the source
        file name, line number and function name.
        """
        f = logging.currentframe().f_back
        rv = "(unknown file)", 0, "(unknown function)"
        while hasattr(f, "f_code"):
            co = f.f_code
            filename = os.path.normcase(co.co_filename)
            if filename in (_srcfile, logging._srcfile): # This line is modified.
                f = f.f_back
                continue
            rv = (filename, f.f_lineno, co.co_name)
            break
        return rv

#================================================================
# INPUT_ERROR:
#    Input_Error (base) -- incorrect input
#    Type_Error (child) -- incorrect type for an input
#================================================================
class Input_Error(ggcm_exception):
    '''
    Exception type-class for input errors.
    '''
    def __init__(self, input_var, info):
        description = "Incorrect input for %s: %s" % (input_var, info)
        ggcm_exception.__init__(self, description, "Input Error")
            
class Type_Error(Input_Error):
    '''
    Exception for inputs that are 
    description = data_name + " has status %s
    '''
    def __init__(self, input_var, exp_type, act_type):
        info = "\nExpected Type: %s\n My_Received Type: %s" % (exp_type, act_type)
        Input_Error.__init__(self, input_var, info)

#================================================================
# EXEC_ERROR:
#================================================================
class Exec_Error(ggcm_exception):
    '''
    Exception designed for use with the execute function. This 
    exception will carry the output of the executed function up
    the chain for examination/record keeping.
    '''
    def __init__(self, command, output, note = None):
        desc = "Execution of '%s' failed" % str(command)
        if note:
            desc += "\nNote:\n%s" % note
        ggcm_exception.__init__(self, desc, "Execution Error")
        if isinstance(output, tuple) or isinstance(output, list):
            self.output = ""
            for item in output:
                self.output += item
        elif isinstance(output, str):
            self.output = output
        return
        
        
