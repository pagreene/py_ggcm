from multiprocessing import Lock as MP_Lock
from threading import RLock
import logging
import time
import os
import sys
from datetime import datetime

import My_Re

DEBUG = True

HOME = os.environ['HOME']

# Create log directory if it doesn't already exist
if 'logs' not in os.listdir(HOME):
    os.makedirs(HOME + "/logs")

# Create log subdirectory
LOG_DIR = HOME + datetime.now().strftime("/logs/%Y-%m-%d_%H-%M-%S")
os.makedirs(LOG_DIR)

print "Logs may be found in '%s'" % LOG_DIR

class mpStreamHandler(logging.StreamHandler):
    '''
    This is a less complex fix than bellow. If a log message is sent
    to the stream from a child process, then it may come out garbled
    with other messages. I could implement a process lock alongside
    the thread lock, however processes 
    '''
    def __init__(self, strm = None):
        if strm is None:
            strm = sys.stdout
        logging.StreamHandler.__init__(self, strm)
        self.lock = MP_Lock()
        

class mpFileHandler(logging.FileHandler):
    '''
    When a process is spawned, a copy of all (or at least all relevant)
    objects is passed to the spawned process. This is a copy, so if the
    original object is changed in the parent process, the object in the
    spawned process doesn't know. This causes a problem with locks. If a
    lock is held by in the parent process when the child process is
    spawned, then the lock will never be released in the child process,
    and the child will hang. To avoid this issue, when the lock is 
    acquired, the current pid is checked, and if it is not present in
    the __lock_dict, it is added to it, and that new lock is used.
    
    To prevent the output from the different processes getting all mucked
    up, the file that is written has '.<pid>' appended to it, so each
    process writes to a different file. Reading from the hundreds of logs
    that are the result of this is not desirable, so when this object is
    dereferenced or deleted by the original parent process, all the logs
    are read a new file is created with all the entries from the other 
    files in order according to their time stamps. 
    
    PLEASE NOTE, THAT SMALL DIFFERENCES IN TIMES BETWEEN PROCESSES ARE 
    NOT CONCLUSIVE; THAT IS TO SAY, IF AN EVENT IN ONE PROCESS IS SHOWN
    OCCURRING MILLISECONDS BEFORE AN EVENT IN ANOTHER PROCESS, THE ACTUAL
    SEQUENCE MAY BE REVERSED.
    
    Also, be careful that __del__ is not called before all child processes
    that may write to the log are completed, or else log output may be 
    left out. Safeguards against this may be added later.
    '''
    def __init__(self, log_name):
        fname = '%s/%s.log' % (LOG_DIR, log_name)
        self.__name = log_name
        self.__ppid = os.getpid()
        
        # Initialization from the inherited class
        logging.FileHandler.__init__(self, fname)
        
        # Set the default formatter
        self.formatter = self.mpFormatter('%(asctime)s: %(message)s')
        
        # The lock dictionary indexed by process id (pid)
        # Note: This will NOT contain all pids, only those
        # inherited from the parent as will as the child's.
        # Again, this is because only copies of this object
        # are passed to new processes.
        self.__lock_dict = {os.getpid():RLock()}
        
        # A list of all the processes that have had a file
        # openned. Just as with the above dictionary, this
        # list will NOT contain all pids. It is more of a
        # family tree than a census.
        self.__openned = [os.getpid()]
        return
   
    def mpFormatter(self, *args, **kwargs):
        '''
        I have to define this class because otherwise I wouldn't get microseconds.
        '''
        class mpFormatter(logging.Formatter):
            def formatTime(self, record, crap_that_i_ignore=None):
                ct = self.converter(record.created)
                t = time.strftime("%Y-%m-%d %H:%M:%S", ct)
                s = "%s,%06d" % (t, record.msecs*1000)
                return s
        return mpFormatter(*args, **kwargs)
 
    def setFormatter(self, fmtr):
        '''
        This API effectively converts the passed formatter object into an mpFormatter
        object. The difference is in the formatting of the time.
        '''
        self.formatter = self.mpFormatter(fmtr._fmt)
        return
    
    def __del__(self):
        '''
        Method called when object dereferenced (deleted).
        '''
        import os
        # If this is the parent process
        print self.__ppid, os.getpid()
        if self.__ppid == os.getpid():
            print 'consolidating log file for %s' % self.__name
            self.__consolidateFile()
        
        self.close()
        return

    def __consolidateFile(self):
        '''
        Internal API to consolidate the log file
        '''
        def read(fname):
            f = open(fname, 'r')
            ret = f.read()
            f.close()
            os.remove(fname)
            return ret
        
        # Get the entries from all the files
        entries = []
        log_names = os.listdir(LOG_DIR)
        for log_name in log_names:
            if My_Re.search(self.__name, log_name):
                pid = log_name.split('.')[:-1]
                log_str = read(log_name)
                re_entries = My_Re.split('(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})', log_str)
                for i in range(len(re_entries)/2):
                    msg = "%s: %s - %s" % (re_entries[2*i], pid, re_entries[2*i + 1])
                    entries.append(msg)
        
        # Sort the list (Python is beautiful)
        entries.sort()
        
        # Convert the entries list into a single string
        fstr = ""
        for entry in entries:
            fstr += entry[1]
        
        # Write the string to the log file
        f = open('%s/%s.log' % (LOG_DIR, self.__name), 'w')
        f.write(fstr)
        f.close()
        return
    
    def acquire(self):
        if os.getpid() not in self.__lock_dict:
            self.__lock_dict[os.getpid()] = RLock()
        self.__lock_dict[os.getpid()].acquire()
        return
    
    def release(self):
        self.__lock_dict[os.getpid()].release()
        return
    
    def _open(self):
        stream = open(self.baseFilename + ".%d" % os.getpid(), self.mode)
        return stream
    
    def emit(self, record):
        if os.getpid() not in self.__openned:
            self.stream = self._open()
            self.__openned.append(os.getpid())
        logging.FileHandler.emit(self, record)
        return

class mpFileHandler2(logging.FileHandler):
    '''
    Let's give this a shot (and kind of hope it fails spectacularly)
    '''
    def __init__(self, *args):
        logging.FileHandler.__init__(self, *args)
        self.lock = MP_Lock()

# Get root logger
LOGGER = logging.getLogger()
if DEBUG:
    LOGGER.setLevel(0)
else:
    LOGGER.setLevel(logging.INFO)

# Create Handlers
detailed = mpFileHandler2(LOG_DIR + '/detailed.log')
console = mpStreamHandler(sys.stdout)
sparce = logging.FileHandler(LOG_DIR + '/sparce.log')

# Set handler levels
detailed.setLevel(10)
console.setLevel(20)
sparce.setLevel(20)

# Create Formatters
d_fmtr = logging.Formatter("%(asctime)s: %(levelno)-2s name:%(name)-22s thread:%(thread)02d pid:%(process)-5d %(module)-15s line %(lineno)03d - %(message)s")
s_fmtr = logging.Formatter("%(message)s")

# Add formatters to Handlers
detailed.setFormatter(d_fmtr)
sparce.setFormatter(s_fmtr)
console.setFormatter(s_fmtr)

# Add handlers to logger
LOGGER.addHandler(detailed)
LOGGER.addHandler(console)
LOGGER.addHandler(sparce)
       
class Logged_Object(object):
    log_fmt = "%(asctime)s: %(levelname)-10s %(module)-15s line no. %(lineno)03d - %(message)s"
    def __init__(self, name, debug = False):
        # Define the log's name
        log_name = '%s.%s' % (self.__class__.__name__, name)
        
        # Define the new log
        self.__logger = logging.getLogger(name)
        
        self._isdebug = debug
        
        return
    
    def debug(self, msg):
        if not self._isdebug:
            return self.__logger.debug(msg)
        else:
            return self.info(msg)
    
    def info(self, msg):
        return self.__logger.info(msg)
    
    def WARNING(self, msg):
        return self.__logger.warning(msg)
    
    def ERROR(self, msg):
        return self.__logger.error(msg)
    
    def CRITICAL(self, msg):
        return self.__logger.critical(msg)
    
    def EXCEPTION(self, msg):
        return self.__logger.exception(msg)

