#======================================================================================
# Directory object
# This object is designed to make handling of directory trees much easier. Although it
# would seem like someone would already have a feature like this available, I have not
# been able to find such a feature, so I made one myself.
#======================================================================================
import os
import shutil
import My_Re
from ggcm_exceptions import ggcm_exception
from ggcm_logger import Logged_Object
from multiprocessing import Lock as MP_Lock
from threading import RLock

_HOME = os.environ['HOME']
_PWD = os.environ['PWD']

def __fixFullPath(path):
    '''
    Function to regulate the formatting of paths.
    '''
    path = path.rstrip('/')
    
    # Creat the full, path, and name attributes.
    pl = path.partition('/')
    if pl[0] == '':
        full = path
    elif pl[0] == '..':
        full = _PWD.rpartition('/')[0] + pl[1] + pl[2]
    elif pl[0] == '~':
        full = _HOME + pl[1] + pl[2]
    elif pl[0] == '.':
        full = _PWD + pl[1] + pl[2]
    elif pl[0][0].isalpha():
        if pl[2] != '':
            full = _PWD + pl[0] + pl[1] + pl[2]
        else:
            full = _PWD + '/' + pl[0]
    else:
        raise Exception("Unkown input format for path")
    return full

_make_proc_lock = MP_Lock()
class Directory(Logged_Object):
    '''
    This object is the logical representation of a directory. It provides several
    features including a standardization of path formats. It also provides APIs for
    searching the contents of a directory. It also allows for easy navigation and
    modification of directory trees.
    
    This object uses a global registry of Directory objects which is thread safe but
    cannot be made multiprocessing safe. Use care when creating or retrieving 
    directories in a multiprocessing context.
    '''
    def __init__(self, path, parent = None):
        class InitError(Exception):
            pass
    
        # I am making the "__parent" attribute private so that a pointer to the parent
        # object must be retrieved if the user wants to use it.
        self.__parent = parent
        
        pl = path.split('/')
        if self.__parent:
            if len(pl) > 1:
                if pl[0] != '' or len(pl) > 2:
                    raise InitError('Attempting to create a subdirectory in a subdirectory.' +
                                    'Create the subdirectory first, then from that object' +
                                    'create the sub-subdirectory.')
                else:
                    self.name = pl[1]
            else:
                self.name = pl[0]
            self.full = "%s/%s" % (self.__parent.full, self.name)
            self.path = self.__parent.full
        else:
            # The values past to this constructor should come from either a parent
            # directory object or the "getDirectory" function, so the path can be
            # assumed to be a consistant full path.
            self.full = path
            self.path, _, self.name = self.full.rpartition('/')
        
        # Check to see if the directory is a root directory. This information
        # affects the way 'getParent' runs.
        if self.path == '':
            self.__is_root = True
            self.path = None
        else:
            self.__is_root = False
        
        # Directories are logged objects.
        Logged_Object.__init__(self, self.name)
        self.debug("Final attrs - name: '%s' path: '%s' full: '%s'" % (self.name, self.path, self.full))
        
        # This is a dictionary the directory object uses to keep track of it's known
        # subdirectories. It is used and modified through 'getSubDirectory' and
        # 'makeSubDirectory'.
        self.__sub_dirs = {}
        
        # Keep track of whether this is a temporary directory or not. Temp
        # directories are removed when the object is dereferenced. As such, temp
        # directories can only contain other temporary directories.
        self.__is_temp = False
        
        # Because references to this object may be held even after it has been
        # deleted, although this should be avoided, this allows damage from such
        # and incident to be minimized.
        self._deleted = False
        return
    
    def __del__(self):
        '''
        Method executed when object is cleaned up
        '''
        if not self.__is_temp:
            return
        
        _registry_thread_lock.acquire()
        try:
            for sub_dir in self.__sub_dirs.itervalues():
                sub_dir._deleted = True
                del _directory_registry[sub_dir.full]
            
            self.__sub_dirs.clear()
        finally:
            _registry_thread_lock.release()
        
        if not self._deleted:
            shutil.rmtree(self.full)
        return
    
    def isTemp(self):
        '''
        API to determine if the directory is a temp directory
        '''
        return self.__is_temp
    
    def clear(self):
        '''
        API to clear a directory
        '''
        if self._deleted:
            self.WARNING("Attempting to clear deleted directory")
            return
            
        file_list = os.listdir(self.full)
        for filename in file_list:
            os.remove(self.full + "/" + filename)
        return
    
    def make(self, replace = False, temp = False):
        '''
        API to make the represented directory. If there is a conflict, the name
        is modified with a "_#". The number is incremented up until there is no 
        longer conflict. The (possibly new) directory name is returned.
        '''
        if self._deleted:
            self.WARNING("Attempting to make deleted directory")
            return
        
        # When making directories, because there is interaction with the file 
        # system, we need to acquire a multiprocessing lock. In general this
        # class is not multiprocessing safe, thinks could go horribly wrong 
        # without this.
        _make_proc_lock.acquire()
        self.debug("making new directory (lock acquired)")
        try:
            # Get the list of things in the parent directory
            dir_list = os.listdir(self.path)
            if replace:
                if self.name not in dir_list:
                    os.mkdir(self.full)
                else:
                    self.clear()
            else:
                dir_name = self.name
                i = 1
                while dir_name in dir_list:
                    if i == 1:
                        dir_name += "_1"
                    else:
                        dir_name = My_Re.replace("_\d+?$", dir_name, new_str = "_%d" % i)
                    i += 1
                self.name = dir_name
                self.full = self.path + "/" + self.name
                os.mkdir(self.full)
        finally:
            _make_proc_lock.release()
            
        parent = self.getParent()
        if temp or parent.isTemp():
            self.__is_temp = True
        
        self.info("Made new directory %s in %s (lock released)" % (self.name, self.path))
        return self.name
    
    def makeIfNeeded(self):
        '''
        API to make the represented directory if the directory does not already exist.
        '''
        if self._deleted:
            self.WARNING("Attempting to make deleted directory")
            return
        
        dirs = os.listdir(self.path)
        if self.name not in dirs:
            self.make(replace = False)
            self.debug("%s did not exist in %s, so I made it." % (self.name, self.path))
        return
    
    def find(self, fname):
        '''
        API for finding a file or subdirectory within the represented directory.
        'fname' is matched as a regular expression from the list of items within
        this directory. If ANY directory is found that matches, this function
        will return true. If you want to have something match only if the complete
        name is found, include "^" at the beginning and "$" at the end of the
        input, such as "^target$".
        '''
        if self._deleted:
            self.WARNING("Attempting to search through deleted directory")
            return
        
        l = os.listdir(self.full)
        
        for item in l:
            if My_Re.find(fname, item):
                return True
        return False
            
    def getFullName(self, name, no_list = False, only_one = False):
        '''
        API for getting the full name of a directory using a regular expression (name).
        By default, a list is returned, as more than one file may match. 
        
        If no_list is True, only the first item found (not a list) will be returned. 
        
        If only_one is True, if more than one item is found, this function will except;
        otherwise, it will return the match (not a list).
        '''
        if self._deleted:
            self.WARNING("Attempting to get name from deleted directory")
            return
        
        l = os.listdir(self.full)
        
        err_msg_fmt = "Multiple items (%s) matched %s in directory %s."
        
        ret = []
        for item in l:
            if My_Re.find(name, item):
                if no_list:
                    return item
                else:
                    ret.append(item)
                    
        if only_one:
            if len(ret) > 1:
                raise ggcm_exception(err_msg_fmt % (str(ret), name, self.full), 'misc catch')
            elif len(ret) == 1:
                return ret[0]
            else:
                class NameNotFound(Exception):
                    pass
                raise NameNotFound("Could not find name from pattern \"%s\"" % name)
        else:
            return ret
    
    def _getParent(self):
        '''
        This is the internal API that does the actual work of getting the parent. 
        "__registry_thread_lock" must be acquired before using this and released
        afterwards. If the directory is a root directory, this function will return
        None.
        '''
        if self._deleted:
            self.WARNING("Can't get parent of deleted directory")
            return
        
        # If this is a root directory, it has no parent.
        if self.__is_root:
            return None
        
        # If this object already knows it's parent, just return the value
        if self.__parent != None:
            return self.__parent
        
        # Check if the parent directory exists already. If it does, pass a reference,
        # if it doesn't make it, stash it, and pass the reference.
        if self.path in _directory_registry.keys():
            self.__parent = _directory_registry[self.path]
        else:
            # I do not tell the newly created parent of it's child (to avoid
            # recursion). But because the 'getSubDirectory' method checks the
            # registry, it will find and store a reference to this object the 
            # first time 'getSubDirectory' is called.
            d = Directory(self.path)
            _directory_registry[self.path] = d
            self.__parent = d
        return self.__parent
    
    def getParent(self):
        '''
        API to get a pointer to the parent of this directory.
        '''
        _registry_thread_lock.acquire()
        try:
            ret = self._getParent()
        finally:
            _registry_thread_lock.release()
        return ret
    
    def getSubDirectory(self, name):
        '''
        API to create a sub directory object in the directory represented by this 
        object. If the directory does not already exist it will not necessarily be
        created. A pointer to the new object will be created, and "makeIfNeeded" or
        "make" may be called if the you want to create the actual directory.
        '''
        if self._deleted:
            self.WARNING("Can't get sub directory of deleted directory")
            return
        
        # If the sub directory object has already been created, just pass the pointer.
        if name in self.__sub_dirs.keys():
            return self.__sub_dirs[name]
        
        _registry_thread_lock.acquire()
        try:
            full = "%s/%s" % (self.full, name)
            if full in _directory_registry.keys():
                sub_dir = _directory_registry[full]
                sub_dir._getParent(self.full)
            else:
                sub_dir = Directory(name, self)
                _directory_registry[sub_dir.full] = sub_dir
                self.__sub_dirs[name] = sub_dir
        finally:
            _registry_thread_lock.release()
        return sub_dir

    def makeSubDirectory(self, name,  persistent = False, temp = False):
        '''
        This API behaves exactly the same as getSubDirectory except that it will try to
        create the actual directory in the file system. By default it only creates the
        directory if it does not already exist. Setting 'persistent' to True will create
        append a number to the end of the directory name, incrementing the number until
        there is no conflict.
        '''
        sub_dir = self.getSubDirectory(name)
        if not persistent:
            sub_dir.makeIfNeeded()
        else:
            sub_dir.make(temp = temp)
        return sub_dir

def register(Obj, key, registry, lock):
    '''
    This is a generalized method for modifying a registry, because it's the same 
    thing every time, and I'm using it rather a lot. When something is called 
    from a registry, if it already exists, a reference to the existing object is
    returned. If it does not exist, the a new object is created, registered, and
    a reference to the new object is returned.
    
    INPUTS:
    obj      -- the object (not an instantiation of the object) being registered
    key      -- whatever information is used to define the object (tuple, string, or number)
    regsitry -- a dictionary that is used to keep track of the objects
    lock     -- a lock object that is used to synchronize access to the registry
    '''
    # If the object is not in the registry, make a new one and pass it, otherwise
    # pass the old one.
    lock.acquire()
    try:
        if key in registry.keys():
            d = registry[key]
        else:
            if isinstance(key, tuple):
                d = Obj(*key)
            else:
                d = Obj(key)
            registry[key] = d
    finally:
        lock.release()
    
    return d

_directory_registry = {}
_registry_thread_lock = RLock()
def getDirectory(full_path):
    '''
    Function to retrieve a pointer to a Directory object. If the object has already been
    created, then this will simply return a reference to it. Otherwise, it will create 
    a new object, store it in the registry, and then pass the reference to it.
    '''
    # Make sure the path is correctly formatted
    full_path = __fixFullPath(full_path)
    
    return register(Directory, full_path, _directory_registry, _registry_thread_lock)


HOME = getDirectory(_HOME)
PWD = getDirectory(_PWD)
LOCAL = getDirectory(os.path.realpath(__file__)).getParent()
