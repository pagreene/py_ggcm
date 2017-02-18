from re import *

SCI = "[+-]?\d+\.\d+[eE][-+]?\d+"
DEC = "[+-]?\d+\.\d+"
INT = "[+-]?\d+"

num_patt_dict = {'dec':DEC, 'DEC':DEC,
                 'sci':SCI, 'SCI':SCI,
                 'int':INT, 'INT':INT}

any_num_patt = ".*?(%(sci)s|%(dec)s|%(int)s)" % num_patt_dict

def find(pattern, string, flags = 0):
    result = search(pattern, string, flags)
    if result != None:
        return True
    else:
        return False

def findNum(string, numtype = None):
    '''
    Method to determine if there are of the standard number formats are present 
    in a string. If numtype is specified, only numbers of that type will cause 
    the function to return True.
    '''
    if numtype != None:
        patt = num_patt_dict[numtype]
    else:
        patt = any_num_patt
    
    return find(patt, string)

def retrieve(pattern, string, flags = 0):
    re_out = compile(pattern, flags).findall(string)
    if not len(re_out):
        raise error("No matches for %s in:\n%s" % (pattern, string))
    return re_out

def retrieveNums(string, multi = True, numtype = None):
    '''
    Method to retrieve a number of set of numbers from a string. The values of
    those numbers will be returned in the order found.
    '''
    # If the user specified a particular type of number (int, sci, or dec)
    if numtype != None:
        patt = num_patt_dict[numtype]
    else:
        patt = any_num_patt
    
    # Get the numbers
    re_n_list = retrieve(patt % num_patt_dict, string)
    n_list = []
    for n in re_n_list:
        if numtype == 'sci' or numtype == 'SCI' or find(SCI, n):
            n_list.append(float(n))
        elif numtype == 'dec' or numtype == 'DEC' or find(DEC, n):
            n_list.append(float(n))
        else:
            n_list.append(int(n))
    
    return n_list

def replace(patt, string, new_str = "", new_fmt = "", reorder = None, flags = 0):
    '''
    A regular expression based method to replace a piece of `string`. There are two
    main options:
    - (Simple) If `new_str` is specified, the match for `patt` will be replaced by 
    `new_str`.
    
    - (Not Simple) If `new_fmt` is specified, and for every %s in `new_fmt` there is 
    exactly one parenthetical capture in `patt`, then the items captured from string 
    will be placed in `new_fmt`, reordered according to `reorder`, which may be either 
    a list or a dict, where the index/key applies to the order in `patt` and the 
    values apply to the order in `new_fmt`. The result will replace piece matched by 
    `patt.`
    
    If neither `new_str` nor `new_fmt` are specified, `patt` will be replace with an empty
    string (i.e. removed).
    
    `flags` may be specified as for any other regular expression.
    '''
    # If there a format was given, get the data for the format
    if new_fmt != '':
        old = search(patt.replace(")", "").replace("(", ""), string, flags)
        if not old:
            return string
        old = old.group(0)
        
        finds = search(patt, string, flags).groups()
        
        # If the results are reordered in the format
        if reorder != None:
            if isinstance(reorder, dict):
                iters = reorder.iteritems()
            elif isinstance(reorder, list) or isinstance(reorder, tuple):
                iters = enumerate(reorder)
            else:
                raise TypeError("Reorder must be an iterable (list, tuple, or dict), not %s" % type(reorder))
            
            d_in = {}
            for i0, i1 in iters:
                d_in[i1] = finds[i0] # The dict will automatically sort by key
            
            inputs = tuple(d_in.values())
        else:
            inputs = tuple(finds)
                    
        new = new_fmt % inputs
    else:
        # Get the entire section of string to be replaced
        old = search(patt, string, flags)
        if not old:
            return string
        old = old.group(0)
        new = new_str
    
    return string.replace(old, new)