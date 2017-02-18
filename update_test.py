#! /usr/bin/env python2.7
# This is the file that contains the instructions to update the website.

from ggcm_simulation import Run
from datetime import datetime, timedelta
from time import sleep
import ggcm_cmaps as cm
import os
import sys
sys.tracebacklimit = 10

web_dir = "/home/pan56/py_ggcm/test_out"
A = 'Aurora'
M = 'Magnetosphere'

# Settings ----------------------------------------------------------------------------
# Before we start converting the outputs, we want to make sure the simulated system has
# settled down into a steadier state. This is called pre-run (thus, pr)
pr = 0.0

# Next, we choose how much time we want outputs for.
dur = 1

# Eventually, the simulation may be able to handle continuous input. When it is able to
# you will be able to run the simulation using data that's not available when this code
# begins. Set how much of that time you want here.
fut = 0

# Set the number of processes.
np = 4

# For testing, the simulation emulator may be used. This just copies one output into
# the target directory over and over again. The biggest time saver is not having to
# wait for the runme to finish.
fake = True 
if not fake:
    np = 2 # If not faking, don't hog all the cpus. Only allow 2 procs for plotting.

# Set the time step (and make a timedelta object with the same value.)
dt = 300 # seconds
DT = timedelta(seconds = dt)

# Turn the above time choices into datetime objects.
end_off_hour = datetime.utcnow() + timedelta(hours = fut)
end_time = end_off_hour.replace(second = 0, microsecond = 0, minute = 0)
start_time = end_time - timedelta(hours = dur)

# Put all these into a dict of simulation options. Here you may also select the spacial
# dimensions and the number of processes for each dimension (nx,ny,nz).
sim_opts = {}
sim_opts.update(start_time = start_time - timedelta(hours = pr), 
                end_time   = end_time + DT, 
                dims       = [-50, 250, 100, 100], 
                procs      = [2,1,1], 
                prerun     = pr,
                emulate    = fake)

# Function Defs -----------------------------------------------------------------------
def printDict(d, indent = ''):
    '''
    Recursive function to print the contents of a dictionary in a human-readable way.
    This is intended as a tool for debugging, although other uses may be found.
    '''
    s = ''
    for key, val in d.iteritems():
        if isinstance(val, dict):
            s += "%s%s:\n%s" % (indent, key, printDict(val, indent + '  '))
        elif isinstance(val, cm.cool.__class__):
            s += "%s%s: %s\n" % (indent, str(key), val.name)
        else:
            s += "%s%s: %s\n" % (indent, str(key), str(val))
    return s

def dJoin(*dicts):
    '''
    Simple function to join two dictionaries.
    '''
    ret = {}
    for d in dicts:
        ret.update(d)
    return ret

def convertToJS(key, item):
    '''
    Recursive function to turn 'item' into a JavaScript variable with name from 'key'. 
    If 'item' is a dict, than each key-item pair will be recursively converted into a 
    JavaScript variable, then added to a JavaScript object with name given by 'key'.
    '''
    ret = ''
    if isinstance(item, str):                               # String
        return "var %s = '%s';\n" % (key, item)
    elif isinstance(item, int) or isinstance(item, float):  # Number
        return "var %s = %s;\n" % (key, str(item))
    elif isinstance(item, datetime):                        # Datetime
        return "var %s = new Date('%s');\n" % (key, item.strftime('%Y/%m/%d %H:%M'))
    elif isinstance(item, list):                            # List (recursive)
        ret += 'var %s = new Array();\n' % key
        for val in item:
            ret += convertToJs('tmp_val', val)
            ret += '%s.push(val);\n' % (key)
        return ret
    elif isinstance(item, dict):                            # Dict (recursive)
        ret += 'var %s = new Object();\n' % key
        for k, v in item.iteritems():
            sub_key = key + "_tmp"
            ret += convertToJS(sub_key, v)
            ret += "%s['%s'] = %s;\n" % (key, k, sub_key)
        return ret

# Display Object ----------------------------------------------------------------------
img_dir = web_dir + "/%s/images"
class Display(object):
    '''
    This object will hold all the need info and some simple methods for generating the
    plots for different displays on the website.
    '''
    def __init__(self, name, region_name, views, view_fmt, step, scalars, vectors, 
                       gen_opts, cmap_options = None):
        if cmap_options != None:
            while len(cmap_options) < len(scalars):
                cmap_options += cmap_options
        
        self.step = step
        self.fields = {'scalars':dict.fromkeys(scalars, step), 
                       'vectors':dict.fromkeys(vectors, step)}
        self.region_name = region_name
        self.views = views
        self.view_fmt = view_fmt
        self.gen_opts = gen_opts
        self.name = name
          
        self.plot_opts = {}
        
        # Select the directory into which the images will be placed.
        this_dir = img_dir % name
        sub_dir = end_time.strftime("%Y_%m_%d")
        full_dir = this_dir + "/" + sub_dir
        if sub_dir not in os.listdir(this_dir):
            os.mkdir(full_dir)
        i = 0
        for s in self.fields['scalars'].iterkeys():
            self.plot_opts[s] = self.gen_opts.copy()
            if cmap_options != None:
                cmap = cm.get_cmap(cmap_options[i])
            else:
                cmap = None
                
            self.plot_opts[s].update(cmap = cmap, img_dir = full_dir)
            i += 1
        
        return
    
    def setRegion(self, region):
        '''
        Set the region object that will be used for ploting. This must be done before
        getPlots or canPlot can be called.
        '''
        self.region = region
    
    def getPlots(self, time):
        '''
        Loop through all the plots that need to be plotted and plot them.
        '''
        opts = self.plot_opts
        plot = self.region.plot
        for view in self.views:
            for s in self.fields['scalars'].iterkeys():
                opts[s]['view_opts'] = view
                for v in self.fields['vectors'].iterkeys():
                    if v == '~':
                        plot(time, s, **opts[s])
                    else:
                        plot(time, s, v, **opts[s])
        return
    
    def canPlot(self, time):
        '''
        Check to see if the datasets I want to plot are available from the outputs.
        '''
        ret = True
        for field in self.fields.itervalues():
            ds_list = field.keys()
            if '~' in ds_list:
                ds_list.remove('~')
            if not self.region.isTimeDsAvail(time, *ds_list):
                ret = False
        return ret 

# Initialize Datasets -----------------------------------------------------------------
disps = {}

# Set some general options.
t_fmt = "%H_%M"
gen_opts = {'wait':False, 'log':False, 'file_time_fmt':t_fmt}

# IONOSPHERE:
# Chose the scalar output values to be plotted.
iof_scalars = ['pot', 'fac_dyn', 'fac_tot']#, 'prec_e_fe_1', 'prec_e_fe_2', 'delphi']
disps[A] = Display(A,                                       # Name of this display
                   'earth',                                 # short key-name
                   [('sm', 90, 0), ('gse', 0, 180)],        # views
                   "centered at %s %f lat. by %f lon.",     # web-display label format
                   dt,                                      # time step
                   iof_scalars,                             # scalars
                   ['~'],                                   # vectors*
                   dJoin(gen_opts, {'alpha':0.5}),          # Other options
                   ['Greens', 'Blues', 'Purples', 'Reds']) 
# *Note: There are currently no vectors available for this region.

# MAGNETOSPHERE:
# Chose the scalar plots to be plotted.
m_scalars = ['btot', 'vtot', 'etot', 'temp', 'pp']#, 'rr', 'ent']
disps[M] = Display(M,                                       # The name of this display
                   'space',                                 # The short key-name
                   [('y', 0, -20, 100, -50, 50),            # Choose the views you want
                    ('y', 0, -20, 20,  -20, 20),            # to plot
                    ('x', 0, -20, 20,  -20, 20)],            
                   "%s = %d, %d-%d by %d-%d earth radii.",  # web-display label format
                   dt,                                      # timestep
                   m_scalars,                               # The scalars
                   ['b', 'v', 'e', '~'],                    # The vectors
                   dJoin(gen_opts, {'linecolor':'white'}))  # Other options
#                   ['Green_faid', 'Blue_faid', 
#                    'Red_faid', 'Purple_faid'])

# Put all the datasets into one dictionary
all_ds = {}
for name, disp in disps.iteritems():
    for field in disp.fields.itervalues():
        all_ds.update(field.copy())

# Remove the none (~) entries. These indicate to do a plot without vectors.
all_ds.pop('~', None) # Remove any null vector plots

# Initialize the run object
run = Run(np, sim_opts, **all_ds)

# Now that the regions have been initialized in the run object, give the Dispaly
# objects references to them.
for disp in disps.itervalues():
    disp.setRegion(run.regions[disp.region_name])

# JavaScript Specifications -----------------------------------------------------------
# Here I prepare the information that needs to go to the JavaScript website code.
file_format = "images/{yr}_{mo}_{da}/{F}_{V}_{hr}_{mi}.jpg"
specs = {}
for name, disp in disps.iteritems():
    spec = {}
    spec['fieldArray'] = {}
    for field, ds_dict in disp.fields.iteritems():
        ds_dict_cp = ds_dict.copy()
        has_empty = ds_dict_cp.pop('~', None)
        spec['fieldArray'][field] = {}
        spec['fieldArray'][field].update(run.getDatasetLabels(*ds_dict_cp.keys()))
        if has_empty != None:
            spec['fieldArray'][field]['~'] = "No %s shown" % field
    
    spec['time_step_min'] = disp.step/60.0
    
    view_opts = {}
    for view in disp.views:
        key = disp.region.viewOptsToString(view)
        view_opts[key] = disp.view_fmt % view
    spec['viewArray'] = view_opts
    
    spec['field_format'] = ''
    for field in spec['fieldArray'].iterkeys():
        spec['field_format'] += '{%s}' % field
    spec['file_format'] = file_format
    
    spec['EARLIEST_TIME'] = start_time + DT
    spec['LATEST_TIME'] = end_time
    specs[name] = spec

# Start the Simulation ----------------------------------------------------------------
run.start()
print 'starting plotting'

# Do the plotting ---------------------------------------------------------------------
T = start_time - timedelta(hours = pr) + DT
while T <= end_time:
    #can_plot = True
    # Check to see if all the displays are able to plot
    #for disp in disps.itervalues():
    #    if not disp.canPlot(T):
    #        can_plot = False
    
    # If they are, then plot.
    #if can_plot:
    if run.isTimeAvailable(T):
        print str(T), "Available"
        if T > start_time: # Only do stuff AFTER the prerun
            for name, disp in disps.iteritems():
                disp.getPlots(T)
                          
        # Increment the time
        T = T + DT
    
    # Aaaaand SLEEP...
    sleep(5)

print 'ending plotting'

n_plots_left = run.finish()

print "Writing results to site"
for name, spec in specs.iteritems():
    spec_str = ''
    for key, value in spec.iteritems():
        spec_str += convertToJS(key, value)
    
    page_dir = web_dir + "/" + name
    f = open(page_dir + '/scripts/spec.js', 'w')
    f.write(spec_str)
    f.close()

print n_plots_left
