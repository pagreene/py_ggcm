from numpy import where, sqrt, array, sin, cos, arccos, pi, zeros
from numpy.linalg import solve
from time import sleep
from ggcm_logger import Logged_Object
from spacepy.coordinates import Coords
from spacepy.time import Ticktock

#======================================================================================
# Base Dataset objects
#======================================================================================

class Data(object):
    '''
    This is really just a struct to hold the information from a dataset
    '''
    def __init__(self, data, crds, time):
        self.data = data # array or dict of arrays
        self.crds = crds # dict of arrays
        self.time = time # datetime object
        return

def keyDict(d):
    '''
    Recursive function to give the types of elements in a dictionary indexed by key.
    '''
    if isinstance(d, dict):
        ret = {}
        for k, v in d.iteritems():
            ret[k] = keyDict(v)
    elif isinstance(d, list) or isinstance(d, tuple):
        ret = []
        for v in d:
            ret.append(keyDict(v))
    else:
        ret = type(d)
    return ret

_ds_dict = {}
class _Base_Dataset(Logged_Object):
    '''
    This is the base dataset object
    '''
    # Basic info defined by subclasses. This and any undefined functions below are all
    # that should be changed by children classes. The purpose of this object is to make
    # the behaviour of different datasets identical to outside interfaces, even though
    # some of the internals may differ in form. Those interfaces should ideally be 
    # defined in this class.
    file_type = None
    file_key = None
    data_type = None
    datasets = []
    
    # Format for "NotImplementedError messages.
    nie_fmt = '%s must be defined in any initialized child class.'
    
    def __init__(self, name):
        Logged_Object.__init__(self, 'ds.%s' % name)
        
        if self.data_type == 'vector':
            self.starts = None
        
        if self.file_type == None:
            msg = 'file_type must be defined in any initialized child class.'
            raise NotImplementedError(msg)
        
        # In general the file key and file type will be the same, but not always.
        if self.file_key == None:
            self.file_key = self.file_type
        
        self.name = name
        self._out_obj = None
        
        self._last_data = (None, None)
        _ds_dict[name] = self
        return
    
    def setOutObj(self, out_obj):
        '''
        Method to set the output object after initialization. No data retrieval can be
        done until the object is set.
        '''
        # Note: The output object is only needed by dataset objects that retrieve the
        # data directly from the output files. Those that get their data from other
        # objects probably don't need this, but I'm not sure. Something worth looking
        # into.
        self._out_obj = out_obj
    
    class StepError(Exception):
        pass
    
    def _getSimData(self, num):
        '''
        API that must be overwritten in sub-classes
        '''
        msg = self.nie_fmt % '_getSimData'
        raise NotImplementedError(msg)
    
    def isTimeAvailable(self, time):
        '''
        API to determine if time available
        '''
        out_obj = self._out_obj
        return out_obj.isTimeAvailable(time)
    
    def getData(self, time, wait = False):
        '''
        Method to retrieve data from the data files.
        
        Input:
        time -- the time (UTC) for the output desired.
        wait -- boolean, whether to wait if the output is not available or except.
        '''
        out_obj = self._out_obj
        f_key = self.file_key
        
        # The internal specifics of how the data is retrieved varies somewhat,and is
        # thus defined in child classes.
        return self._getData(time, wait, out_obj, f_key)
    
    def _acquireData(self, time, wait, out_obj, f_key):
        '''
        Yet another sub function in the process of getting the data for the dataset.
        This should be defined by the child class.
        '''
        raise NotImplementedError("This must be defined in a child class")
    
    def _getData(self, time, wait, out_obj, f_key):
        '''
        API to get the data for output 'num'
        '''
        self.debug("Getting data requested for %s" % str(time))
        
        # If this is the same data we got last time, just get the same data. Note: This
        # only works when this function is called IN THE SAME PROCESS twice. It may not
        # be a bad idea to save this to a file and read it from that so that it is
        # available to multiple processes, although that may add more complications and
        # in fact be slower.
        if self._last_data[0] == time:
            return self._last_data[1]
        
        # All of the arguments passed to getData should be passed to this function,
        # which is definedin the child classes. The child class should be able to draw
        # from anything passed to this, and moreover it must be able to call this 
        # fucntion from other dataset objects.
        outs, crds, ret_time = self._acquireData(time, wait, out_obj, f_key)
        
        # Check to make sure we actually got something.
        if not len(outs):
            raise self.StepError("No output captured")
        
        # Process the outputs. This is where the multiple datasets should be combined
        # into either one scalar or one vector quantity.
        data = self._processOuts(outs, time)
        
        # Package the results in a nice convenient form
        ret_data = Data(data, crds, ret_time)
        
        # Store the results so I can grab them later if I need this again for the same
        # time. This WILL happen if I call for, for example, plasma pressure and 
        # temperature, because the temperature calls for the pressure. As it stands,
        # this only works if the call is made from the SAME PROCESS. See above.
        self._last_data = (time, ret_data)
        
        # THIS IS TEMPORARY CODE TO DETERMINE REASONABLE BOUNDS ON DATA
        if not isinstance(data, dict):
            bound = "%s %g %g %g\n" % (self.name, data.min(), data.mean(), data.max())
            f = open('/home/pan56/py_ggcm/ref/bounds.log', 'a')
            f.write(bound)
            f.close()
        # -------------------------------------------------------------
        
        self.debug("Returning dataset for %s" % str(ret_time))
        return ret_data 
    
class Average_Dataset(_Base_Dataset):
    '''
    This is for datasets that are come strait from the output file
    '''
    file_type = None
    datasets = []
    def __init__(self, *args, **kwargs):
        _Base_Dataset.__init__(self, *args, **kwargs)
        
        # Choose the name of the components if there are multiple subdatasets. This 
        # assumes that multiple subdatasets implies a vector for Average_Dataset.
        if len(self.datasets) > 1:
            # I will look for bits of the dataset names that are different...
            diffs = []
            for j in range(1,len(self.datasets)):
                for i, l in enumerate(self.datasets[0]):
                    if l != self.datasets[j][i]:
                        if i not in diffs:
                            diffs.append(i)
                            break
            
            # ... then match each dataset with the differing part of its name.
            self.comps = {}
            for ds in self.datasets:
                comp = ''
                for diff in diffs:
                    comp += ds[diff]
                self.comps[ds] = comp
        else:
            self.comps = {self.datasets[0]:self.datasets[0]}
            
        return
    
    def _acquireData(self, time, wait, out_obj, f_key):
        '''
        Internal recursive function that is implemented within _getData to retrieve the
        data (as opposed to the coordinates or the time, which are also gotten in
        _getData)
        '''
        # Define a dict to hold the ouptuts.
        outs = {}
        
        # Get data
        for ds in self.datasets:
            # Get the data from the output object.
            outs[ds] = out_obj.loadData(ds, time, fail_ok = wait)
            t = 0
            dt = 10
            # Wait if that is what was asked for.
            while outs[ds] == None:
                print 'Retrying to load data for %s %d' % (ds, t) 
                sleep(dt)
                t += dt
                outs[ds] = out_obj.loadData(ds, time, fail_ok = wait)
            
            # Get the coordinates and the actual time used. (Remember that there are
            # only discrete times available, but any time can be requested, so the time
            # nearest to the requested time is retrieved.)
            crds = out_obj.getCrds()
            ret_time = out_obj.getNearestTime(time)
         
        # If multiple simulations are used (not yet globally supported) then average
        # the outputs. Another object could use _processSims to find the difference.
        outs = self._processSims(outs)
        
        return (outs, crds, ret_time)
        
    def _processSims(self, outs):
        '''
        Internal API to process the outputs from the simulation.
        '''
        ret = {}
        for key, ds in outs.iteritems():
            k = self.comps[key]
            if len(ds) == 1:
                # If there is output from only one simulation, just return that one output
                ret[k] = ds.values()[0]
            else:
                # Average if there are multiple simulations
                raise NotImplementedError("This feature has not yet been implemented.")
        
        # If there was just one dataset just return the info for that one dataset.
        if len(ret) == 1:
            return ret.values()[0]
        else:
            return ret
    
    def _processOuts(self, outs, time):
        '''
        The outputs are not processed further by this class.
        '''
        return outs

class Ave_Calc_Ds(Average_Dataset):
    def __init__(self, name):
        Average_Dataset.__init__(self, name)
        for ds in self.datasets:
            if not _ds_dict.has_key(ds):
                _ds_dict[ds] = DS_INIT_DICT[ds](ds)
        return
        
    def _calc(self, ins, time):
        raise NotImplementedError
    
    def _acquireData(self, *args, **kwargs):
        '''
        Internal recursive function that is implemented within _getData to retrieve the
        data (as opposed to the coordinates or the time, which are also gotten in
        _getData)
        '''
        # Define a dict to hold the ouptuts.
        outs = {}
        
        # Get data
        for ds in self.datasets:
            # Retrieve the data from the other dataset object. All dataset objects
            # are registered in the global _ds_dict, which should only be used in
            # this file.
            try:
                data_obj = _ds_dict[ds]._getData(*args, **kwargs)
                outs[ds] = data_obj.data
                crds = data_obj.crds
                ret_time = data_obj.time
            except RuntimeError:
                # If there is a recursion error, I want to know where it happened.
                msg = "Error occured for " + ds
                class RecursionError(Exception):
                    pass
                
                raise RecursionError(msg)
        
        return (outs, crds, ret_time) 
    
    def _processOuts(self, outs, time):
        '''
        API to process the outputs from the subdatasets. This passes on to _calc, which
        must be defined for each child class.
        '''
        keys = outs.keys()
        keys.sort()
        ins = []
        for k in keys:
            ins.append(outs[k])
        try:
            return self._calc(ins, time)
        except:
            print "\nIN PROCESS_OUTS ====================="
            print keyDict(outs)
            raise

#======================================================================================
# Helpful Functions for performing the calculation for the dataset children
#======================================================================================
def toPoints(where_out):
    '''
    Function to convert the output of numpy.where into the index points.
    '''
    n = len(where_out)
    N = len(where_out[0])
    ret = []
    for i in range(N):
        item = []
        for j in range(n):
            item.append(where_out[j][i])
        ret.append(tuple(item))
    return ret

def div(a, b):
    '''
    Convenience function to divide two arrays and avoid division by zero and negative 
    results. THIS IS NOT DIVERGENCE!!!
    '''
    a_pts = toPoints(where(a < 0))
    n = len(a_pts)
    for k in range(n):
        a[a_pts[k]] = 0
    
    b_pts = toPoints(where(b <= 0))
    n = len(b_pts)
    for k in range(n):
        b[b_pts[k]] = 1.0e-10
    
    return a/b

def mag(*v):
    '''
    Function to find the magnitude of a vector
    '''
    try:
        s = v[0]**2
    except:
        print "\nIN MAG ============================="
        print keyDict(v)
        raise
    for x in v[1:]:
        s += x**2
    return sqrt(s)

#======================================================================================
# Dataset Definitions (These should eventually be automatically generated.)
#======================================================================================
class Btot(Ave_Calc_Ds):
    data_type = 'scalar'
    file_type = '3d'
    label = "Magnetic field magnitude"
    units = "nT"
    min_val = 0.0
    max_val = 2.0e4
    datasets = ['b']
    def _calc(self, ins, time):
        return mag(*ins[0].values())

class Vtot(Ave_Calc_Ds):
    data_type = 'scalar'
    file_type = '3d'
    label = 'Velocity Magnitude'
    units = 'm/s'
    min_val = 0.0
    max_val = 1.0e3
    datasets = ['v']
    def _calc(self, ins, time):
        return mag(*ins[0].values())

class XJtot(Ave_Calc_Ds):
    data_type = 'scalar'
    file_type = '3d'
    label = "Current magnitude"
    units = '$\mu A/m^2$'
    min_val = 0.0
    max_val = 0.05
    datasets = ['xj']
    def _calc(self, ins, time):
        return mag(*ins[0].values())

class Temp(Ave_Calc_Ds):
    data_type = 'scalar'
    file_type = '3d'
    label = 'Plasma temperature'
    units = '$K$'
    min_val = 1.0e4
    max_val = 1.0e8
    datasets = ['pp', 'rr']
    def _calc(self, ins, time):
        pp, rr = ins
        return 72429.0*div(pp, rr)

class CS(Ave_Calc_Ds):
    data_type = 'scalar'
    file_type = '3d'
    label = 'Sond speed'
    units = '$m/s$'
    min_val = 0.0
    max_val = 1.5e3
    datasets = ['temp']
    def _calc(self, ins, time):
        return 0.11771422*sqrt(ins[0])

class VA(Ave_Calc_Ds):
    data_type = 'scalar'
    file_type = '3d'
    label = "Alfven speed"
    units = ""
    min_val = 0.0
    max_val = 4.0e5
    datasets = ['btot', 'rr']
    def _calc(self, ins, time):
        btot, rr = ins
        return 21.89*div(btot, sqrt(rr))

class MVA(Ave_Calc_Ds):
    data_type = 'scalar'
    file_type = '3d'
    label = 'Alfven mach number'
    units = ''
    min_val = 0.0
    max_val = 15.0
    datasets = ['va', 'vtot']
    def _calc(self, ins, time):
        va, vtot = ins
        return div(vtot, va)
    
class VMS(Ave_Calc_Ds):
    data_type = 'scalar'
    file_type = '3d'
    label = 'magnetosonic speed'
    units = '$m/s$'
    min_val = 0.0
    max_val = 1.0e6
    datasets = ['cs', 'va']
    def _calc(self, ins, time):
        cs, va = ins
        return sqrt(cs**2 + va**2)

class MVMS(Ave_Calc_Ds):
    data_type = 'scalar'
    file_type = '3d'
    label = 'magnetosonic mach mumber'
    units = '$unitless$'
    min_val = 0.0
    max_val = 10.0
    datasets = ['vms', 'vtot']
    def _calc(self, ins, time):
        vms, vtot = ins
        return div(vtot, vms)

class Beta(Ave_Calc_Ds):
    data_type = 'scalar'
    file_type = '3d'
    label = 'Plasma beta'
    units = ''
    min_val = 0.0
    max_val = 20.0
    datasets = ['btot','pp']
    def _calc(self, ins, time):
        btot, pp = ins
        return 2.5133*div(pp, btot**2)

class MCS(Ave_Calc_Ds):
    data_type = 'scalar'
    file_type = '3d'
    label = 'Sound mach number'
    units = ''
    min_val = 0.0
    max_val = 28.0
    datasets = ['cs', 'vtot']
    def _calc(self, ins, time):
        cs, vtot = ins
        return div(vtot, cs)

class VD(Ave_Calc_Ds):
    data_type = 'scalar'
    file_type = '3d'
    label = 'Ion-electron drift speed'
    units = ''
    min_val = 0.0
    max_val = 1.5e2
    datasets = ['rr', 'xjtot']
    def _calc(self, ins, time):
        rr, xjtot = ins
        return 6241.0*div(xjtot, rr)

class VDVA(Ave_Calc_Ds):
    data_type = 'scalar'
    file_type = '3d'
    label = 'Ion-electron drift speed normalized Alfven speed'
    units = ''
    min_val = 0.0
    max_val = 0.5
    datasets = ['va', 'vd']
    def _calc(self, ins, time):
        va, vd = ins
        return div(vd, va)

class ENT(Ave_Calc_Ds):
    data_type = 'scalar'
    file_type = '3d'
    label = 'Specific Entropy'
    units = ''
    min_val = 1.0e4
    max_val = 2.0e8
    datasets = ['pp', 'rr']
    def _calc(self, ins, time):
        pp, rr = ins
        return 72429.0*div(pp, rr**(5.0/3.0))

class E(Ave_Calc_Ds):
    data_type = 'vector'
    file_type = '3d'
    label = 'Electric field'
    datasets = ['b','v']
    def _calc(self, ins, time):
        b, v = ins
        E = {}
        E['x'] = 0.001*(b['y']*v['z'] - b['z']*v['y'])
        E['y'] = 0.001*(b['z']*v['x'] - b['x']*v['z'])
        E['z'] = 0.001*(b['x']*v['y'] - b['y']*v['x'])
        return E

class Etot(Ave_Calc_Ds):
    data_type = 'scalar'
    file_type = '3d'
    label = 'Electric field magnitude'
    units = ''
    min_val = 0.0
    max_val = 15.0
    datasets = ['e']
    def _calc(self, ins, time):
        try:
            return mag(*ins[0].values())
        except:
            print "\nIN ETOT ======================="
            print keyDict(ins)
            raise
    
class PP(Average_Dataset):
    data_type = 'scalar'
    file_type = '3d'
    label = 'Plasma Pressure'
    units = '$nPa$'
    min_val = 0.0
    max_val = 8000.0
    datasets = ['pp']

class RR(Average_Dataset):
    data_type = 'scalar'
    file_type = '3d'
    label = 'Plasma Density'
    units = '$cm^{-3}$'
    min_val = 0.0
    max_val = 80.0
    datasets = ['rr']

class B(Average_Dataset):
    data_type = 'vector'
    file_type = '3d'
    label = "Magnetic Field"
    units = "$nT$"
    datasets = ['by', 'bx', 'bz']

class V(Average_Dataset):
    data_type = 'vector'
    file_type = '3d'
    label = "Plasma Velocity Field"
    units = "$m/s$"
    datasets = ['vx', 'vy', 'vz']

class XJ(Average_Dataset):
    data_type = 'vector'
    file_type = '3d'
    label = "Current density"
    units = '$\mu A/m^2$'
    datasets = ['xjx', 'xjy', 'xjz']

class POT(Average_Dataset):
    data_type = 'scalar'
    file_type = 'iof'
    label = 'Potential'
    units = '$V$'
    min_val = -2.0e5
    max_val = 2.0e5
    datasets = ['pot']

class PACURR(Average_Dataset):
    data_type = 'scalar'
    file_type = 'iof'
    label = "Field Aligned Current (FAC) density"
    units = "$\mu A/m^2$"
    min_val = -5e-6
    max_val = 5e-6
    datasets = ['pacurr']

class SIGH(Average_Dataset):
    data_type = 'scalar'
    file_type = 'iof'
    label = "Hall conductance"
    units = "$S$"
    min_val = 0.0
    max_val = 30.0
    datasets = ['sigh']
    
class SIGP(Average_Dataset):
    data_type = 'scalar'
    file_type = 'iof'
    label = "Pedersen conductance"
    units = "$S$"
    min_val = 0.0
    max_val = 1.5e3
    datasets = ['sigp']

class Prec_E_Fe_1(Average_Dataset):
    data_type = 'scalar'
    file_type = 'iof'
    label = "Diffuse Auroral e- Precipitation Energy Flux"
    units = "$W/m^2$"
    min_val = 0.0
    max_val = 0.05
    datasets = ['prec_e_fe_1']

class Prec_E_Fe_2(Average_Dataset):
    data_type = 'scalar'
    file_type = 'iof'
    label = "Discrete Auroral e- Precipication Energy Flux"
    units = "$W/m^2$"
    min_val = 0.0
    max_val = 0.01
    datasets = ['prec_e_fe_2']

class Prec_E_E0_1(Average_Dataset):
    data_type = 'scalar'
    file_type = 'iof'
    label = "Diffuse auroral e- precipitation mean energy"
    units = "$eV$"
    min_val = 0.0
    max_val = 5.0e3
    datasets = ['prec_e_e0_1']
    
class Prec_E_E0_2(Average_Dataset):
    data_type = 'scalar'
    file_type = 'iof'
    label = 'Discrete auroral e- precipitation mean energy'
    units = '$eV$'
    min_val = 0.0
    max_val = 10.0e3
    datasets = ['prec_e_e0_2']

class DELPHI(Average_Dataset):
    data_type = 'scalar'
    file_type = 'iof'
    label = "Knight potential"
    units = "$V$"
    min_val = 0.0
    max_val = 6.0e3
    datasets = ['delphi']
    
class RRIO(Average_Dataset):
    data_type = 'scalar'
    file_type = 'iof'
    label = 'mapped density' 
    units = '$cm^{-3}$'
    min_val = 0.0
    max_val = 4.0e6
    datasets = ['rrio']

class PPIO(Average_Dataset):
    data_type = 'scalar'
    file_type = 'iof'
    label = 'mapped pressure'
    units = '$pPa$'
    min_val = 0.0
    max_val = 1.0e-5
    datasets = ['ppio']

class TTIO(Average_Dataset):
    data_type = 'scalar'
    file_type = 'iof'
    label = 'mapped temperature'
    units = '$K$'
    min_val = 0.0
    max_val = 5.0e7
    datasets = ['ttio']

class Fac_Dyn(Average_Dataset):
    data_type = 'scalar'
    file_type = 'iof'
    label = 'dynamo FAC from CTIM'
    units = '$\mu A/m^2$'
    min_val = 0.0
    max_val = 1.0e-5
    datasets = ['fac_dyn']
    
class Fac_Tot(Average_Dataset):
    data_type = 'scalar'
    file_type = 'iof'
    label = 'total FAC'
    units = '$\mu A/m^2$'
    min_val = -5.0e-6
    max_val = 5.0e-6
    datasets = ['fac_tot']
    
class XJH(Average_Dataset):
    data_type = 'scalar'
    file_type = 'iof'
    label = 'Joule heating rate'
    units = '$W/m^2$'
    min_val = 0.0
    max_val = 0.2
    datasets = ['xjh']
    
class Delbt(Average_Dataset):
    data_type = 'scalar'
    file_type = 'iof'
    label = 'Ground magnetic H perturbation'
    units = '$nT$'
    min_val = -1.5e-6
    max_val = 1.5e-6
    datasets = ['delbt']

class EIO(Average_Dataset):
    data_type = 'scalar'
    file_type = 'iof'
    label = 'azimuthal electric field'
    units = '$mV/m$'
    min_val = 0.0
    max_val = 1.0
    datasets = ['epio', 'etio']

# Define the dictionary with all the name keys for the datasets tied to the objects.
# Currently, this must be updated manually. That should be changed.
DS_INIT_DICT = {'pp':PP, 
                'rr':RR, 
                'vms':VMS,
                'btot':Btot, 'b':B, 
                'vtot':Vtot, 'v':V, 
                'etot':Etot, 'e':E,
                'xjtot':XJtot, 'xj':XJ,
                'temp':Temp,
                'cs':CS,
                'va':VA,
                'mva':MVA,
                'vms':VMS,
                'mvms':MVMS,
                'beta':Beta,
                'mcs':MCS,
                'vd':VD,
                'vdva':VDVA,
                'ent':ENT,
                'pot':POT,
                'pacurr':PACURR,
                'sigh':SIGH,
                'sigp':SIGP,
                'prec_e_fe_1':Prec_E_Fe_1, 'prec_e_fe_2':Prec_E_Fe_2, 
                'prec_e_e0_1':Prec_E_E0_1, 'prec_e_e0_2':Prec_E_E0_2,
                'delphi':DELPHI,
                'ppio':PPIO,
                'rrio':RRIO,
                'ttio':TTIO,
                'fac_dyn':Fac_Dyn,
                'fac_tot':Fac_Tot,
                'xjh':XJH,
                'delbt':Delbt,
                'eio':EIO}
