import os
import stat
from numpy import min, max, empty, mgrid, transpose, ndarray, diff, where
from numpy import arange, array, nditer, float32, zeros, sin, cos, log, pi
from h5py import File
from time import sleep
from datetime import datetime, timedelta, time
from subprocess import Popen
from copy import copy
from threading import Thread, RLock, Lock
from urllib2 import urlopen, URLError
from multiprocessing import Process
from warnings import warn

from os import environ
if 'DISPLAY' not in environ:
    # This is necessary if there is no access to a screen (X-windows)
    from matplotlib import use
    use("agg", warn = False)

# Any and all matplotlib imports must be made below this point.
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.interpolate import griddata, interp1d
from my_basemap import My_Basemap
from PIL import Image
from spacepy.coordinates import Coords
from spacepy.time import Ticktock
from my_basemap import Img_Array

try:
    from matplotlib.pyplot import streamplot
    STREAMPLOT = True
except:
    print "streamplot could not be imported."
    STREAMPLOT = False
    def streamplot(*args, **kwargs):
        raise ggcm_exception("streamplot was not imported, but was used.", "Import Error")


# Setting some default formatting for plots in matplotlib
rc('text', color = 'white')
rc('font', size = 7.0, weight = '450')
rc('mathtext', tt = 'monospace:bold', default = 'tt')
rc('axes', labelcolor = 'white', labelweight = '650', edgecolor = 'w')
rc('xtick', color = 'w')
rc('ytick', color = 'w')

import My_Re

from ggcm_logger import Logged_Object, LOG_DIR
from ggcm_dir import getDirectory, LOCAL, PWD
from sim_defaults import *
from ggcm_exceptions import ggcm_exception
from ggcm_datasets import DS_INIT_DICT

TO_RAD = pi/180

class Usage_Error(Exception):
    '''
    A simple exception for instances were methods are called at inappropriate times.
    '''
    pass

#================================================================================
# Plotting Utilities
#================================================================================

class MismatchedFiletypeError(Exception):
    def __init__(self, got, expected):
        Exception.__init__(self, 'Got filetypes %s! Expected %s' % (got, expected))

class Region(object):
    '''
    This is the base class for all region objects. Region objects represent a 
    plot region, such as the magnetosphere or an earth map. Each region will be
    associated with a group of datasets. More specifically this object should be
    associated with a particular output file type.
    
    INIT:
    name    - (string) the name of the region (e.g. Earth)
    ds_dict - (dict) containing dataset (ds) objects indexed by their key
    queue   - (int) the plot queuing object
    '''
    class_default_view = None
    file_type = None
    def __init__(self, name, ds_dict, queue, default_view = None):
        self.name = name
        self.ds_dict = ds_dict
        self.queue = queue
        
        # Set the default view
        if default_view != None:
            self.default_view = default_view
        elif self.class_default_view != None:
            self.default_view = self.class_default_view
        else:
            raise Exception('There must be a default view available!')
        
        # Each region has its own figure on which to plot. When there are multiple plot
        # requests, the same object is used, which is only possible because each
        # goes into a separate process, and the object is COPIED into that process.
        # Thus changes made to the object while in that process DO NOT carry over.
        self.fig = plt.figure(name + '_Figure')
        return
    
    def addDsObj(self, ds_obj):
        '''
        A method to add on another dataset object to the ds_dict.
        '''
        self.ds_dict[ds_obj.name] = ds_obj
    
    def isTimeDsAvail(self, time, *datasets):
        '''
        API to determine if a given time is within available range for all datasets required for a plot.
        
        Args:
        time -- (datetime objects) the time to check if dataset(s) available
        
        At least one dataset name must be given. For instance, region.isTimeDsAvail(datetime, pp, rr),
        will check if plasma pressure (pp) and plasma density (rr) are available in the outputs from
        the simulation. Only if ALL the datasets are available will this return true.
        '''
        ret = True
        for ds in datasets:
            if not self.ds_dict[ds].isTimeAvailable(time):
                ret = False
        
        return ret
    
    def isTimeAvailable(self, time):
        '''
        API to determine if the given time is available for all the datasets for this region.
        
        Args:
        time -- (datetime object) the time to check for dataset availability.
        '''
        ret = True
        for dataset in self.ds_dict.itervalues():
            if not dataset.isTimeAvailable(time):
                ret = False
                break
        return ret
    
    def plot(self, time, *datasets, **plot_options):
        '''
        API to plot a dataset for a particular time.
        
        Arguments:
        time = datetime: the time for the dataset to be plotted
        The names of datasets to be plotted.
        
        Keyword Arguments:
        wait = True|False: If true, wait for dataset to become available, if false, 
                           except if not available.
        facecolor -- the color of the axes
        edgecolor -- the color of the edges
        view      -- a tuple or list with specifications for the view point of the plot
        '''
        # Get the dataset object from the dataset names
        data_dict = {}
        for ds in datasets:
            data_dict[ds] = self._getData(self.ds_dict[ds], time, plot_options)
        
        # Add the request to the multiprocessing queue.
        self.queue.addToQueue(self._plot, time, data_dict, plot_options)
        return
    
    def _getData(self, ds, time, opts):
        '''
        This is a sub-function that may be redefined in a child class that retrieves 
        the data from the dataset object.
        '''
        wait = opts.pop('wait', False)
        if ds.file_type != self.file_type:
            raise MismatchedFiletypeError(ds.file_type, self.file_type)
        
        return ds.getData(time, wait)
    
    def _plot(self, time, data_dict, plot_options):
        '''
        Internal function to execute the plots. EVERYTHING IN THIS METHOD OCCURS IN A 
        DIFFERENT PROCESS!!! That means that nothing that modifies the Region object
        will carry through. Proceed with caution.
        '''
        names = []
        for ds in data_dict.iterkeys():
            names.append(self.ds_dict[ds].name)
        
        print "Plotting", names
        #print 'Plot Options for %s:' % str(datasets), plot_options
        
        # Make sure we're working with the correct plot.
        plt.figure(self.fig.number)
        
        # Remove the plot options that don't go to the axes.
        facecolor = plot_options.pop('facecolor', 'k')
        edgecolor = plot_options.pop('edgecolor', 'k')
        
        # Plot the datasets to the axis
        ax = None
        for ds, data in data_dict.iteritems():
            ds_obj = self.ds_dict[ds]
            
            # If we don't have the axes already, make them. I don't make them 
            # ahead of time because I need the real time from the data.
            if ax == None:
                if plot_options.has_key('ax'):
                    ax = plot_options.pop('ax')
                else:
                    view_opts = plot_options.pop('view_opts', self.default_view)
                    ax = self.getAxes(data.time, view_opts)
            
            # Make the plot
            if ds_obj.data_type == 'scalar':
                names[0] = ds_obj.name # For creating the image file name
                m = ds_obj.min_val
                M = ds_obj.max_val
                plot_options['cb_scale'] = arange(m, M, (M - m)/100.0)
                self._contour(ax, data, **plot_options)
            elif ds_obj.data_type == 'vector':
                names[1] = ds_obj.name # For creating the image file name
                self._streamline(ax, data, **plot_options)
        
        # Name the image
        if plot_options.has_key('plot_name'):
            image_name = plot_options['plot_name']
        else:
            img_dir = plot_options.pop('img_dir', '.')
            image_name = img_dir + '/'
            for name in names:
                image_name += name
            image_name += '_' + self.viewOptsToString(ax.view_opts)
            t_fmt = plot_options.pop("file_time_fmt", "%Y%m%d%H%M")
            image_name += '_' + time.strftime(t_fmt)
            
        # Save the image
        print "Saving", image_name
        self.fig.savefig(image_name + '.png', facecolor = facecolor, 
                         edgecolor = edgecolor)
        
        # Convert the image
        self.convertImage(image_name)
        
        return
    
    def convertImage(self, image_name):
        '''
        A function to convert the output image file from png to jpg.
        '''
        im = Image.open(image_name + '.png')
        bg = Image.new("RGB", im.size, (255,255,255))
        bg.paste(im,im)
        bg.save(image_name + '.jpg')
        os.remove(image_name + '.png')
        return
    
    def _choosePlot(self, how, ax, data, **plot_options):
        '''
        Specific plots for different types of data
        
        This must be implemented by the child class
        '''
        raise  NotImplementedError('This function must be defined in sub-class')
        
    def getAxes(self, time, view_opts):
        '''
        API to get the current axes
        
        This must be implemented by the child class
        '''
        raise NotImplementedError('This function must be defined in sub-class')
    
    def viewOptsToString(self, view_opts):
        '''
        API to convert a Regions view options into a string for the image name.
        
        This must be implemented by the child class
        '''
        raise NotImplementedError('This function must be defined in sub-class')
    
    def logShift(self, value):
        '''
        API to make whatever transformation I want to make consistent
        '''
        # Check to make sure we have no negative values.
        if float(value) < 0.0:
            if abs(float(value)) > 0.001:
                raise ValueError("Bad value: %s" % str(value))
            value = 0.0
        
        # My_Returning log base ten shift of the value + 1, so that we don't 
        # get the asymptotic drop to -infinity for decimals near zero.
        return log(float(value) + 1.0)/log(10)
    
    def logShiftArray(self, a):
        '''
        API to take the natural log of a number of list of numbers or array of numbers
        '''
        if isinstance(a, ndarray):
            # Apply shift to all elements of an array.
            for x in nditer(a, op_flags = ['readwrite']):
                x[...] = self.logShift(x)
            return a
        else:
            # If 'a' is none of the above, something is wrong
            print 'incorrect item:', a
            raise ValueError("Got %s, not `ndarray'" % type(a))
    
class Earth(Region):
    '''
    This is the object for plotting on basemaps
    '''
    file_type = 'iof'
    class_default_view = ('sm', 0, -180)
    
    def makeNightDay(self, time, scale = 1.0):
        '''
        Method to piece together a map of the earth with night side and day side
        using NASA's bluemarble and nightlights images.
        '''
        nit_tmp = Img_Array(LOCAL.full + '/ref/earth_night.jpg')
        day_tmp = Img_Array(LOCAL.full + '/ref/earth_day.jpg')
        nit = nit_tmp.resize(scale)
        day = day_tmp.resize(scale)
        
        del nit_tmp
        del day_tmp
        
        W = nit.shape[1] # width
        H = nit.shape[0] # height
        
        # In GSE coordinates, define a circle as the night-day border.
        phs = arange(0, 2*pi*(1 + 1.0/W), 2*pi/W)
        edge = empty([W+2,3])
        edge[1:,:] = array([zeros(phs.size), sin(phs), cos(phs)]).transpose()
        
        # Add a tag-along that will allow me to determine which side is day.
        edge[0,:] = array([1,0,0])
        
        # Create an array with the same number of points as our night-day border to
        # hold the relevant time (in UTC). All elements will be the same, because the
        # package I use for the coordinate transformation requires that each input
        # have an associated time.
        nows = empty(len(edge), dtype = "object")
        nows[:] = time
        ticks = Ticktock(nows, 'UTC')
        
        # Now convert the coordinates. First, I define a Coords object (from imported 
        # package) for the line in GSE, and then I use the object methods to transform
        # it into geographic (GEO) coordinates, and save the data.
        edge_gse = Coords(edge, 'GSE', 'car', ticks = ticks)
        edge_geo = edge_gse.convert('GEO', 'car').convert('GEO', 'sph').data
        
        # I now need to sort the points, ordering by the W components. GEO coordinates
        # have a direct correspondance to pixels in the image (because of the type of
        # projection).
        edge_pix_unsort = empty([W+2,2], dtype = 'int')
        edge_pix_unsort[:,0] = edge_geo[:,2]*W/360.0 # Convert from degrees to pixels
        edge_pix_unsort[:,1] = edge_geo[:,1]*H/180.0 # Convert from degrees to pixels
        
        # Use the tag-along I placed earlier to get the point that is on the day side.
        x_pix = array([W/2, H/2]) + edge_pix_unsort[0,:]
        
        # I ditch the tag-along, and I make the rest of the array into a list, then I
        # sort by the first item in each of the sub-lists, which is the horizontal 
        # (longitudinal) coordinate of our line.
        edge_pix_list = edge_pix_unsort[1:,:].tolist()
        edge_pix_list.sort()
        
        # Convert back into an array.
        edge_pix = array(edge_pix_list)
        
        # If the first longitude is repeated twice, shift the lower one back a pixel.
        # The lower one will be earlier because the sort applied above sorts by the
        # second item if the first items are identical.
        diffs = diff(edge_pix[:,0])
        if diffs[0] == 0:
            edge_pix[0,0] = edge_pix[0,0] - 1
        
        # Interpolate to get a latitude value for every longitudinal pixel.
        interp = interp1d(edge_pix[:,0], edge_pix[:,1], 
                          bounds_error = False, fill_value = edge_pix[0,1])
        
        # Shift the pixels, because we measure height from the bottom, but so far 
        # we have be measuring from the middle.
        heights = H/2 - interp(range(-W/2, W/2))
        
        # Convert the colors in the image arrays into the correct format
        norm_day = day.astype(float32)/255.0
        norm_nit = nit.astype(float32)/255.0
        
        # Determine which side of the border is day using that nifty tag-along.
        top_day = True
        if heights[x_pix[0]] < x_pix[1]:
            top_day = False
        
        # Make the image array by appending the night images to the day images, with
        # the border defined by the line we went to so much trouble to determine.
        img = empty([H,W,3], dtype = float32)
        for w, h in enumerate(heights): # Going column by column.
            try:
                if top_day:
                    img[h:,w,:] = norm_nit[h:,w,:]
                    img[:h,w,:] = norm_day[:h,w,:]
                else:
                    img[:h,w,:] = norm_nit[:h,w,:]
                    img[h:,w,:] = norm_day[h:,w,:]
            except:
                print "h: %s/%s, w: %s/%s" % (str(h), str(H), str(w), str(W))
                raise
        
        # Make an `Image' object (imported from Image) from the `img' array.
        Img = Img_Array(img)
        Img.fname = 'nightAndDay_%s.jpg' % time.strftime("%Y%m%d%H%M")
        
        return Img
    
    def transfromCrds(self, time, crds):
        '''
        API to transform the datafile Solar-Magetic coords to Geographic coordinates.
        '''
        # For ease of use, put the phi and theta coords in phs and ths
        phs = crds['ph']
        ths = crds['th']
        
        # Get the lengths of the phi and theta arrays, and the total number of points
        # defined by that grid.
        I = len(phs)
        J = len(ths)
        N = I*J
        
        # Define each coordinate point by (r, phi, theta)
        c_sph = empty([N, 3])
        for i in range(I):
            for j in range(J):
                n = i*J + j
                c_sph[n,:] = array([1.0, phs[i], ths[j]])
        
        # Make an array of the time that is relevant (this is required to make the
        # transformation, because the transformation is of course time dependant.
        nows = empty(N, dtype = 'object')
        nows[:] = copy(time)
        ticks = Ticktock(nows, 'UTC')
        
        # Instantiate the Coords object with the above information
        sm_sph = Coords(c_sph, 'SM', 'sph', ticks = ticks)
        
        # Transform the coordinates into geographic coords.
        geo = sm_sph.convert('SM', 'car').convert("GEO", 'car').convert("GEO", 'sph')
        
        # From the geographic coordinates, get the latitudes and longitudes.
        lats = empty([I,J])
        lons = empty([I,J])
        for i in range(I):
            for j in range(J):
                n = i*J + j
                
                # NOTE: These are in DEGREES
                lats[i,j] = geo.data[n,1]
                lons[i,j] = geo.data[n,2]
        
        return (lats, lons)
    
    def _plot(self, time, *datasets, **plot_options):
        return Region._plot(self, time, *datasets, **plot_options)
    
    def _contour(self, ax_obj, data, **plot_options):
        '''
        API to plot a contour in a manner appropriate to this region.
        '''
        # Get the axis object to plot to
        ax = ax_obj.ax
        
        lats, lons = self.transfromCrds(data.time, data.crds) # Transform the coords
        x, y = ax(lons, lats) # Make a grid of x-y crds from the lats and lons
        
        # Get the relevant plot options
        log_scale = plot_options.pop('log', False)
        alpha = plot_options.pop('alpha', 0.75)
        cmap = plot_options.pop('cmap', None)
        levels = plot_options.pop('cb_scale', None)
        
        # scale the data
        data_set = data.data
        if log_scale:
            data_set = self.logShiftArray(data_set)
        
        cp = ax.contourf(x, y, data_set, alpha = alpha, cmap = cmap, levels = levels, 
                         zorder = 2)
        bar = self.fig.colorbar(cp, fraction = 0.1, pad = 0.1)
        return
        
    #TODO: Get this working
    def _streamline(self, ax_obj, data, **plot_options):
        '''
        API to plot streamlines in a manner appropriate to this region
        '''
        ax = ax_obj.ax
        
        lats, lons = self.transfromCrds(data.time, data.crds) # Transform the coords
        x, y = ax(lons, lats) # Make a grid of x-y crds from the lats and lons
        
        data_set = data.data
        print data_set['t'], data_set['p']
        ax.streamplot(x, y, data_set['p'], data_set['t'])
        return
    
    def viewOptsToString(self, view_opts):
        '''
        Function to convert the view options to a string that can be used to
        distinguish output files.
        '''
        if view_opts == 'default':
            pos = self.default_view
        else:
            pos = view_opts
        
        return "%s_%d_%d" % pos
    
    def getAxes(self, time, view_opts):
        '''
        Method to instantiate an axis object
        '''
        start_time = datetime.now()
        position = view_opts
        
        class Earth_Axes(object):
            def __init__(self, map_ax, view_opts):
                self.ax = map_ax
                self.view_opts = view_opts
        
        # Account for the view
        lat_0, lon_0 = self.getViewCenter(time, position)
        
        # Make the night-day image of earth
        image = self.makeNightDay(time, 0.5)
        image.flipVertical()
        
        # Make the map and make it look nice.
        m = My_Basemap(projection="nsper", lat_0=lat_0, lon_0=lon_0)
        m.warpimage(image, zorder = 0)
        m.drawcoastlines(linewidth = 0.5, color = '0.5', zorder = 1)
        m.drawcountries(linewidth = 0.25, color = '0.5', zorder = 1)
        m.drawparallels(range(-90,90,15), color='0.5', zorder = 1)
        m.drawmeridians(range(-180,180,30), color='0.5', zorder = 1)
        
        del image
        print "Time to make Earth axes:", datetime.now()-start_time
        return Earth_Axes(m, view_opts)
    
    def OrientationError(self, orientation):
        class OrientationError(Exception):
            pass
        string = 'Unknown orientation %s. Known orientations are %s.'
        return OrientationError(string % (orientation, str(self.good_args)))
    
    def getViewCenter(self, time, position):
        '''
        Get the view center from the position. Transfrom from the given coordinates to
        the used geographic coordinates.
        '''
        ref, lat, lon = position[:]
        ref = ref.upper()
        ticks = Ticktock(time, 'UTC')
        old = Coords(array([[1.0, lat, lon]]), ref, 'sph', ticks = ticks)
        new = old.convert(ref, 'car').convert('GEO', 'car').convert('GEO', 'sph')
        return new.data[0,1:]

class Magnetosphere(Region):
    '''
    This is the object for plotting the magnetosphere
    '''
    file_type = '3d'
    def __init__(self, *args, **kwargs):
        Region.__init__(self, *args, **kwargs)
        self.v_crds = {}
        self.points = {}
        return
    
    def _interp(self, data, cdim, view):
        '''
        Internal method to interpolate the 3d data down to a plane for a particular
        view.
        '''
        # I am looking for the point where the view coordinate is closest to the view
        # center. 
        distance_from_view = abs(cdim - view[1])
        w = where(distance_from_view == distance_from_view.min())[0].tolist()
        
        # Unless the specified coordinate is right in the middle between two coords,
        # there will only be one smallest difference, but I need to points to 
        # interpolate so I chose the next point that lands me on the other side of the
        # requested point.
        if len(w) == 1:
            if w[0] < view[1]:
                w.append(w[0] + 1)
            elif w[0] > view[1]:
                w.insert(0,(w[0] - 1))
        
        # Select the data ranges
        if view[0] == 'z':
            d_lo = data[w[0],:,:]
            d_hi = data[w[1],:,:]
        elif view[0] == 'y':
            d_lo = data[:,w[0],:]
            d_hi = data[:,w[1],:]
        elif view[0] == 'x':
            d_lo = data[:,:,w[0]]
            d_hi = data[:,:,w[1]]
        
        # Get the distance from the request point for the upper and lower 
        # available points
        t_lo, t_hi = cdim[w]
        
        # Linear interpolation
        data_set = d_lo + (view[1] - t_lo)*(d_hi - d_lo)/(t_hi - t_lo)
        return data_set
        
    def _contour(self, ax_obj, data, **plot_options):
        ax = ax_obj.ax
        view = ax_obj.view_opts
        
        crds = data.crds.copy()
        
        cdim = crds.pop(view[0])
        data_set = self._interp(data.data, cdim, view)
        
        x1, x2 = crds.items()
        
        # Get the relevant plot options
        log_scale = plot_options.pop('log', False)
        cmap = plot_options.pop('cmap', None)
        levels = plot_options.pop('cb_scale', None)
        
        # scale the data, if desired
        if log_scale:
            data_set = self.logShiftArray(data_set)
        
        # Plot
        cp = ax.contourf(x1[1], x2[1], data_set, cmap = cmap, levels = levels)
        bar = self.fig.colorbar(cp, ax = ax)
        
        self.__adjustAxes(ax, view)
        return
        
    def _streamline(self, ax_obj, data, **plot_options):
        ax = ax_obj.ax
        view = ax_obj.view_opts
        
        # Interpolate the 3d data down to 2d data depending on view
        crds = data.crds.copy()
        cdim = crds.pop(view[0])
        
        data_set = {}
        for key, d in data.data.iteritems():
            data_set[key] = self._interp(d, cdim, view)
        
        x1, x2 = crds.items()
        
        # Get plot options
        linecolor = plot_options.pop('linecolor', 'white')
        
        # Interpolate the field
        U, e1, e2 = self.interpolateField(data_set, x1, x2)
        
        # Plot
        vp = ax.streamplot(e1[1], e2[1], U[x1[0]], U[x2[0]], color = linecolor, minlength = 0.15)
        
        self.__adjustAxes(ax, view)
        return
    
    def __adjustAxes(self, ax, view):
        '''
        Internal method to enforce the plot settings.
        '''
        dims = view[2:]
        if dims != "default":
            ax.set_xbound(dims[:2])
            ax.set_ybound(dims[2:])
        
        ax.set_aspect('equal')
        ax.set_xlabel('Position along the Earth-Sun line [Earth Radii]')
        ax.set_ylabel('Position Perpendicular to the Ecliptic Plane [Earth Radii]')
        
        ax.tick_params(color = 'white')
        return
    
    def getAxes(self, time, view):
        '''
        Method to get the axes
        '''
        ax = self.fig.gca()
        
        ax.tick_params(color = 'white')
        
        # I need to save the view information for labeling purposes later.
        class Mag_Axes(object):
            def __init__(self, ax, view_opts):
                self.ax = ax
                self.view_opts = view_opts
        
        return Mag_Axes(ax, view)
    
    def viewOptsToString(self, view_opts):
        '''
        Method to convert the view options into a string for use in distinguishing
        different views in the outputs.
        '''
        return "%s%d_%d-%d_%d-%d" % tuple(view_opts)
    
    def interpolateField(self, V, X1_tpl, X2_tpl):
        '''
        API to interpolate the vector field. plt.streamplot does not work for non-
        square grids, so I have to square the grid. This is devilishly inefficient, 
        though, so I hope to eventually replace it with a custom-made cdef function. 
        plt.streamplot is also unable to plot 3d vector fields, which will also
        hopefully be mended by the custom job.
        
        ARGUMENTS:
        V      -- the dictionary of 2d arrays that defines the vector field.
        X1_tpl -- a tuple of the name and array for the x1 coordinate axis
        X2_tpl -- a tuple of the name and array for the x2 coordinate axis
        '''
        # Unpack the tuples
        k1, X1 = X1_tpl
        k2, X2 = X2_tpl
        
        # Find the total number of coordinate points defined.
        num_points = len(X1)*len(X2)
        
        # This will be used as the identifying trait of the coordinate system.
        pln_list = [k1,k2]
        pln_list.sort()
        pln_tpl = tuple(pln_list)
        
        # If the coordinates are the same, we can save ourselves some time by not 
        # making the new grid
        if not self.v_crds.has_key(pln_tpl):
            # For the bi-linear interpolator (griddata), I need an array of coordinate
            # pairs. `points' is an array of coordinate pairs with num_points rows and 
            # 2 columns
            points = empty([num_points,2])
            n = 0
            for x1 in X1:
                for x2 in X2:
                    points[n] = [x1,x2]
                    n = n + 1;
            
            # make the new (even) grid
            crds = {}
            crds[k1], crds[k2] = mgrid[min(X1):max(X1),min(X2):max(X2)]
            
            self.v_crds[pln_tpl] = crds
            self.points[pln_tpl] = points
        else:
            crds = self.v_crds[pln_tpl]
            points = self.points[pln_tpl]
        
        U = {}
        for comp, comp_vals in V.iteritems():
            # I need to create the empty array to fill. This is an array of values
            # for each point
            values = empty([num_points],dtype = 'float')
            n = 0
            # The components should be the same shape. If not, something is wrong.
            for i in range(V[comp].shape[1]):
                for k in range(V[comp].shape[0]):
                    values[n] = comp_vals[k,i]
                    n += 1
            
            # Now create that set of values
            gd_input = (crds[k1], crds[k2])
            U[comp] = transpose(griddata(points, values, gd_input, method = 'cubic'))
            
            # Translate new coords back into original format
            E1 = crds[k1][:,1]
            E2 = crds[k2][1,:]
            
        return (U,(k1,E1),(k2,E2))

#================================================================================
# Output File type objects
#================================================================================

class OTError(Exception):
    pass

class _Output_Type(object):
    '''
    There are three different types of output from the simulation: one for the 
    ionosphere (io), and two for the magnetosphere (2d and 3d). This object is 
    responsible for holding all the information necessary to interact with a particular
    output type, as well as containing methods to apply that information and access the
    data in the files.
    '''
    # For the magnetosphere 2d types, there are 3 separate outputs, one for each of the
    # coordinate planes (x, y, and z), each individually aligned. This is rather 
    # inconvenient, so that is not currently used, and I am not sure it is supported in
    # this code anymore.
    runme_entry = None
    f_out = None
    def __init__(self, file_key, crd_dims, sim_info_dict):
        # This is a method to hold all the information from the simulation that is 
        # relevant to this object. The idea is that the output object can be used
        # without a simulation running, for instance to look at data already output
        # by a previously run simulation.
        class Sim_Info(object):
            '''
            This is a struct to hold relevant information from the simulation
            '''
            def __init__(self, name, target, start_time):
                self.target = target
                self.start = start_time
                self.name = name
            
            def secFromStart(self, time):
                return (time - self.start).seconds
        
        # Get the information from the simulation information
        self.sim_dict = {}
        for name, info in sim_info_dict.iteritems():
            self.sim_dict[name] = Sim_Info(name, *info)
        
        self.key = file_key
        self.crd_dims = crd_dims
        self.h5_dict = {}
        self.avail = []
        self._crds = None
        self.re_fmt = ".*?\.%s\.%06d_p000000.h5"
        return
    
    def isNumAvail(self, num):
        '''
        API to determine if the output file 'num' has been written
        '''
        if num % self.f_out:
            err = 'num (%d) must be a multiple of output frequency (%d)'
            raise ValueError(err % (num, self.f_out))
        
        re_fmt = ".*?\.%s\.%06d_p000000.h5"
        
        # If 'num' is in self.avail, it's already been checked.
        if num in self.avail:
            return True
        
        # If I haven't yet looked for this num...
        if not self.h5_dict.has_key(num):
            self.h5_dict[num] = {}
        
        # Go through and make sure that the outputs from every simulation are 
        # available.
        re_patt = re_fmt % (self.key, num)
        
        # Look for outputs from all the running sims.
        for sim_name, sim_info in self.sim_dict.iteritems():
            if self.h5_dict[num].has_key(sim_name):
                continue # Nothing to do here
            
            # Use the directory objects 'find' API to determine if the file 
            # is available.
            if sim_info.target.find(re_patt):
                fname = sim_info.target.getFullName(re_patt)[0]
                self.h5_dict[num][sim_name] = _HDF5(sim_info.target.full + "/" + fname)
            else:
                # If ANY of the files are not available, the 'num' as a whole
                # is not yet available.
                if not len(self.h5_dict[num]):
                    self.h5_dict.pop(num) # Remove the empty entry from the dictionary
                return False
        
        self.avail.append(num)
        return True
    
    def getLatestNum(self):
        '''
        API to check if the next number in the sequence is available. This 
        function is recursive.
        '''
        new = self.f_out
        if len(self.avail):
            old = max(self.avail)
            new = old + self.f_out
        else:
            old = None
            
        if self.isNumAvail(new):
            return self.getLatestNum() # RECURSION
        
        return old
        
    def getNearestNum(self, time, sim_name = None):
        '''
        API to get the number hdf5 file available that corresponds to data
        taken closest to the requested time
        '''
        if sim_name != None:
            n = self.sim_dict[sim_name].secFromStart(time)
            return n - (n % self.f_out)
        else:
            n_dict = {}
            for sim_name, sim_info in self.sim_dict.iteritems():
                n = sim_info.secFromStart(time)
                n_dict[sim_name] = n - (n % self.f_out)
            
            return n_dict.values()[0] # For now, until I decide how I'm going to support multiple sims.
    
    def isTimeAvailable(self, time, sim_name = None):
        '''
        API to determine if a time is within the available range of times.
        '''
        num = self.getNearestNum(time, sim_name)
        latest = self.getLatestNum()
        
        if num == 0 or latest == None:
            return False
        elif num > latest:
            return False
        return True
    
    def loadData(self, ds, time, fail_ok = True, fail_ret = None):
        '''
        API to load data from the relevant hdf5 arrays.
        For usage of field_fmt and sub_fields, see _HDF5.loadValue()
        documentation.
        
        Failure occurs when there are no datasets available before 
        requested time. If failure is OK (fail_ok = True), then this
        function return `fail_ret', which is None by default. Otherwise 
        this function raises an exception.
        '''
        ret = {}
        num = self.getNearestNum(time)
        
        if not self.isTimeAvailable(time):
            if fail_ok:
                return fail_ret
            else:
                raise OTError('No outputs available before requested time! ' + 
                              'If this is not a problem, don\'t set fail_ok to False!')
        
        for sim_name, h5 in self.h5_dict[num].iteritems():
            h5.open()
            ret[sim_name] = h5.getValue(ds)
            h5.close()
        
        return ret
    
    def getCrds(self):
        '''
        API to get the coordinates from an hdf5 file (or xdmf in the case of iof).
        Because loading the coordinates takes some time, and the a file type 
        represented by this object should always have the same coordinates, they 
        are stashed in the object after being loaded the first time. Subsequent 
        calls to this function will return the stashed coordinates.
        '''
        # If we've already gotten the coordinates, just return what we have.
        if self._crds != None:
            return self._crds
        else:
            self._crds = {}
            
        if not len(self.avail):
            raise OTError("No outputs available from which to load coordinates.")
        
        if self.key == 'iof': # Then generate the points.
            h5 = self.h5_dict.values()[0].values()[0]
            h5.open()
            shape = h5.getValue('rrio').shape # rrio chosen randomly.
            h5.close()
            
            dph = 180.0/(shape[0] - 1)
            dth = 360.0/(shape[1] - 1)
                        
            phs = arange(-90.0, 90 + dph, dph)
            ths = arange(-180, 180 + dth, dth)
            
            crds = {'ph':phs, 'th':ths}
            self._crds = crds
            return crds
        
        else: # Read them from the hdmf5 file.
            raw_crds = dict.fromkeys(self.crd_dims)
            for dim in self.crd_dims:
                some_num_h5 = self.h5_dict.values()[0]
                some_h5 = some_num_h5.values()[0]
                some_h5.open()
                raw_crds[dim] = some_h5.getValue('V%s' % dim.upper())
                some_h5.close()
            
            self._crds = {}  # Coordinate data 
            for crd in self.crd_dims:
                c1 = raw_crds[crd]
                
                # The coordinates are defined a bit awkwardly for plotting. They mark
                # the corners of the values, not the point where the values exist, so I
                # have to shift the axes.
                n = max(c1.shape)
                c2 = empty([n-1])
                c2 = (c1[:-1] + c1[1:])/2.0
                self._crds[crd] = c2
                
            return self._crds
    
    def getNearestTime(self, req_time):
        '''
        API to get the time from the hdf5 file closest to the requested time req_time
        '''
        num = self.getNearestNum(req_time)
        
        time = None
        for h5 in self.h5_dict[num].itervalues():
            if time != None:
                if time != h5.getTime():
                    warn('Times from simulation outputs do not agree!')
            h5.open()
            time = h5.getTime()
            h5.close()
        
        return time
 
# Below are the declarations for the two output file-types used, the 2d ionosphere 
# output and the 3d magnetosphere output. The 2d magnetosphere output is currently not 
# in use for the sake of consistant handling elsewhere. I can choose my perspective any
# time to be anywhere if I use th 3d outputs, but I cannot with the 2d.
NO_OUTPUT = 600000
class Iono(_Output_Type):
    runme_entry = 'outtimeio'
    f_out = NO_OUTPUT
    def __init__(self, *args, **kwargs):
        _Output_Type.__init__(self, 'iof', ['lat', 'lon'], *args, **kwargs)

#class Mag2d(_Output_Type):
#    runme_entry = 'outtime2d'
#    f_out = NO_OUTPUT
#    def __init__(self, out_plane):
#        excluded_plane = My_Re.retrieve('p(x|y|z)_', out_plane)[0] 
#        crd_dims = ['x', 'y', 'z']
#        crd_dims.remove(excluded_plane)
#        _Output_Type.__init__(self, out_plane, crd_dims)

class Mag3d(_Output_Type):
    runme_entry = 'outtime3d'
    f_out = NO_OUTPUT
    def __init__(self, *args, **kwargs):
        _Output_Type.__init__(self, '3df', ['x', 'y', 'z'], *args, **kwargs)

class H5Error(Exception):
    pass

class NotOpenError(H5Error):
    pass
    
# Below is what is called a function wrapper. Basically what it does, is it executes a
# function while doing other stuff around that function. The syntax is a bit delicate,
# especially for use on object methods. It took some finaigling to get it to work.
# I would recommend reading up on some documentation before messing with it (Just
# "python function wrapper"...that should get you what you need).
# You can see it used below in the _HDF5 object. A function wrapper is placed before a
# a function definition with an @ in front of it, like
# 
# @wrapper
# def a_function(self, argumentative):
#       blah blah blah
#
def mustBeOpen(func):
    '''
    Function wrapper for the API's in the _HDF5 object that require the file to be
    open.
    '''
    def call(*args, **kwargs):
        # "self", which is a reference to the object, is the first argument of object 
        # API's, so I get the object reference from that.
        obj = args[0]
        
        # If the "file" attribute of this object is not assigned, then the file is not
        # open, which is a problem. So I except.
        if obj.file == None:
            description = "open() not called before calling %s." % func.func_name
            raise NotOpenError(description)
        
        # If all is well, proceed.
        ret = func(*args, **kwargs)
        return ret
    return call

class _HDF5(object):
    '''
    This is a slight modification of the hdf5 file object. The only change is
    to add the attribute 'num' that corresponds to the step number that 
    distinguishes outputs of the same type, as wel as defining several API's to more
    conveniently retrieve data and write the files.
    '''
    crd_patt = '<DataItem Name="(.*?)".*?>.*?:(.*?)\n.*?</DataItem>'
    val_patt = '<Attribute Name="(.*?)".*?>.*?<DataItem.*?>.*?:(.*?)\n.*?</DataItem>'
    def __init__(self, fname):
        self.name = fname
        self.fname = fname.split('/')[-1]
        self.file = None
        self.__xdmf = None
        self.n_users = 0
        self.thread_lock = RLock()
        return
        
    def open(self, rwa = 'r'):
        '''
        API to access the data in the file.
        '''
        # If I faile to open the file, I'll wait a bit and try again, just in case it 
        # was just a freak conflict.
        wait = 10
        
        # Also, in an attempt to avoid aforementioned freak conflicts, let's get a lock.
        self.thread_lock.acquire()
        try:
            # If the file is already open, just keep track that another user has been 
            # added. The file will not actually be closed until this is dropped back 
            # down to 0, which happens by calling the close command. See the close 
            # command comments for possible issues that may occur.
            self.n_users += 1
            
            # If the file hasn't already been openned...
            if self.file == None:
                # Open the file. Try a second time if it fails the first.
                retry = 2
                while retry:
                    try:
                        self.file = File(self.name, rwa)
                        
                        if self.__xdmf == None:
                            # Get the value-key relations from the xdmf file.
                            xdmf_file = open(self.name.replace("_p000000.h5", ".xdmf"), 'r')
                            xdmf_str = xdmf_file.read()
                            xdmf_file.close()
                            self.__xdmf = {}
                            vals = My_Re.findall(self.val_patt, xdmf_str, My_Re.DOTALL)
                            self.__xdmf.update(vals)
                            
                            # iof files store their coords weirdly. I believe I am now 
                            # generating the coordinates rather than trying to read 
                            # them, but I'm not sure. If I'm not, I should be. If I am, 
                            # this may not be useful.
                            if not My_Re.find("iof", self.fname):
                                crds = My_Re.findall(self.crd_patt, xdmf_str, My_Re.DOTALL)
                                self.__xdmf.update(crds)
                                
                        break # If I successfully found it.
                    except:
                        # I retry...the message pretty much says it all...
                        print "Excepted while trying to open data file %s (or associated xdmf). Retrying in %d seconds" % (self.name, wait)
                        sleep(wait)
                        retry -= 1
                        if not retry:
                            raise
        finally: # No matter what happens, I want to release this lock.
            self.thread_lock.release()
        
        return
        
    def close(self):
        '''
        Minor modification of the API that prevents a process that did not open
        this file from closing it. This is because processes get copies, not 
        references, so if the file was already closed, the process wouldn't know 
        it.
        '''
        # Get in line...
        self.thread_lock.acquire()
        try:
            # I keep track of how many times requests to open (and presumably use) this
            # file have been made, presumably from different threads (thus the need for
            # the thread lock above). One of them asking to close means one less user.
            # This could cause problems for a careless programmer, because the file may
            # be openned by one thread when another thread calles something that needs 
            # the file open, and thus the code would work, but only sometimes. It 
            # wouldn't be a bad idea to fix that.
            self.n_users -= 1
            
            # If there's no one else using this, close it.
            if self.file != None and self.n_users == 0:
                # Close the file and derefrence the dead object.
                self.file.close()
                self.file = None
        finally:
            self.thread_lock.release()
        
        return
    
    def getFilename(self):
        return self.fname
    
    __time_pat = "(\d{4}:\d{2}:\d{2}:\d{2}:\d{2}):"
    __time_fmt = "%Y:%m:%d:%H:%M"
    
    def __getOpenggcmKey(self):
        '''
        This is an internal API to retrieve the member key for the openggcm sub-group.
        This is necessary because the full name in the output is openggcm-<pointer>,
        and there is no way to predict the value of the pointer. Thus, the only way to
        proceed is to search through the member keys and find the one that contains the
        word 'openggcm'.
        '''
        my_key = None
        for member_key in self.file.iterkeys():
            if My_Re.find('openggcm', member_key):
                my_key = member_key
        if my_key == None:
            raise H5Error("Cannot find desired key")
        return my_key
    
    @mustBeOpen
    def getTime(self):
        '''
        API to get the time from an hdf5 file.
        '''
        # Get the key for the 'openggcm' sub-group.
        my_key = self.__getOpenggcmKey()
        
        # Retrieve the full time string (it's long, thus 'long_time_string'). If I
        # don't find it, show the contents of the hdmf5 file so I can hopefully find
        # what went wrong.
        try:
            long_time_str = self.file[my_key].attrs['time_str']
        except KeyError:
            self.showContents()
            raise
        
        # Get the bit of the string I actually want.
        re_time = My_Re.retrieve(self.__time_pat, long_time_str)[0]

        # Return a datetime object constructed from the above string.
        return datetime.strptime(re_time, self.__time_fmt)
    
    @mustBeOpen
    def setTime(self, time):
        '''
        API to set the time of an hdf5 file
        '''
        # Currently, only part of the time string is modified (the part I use).
        # For the purposes of generality, the entire thing should probably be
        # changed, but that would require more input information.
        
        # Get the file reference.
        f = self.file
        
        # Get the key for the 'openggcm' sub-group.
        my_key = self.__getOpenggcmKey()
        
        # Get the time string
        lts = f[my_key].attrs['time_str']  # Long Time String (lts)
        
        # Get the part of the 'long' time string that I want
        old = My_Re.retrieve(self.__time_pat, lts)[0]
        
        # Modify and replace it.
        lts_new = lts.replace(old, time.strftime(self.__time_fmt))
        
        # Set the new string.
        f[my_key].attrs['time_str'] = lts_new
        return
    
    @mustBeOpen
    def getValue(self, field_fmt, sub_fields = None):
        '''
        API to get a dataset from the currently open hdf5 file.
        `field_fmt` is either:
        
        - the full path of the desired dataset, in which case a numpy.ndarray 
          object containing the specified data is returned, or
            
        - a format with one '%s' which will be filled with each of the 
          sub_fields. A dictionary indexed by the sub_fields will be returned,
          with the corresponding numpy.ndarrays as values.
        
        `sub_fields` is an optional list of keys corresponding to sub-datasets, 
        such as x, y and z components of a vector. If sub_fields is specified,
        field_fmt MUST be a format string. If not, field format should be an
        ordinary string.
        '''
        f = self.file
        
        # I feel no desire to maintain this function twice.
        def loadArray(field):
            # There have been some issues reading the 'pot' file. This is a hopeful
            # quick fix.
            retry = 3
            while retry:
                try:
                    key = self.__xdmf[field]
                    break
                except:
                    retry -= 1
                    if not retry:
                        raise
            
            data_set = f[key]
            
            # Create an empty array
            a = empty(shape=data_set.shape, dtype=data_set.dtype)
            
            # Put the data into it
            a[:] = data_set[:]
            
            return a
        
        # If sub_fields is specified...
        if isinstance(sub_fields, list):
            if len(sub_fields):
                ret = {}
                for sub in sub_fields:
                    field = field_fmt % sub
                    ret[sub] = loadArray(field)
                return ret
        
        # Id sub_fields is none...
        elif sub_fields == None:
            return loadArray(field_fmt)
        
        # If something didn't validate above...
        raise H5Error('Could not load array with field_fmt "%s" and sub_fields "%s"' % (field_fmt, str(sub_fields)))
    
    @mustBeOpen
    def showContents(self, group_name=None):
        '''
        API to show the contents of an hdf5 file. Uses a recursive method to list
        everything in the hdf5 file.
        '''
        # Define a recursive function to probe the depths of the hdf5 file.
        def showMembers(name=None, level=0):
            if name:
                # If we are looking into a particular group, only take the group.
                obj = self.file[name]
                fmt = "%s/%%s" % name
            else:
                # Otherwise, take the whole object.
                obj = self.file
                fmt = "/%s"
            
            # Set the spacing and define some basic component strings.
            short_str = ""
            for _ in range(level):
                short_str += "    "
            long_str = short_str + "  "
            n = "\n"
            
            # Prime the string that will hold all the results from this level and
            # below.
            string = ''
            
            # First check for any attributes at this level and add them to the 
            # string.
            attrs = obj.attrs
            if len(attrs):
                string += (short_str + "Attributes:" + n)
                for attr, value in attrs.iteritems():
                    if isinstance(value, str):
                        value = "'%s'" % value
                    string += (long_str + "%s: %s" % (attr, value) + n)
            
            # Next, look into the member objects
            try:
                members = obj.keys()
                if len(members):
                    if len(members) > 1:
                        string += (short_str + "Members:" + n)
                    for member in members:
                        new_name = fmt % member
                        string += "%s%s name: '%s'" % (long_str, member, new_name)
                        string += showMembers(new_name, level + 1)  # <- RECURSION
            except:
                pass
                
            # Try to add the data type.            
            try:
                string += (' Datatype: ' + str(obj.dtype) + n)
            except:
                string = n + string
                pass
            
            return string
        
        # Now apply the recursion function and print the result!
        if group_name != None:
            ret = showMembers("/%s" % group_name)
        else:
            ret = showMembers()
        print ret
        return ret
        
    @mustBeOpen
    def setRunName(self, run_name):
        '''
        Stub for an API that will set the run name in the hdf5 file.
        '''
        # This is a stub to be added later

#================================================================================
# Simulation interface
#================================================================================

class ExecError(Exception):
    '''
    Exception object to be raised out of the Shell_Command class or children thereof.
    '''
    def __init__(self, msg):
        full_msg = "An error has occured while attempting to execute a shell command:\n" + msg
        Exception.__init__(self, full_msg)
        return

class Shell_Command(object):
    '''
    Object to hand the execution of commands in a virtual terminal.
    '''
    def __init__(self, name, cmd, now=False, loc=None, catches=None):
        self.name = name
        self.__command = cmd
        self.__loc = loc
        self.__catch_list = catches
        self.__out_fmt = LOG_DIR + "/%s.%%s" % self.name
        self.__is_running = False
        
        # If the user wants to just run this all in one go, then let them.
        # Used this way, this object behaves exactly like an ordinary
        # function. This only works because it does not need to return any
        # values.
        if now:
            self.start()
            self.join()
            self.checkOutputs()
            
        self.__proc = None
        return
    
    def __str__(self):
        '''
        Simple method to return a nice string if str(Shell_Command_Instance) is called.
        '''
        if isinstance(self.__command, list):
            string = ''
            for arg in self.__command:
                string += " %s" % arg
        else:
            string = self.__command
        return string
    
    def __del__(self):
        '''
        This is an attempt to avoid hanging processes.
        '''
        if self.__proc != None:
            self.__proc.kill()
        return
    
    def isRunning(self):
        '''
        Return if the command is running
        '''
        return self.__is_running
    
    def start(self):
        '''
        API to start the execution
        '''
        # If I need to be executing this somewhere else, go there.
        # This may cause problems if threading.
        if self.__loc:
            os.chdir(self.__loc)
        
        # Get the output and error files.
        self.__out_file = open(self.__out_fmt % "out", 'w')
        self.__err_file = open(self.__out_fmt % "err", 'w')
        
        # Execute the command
        self.__proc = Popen(self.__command,
                            stdout=self.__out_file,
                            stderr=self.__err_file)
        
        self.__is_running = True
        
        # If I executed this somewhere else, come back.
        if self.__loc:
            os.chdir(PWD.full)
            
        return
    
    def join(self):
        '''
        API to wait for the process to end.
        '''
        # Wait for the process to complete.
        self.__proc.wait()
        
        # Close the output and error files.
        self.__out_file.close()
        self.__err_file.close()
        
        # Update my status.
        self.__is_running = False
        return
    
    def checkOutputs(self):
        '''
        After the process is joined, this API may be used to check for signs of failure
        in the output using the key words stored in `self.__catch_list', which is
        optionally defined in __init__. If nothing is defined there, this function does
        nothing. This is not fool-proof because you have to know what kind of errors 
        might occur and what to look for in the output when they do, but it's something.
        
        Note: This WILL EXCEPT (ExecError) if the process is still running.
        '''
        # If the executable is still running, the output files are still being written,
        # so I can't read them. This is meant to be called AFTER the executable's 
        # process is joined.
        if self.__is_running:
            raise ExecError("Cannot check outputs while command is running")
        
        # Open the output file and read the contents.
        out_file = open(self.__out_fmt % 'out', 'r')
        line_list = out_file.readlines()
        out_file.close()
        
        # Use regular expressions to look at each line from the file and check for the
        # catchs. If one is found, except.
        msg_fmt = "Caught '%s' in '%s' (line %d/%d) of output from '%s'."
        N = len(line_list)  # total number of lines
        if self.__catch_list:
            for n, line in enumerate(line_list):
                for catch in self.__catch_list:
                    if My_Re.find(catch, line, My_Re.IGNORECASE):
                        # Remove any \n's from line string
                        line = line.replace('\n', '')
                        
                        # Raise the exception
                        raise ExecError(msg_fmt % (catch, line, n, N, self.__str__()))
        return

class Openggcm(Logged_Object):
    '''
    This is an object to interface with the openggcm simulation. It is responsible for
    the modification and execution of the runme file which initializes the simulation
    and monitoring the progress of the simulation indirectly by looking for outputs.
    
    This object also provides the simulation_emulator utility that allows one to
    imitate the behavior of the simulation for testing purposes.
    '''
    NOT_STARTED = 'NOT_STARTED'
    PRIMING = "PRIMING"
    PRE_RUN = 'PRE_RUN'
    MAIN_SEQ = "MAIN_SEQ"
    FINISHED = "FINISHED"
    
    def __init__(self, start_time, end_time, **params):
        # Set name
        name = end_time.strftime("run%Y%m%d%H")
        
        # Create the run directory (making it may change name)
        run_dir = getDirectory(params.pop('run_dir', RUN_DIR))
        self.dir = run_dir.makeSubDirectory(name, persistent=True, temp=True)
        self.name = self.dir.name
        
        # Init logging
        Logged_Object.__init__(self, self.name)
        self.info("%s Openggcm object has been initialized" % self.name)
        
        # get directories
        self.target = self.dir.getSubDirectory("target")
        self.sim = getDirectory(params.pop('sim_dir', SIM_DIR))
        
        # Set time values
        self.start_time = start_time
        self.end_time = end_time
        self.nums = {}
        self.nums[self.MAIN_SEQ] = timedelta(hours = params.pop('prerun', 0)).seconds
        self.nums[self.FINISHED] = (end_time - start_time).seconds
        
        # Set defaults
        if not params.has_key('procs'):
            params['procs'] = [2, 1, 1] # Default to the lowest number
        
        if not params.has_key('dims'):
            params['dims'] = [-100,200,-100,100]
        
        if not params.has_key('res'):
            p = params['procs']
            c = 50  # constant for adjustment.
            params['res'] = [c * p[0], c * p[1], c * p[2]]
        
        # Set some general parameters
        self.emulate = params.pop('emulate', False)
        self.parameters = params
        self.ace_loc = {}
        
        # Log the information about this simulation
        info_str = ""
        for parameter, value in self.parameters.iteritems():
            info_str += "%s: %s  " % (parameter, str(value))
        self.debug(info_str)
        
        # Here I define the solar wind data file object which is responsible for taking
        # data provided by the ACE solar wind satellite available online. The entire
        # object should probably eventually be merged with this object eventually.
        self.swd = swdata('www.swpc.noaa.gov/ftpdir/lists/ace', self.dir.full + "/swdata",
                           self.start_time, self.end_time)
        
        # Stage indicator: to check this from another function, there are a set of 
        # methods to check for specific stages in the running of the simulation.
        self.__stage = self.NOT_STARTED
        
        # This will hold the output-type objects that this simulation will output. All
        # objects must be added using addOutObj() BEFORE the start() is called.
        self.out_obj_dict = {}
        return
    
    def addOutObj(self, **out_obj_dict):
        '''
        API to add an output object to the Simulation object's dictionary. All desired 
        output objects must be added BEFORE the runme file is written and run. (i.e. 
        before Simulation.start() is called)
        '''
        if self.__stage != self.NOT_STARTED:
            raise Usage_Error("You may only add objects BEFORE the simulation is started.")
        self.out_obj_dict.update(out_obj_dict)
        return
    
    def secFromStart(self, time):
        '''
        Convenience method to retrive the number of seconds from the start of the 
        simulation (in simulation time, NOT real time.)
        '''
        return (time - self.start_time).seconds
    
    def __checkStatus(self):
        '''
        Internal API to check the status of the simulation
        '''
        # If we are in one of the first to stages, these are set after certain API's
        # have been called, so I don't need to actively check anything.
        if self.__stage == self.PRIMING or self.__stage == self.NOT_STARTED:
            return self.__stage
        
        # For the rest of the stati, I check to see what outputs have been found, and
        # from that and the initialization information, I can determine what stage I
        # am in. I am currently having the simulation object distinguish between the
        # 'pre-run', in which the simulation runs for a bit untill it settles into a
        # relatively steady state. This is a rather artificial and imposed observation
        # on the simulatino object's part, and I'm thinking it may be good to move it
        # higher up the chain (to whatever is calling the simulation class methods).
        past_main = True
        past_end = True
        for obj in self.out_obj_dict.itervalues():
            latest = obj.getLatestNum()
            if latest <= self.nums[self.MAIN_SEQ]:
                past_main = False
            elif latest <= self.nums[self.FINISHED]:
                past_end = False
        
        if not past_main:
            self.__stage = self.PRE_RUN
        elif not past_end:
            self.__stage = self.MAIN_SEQ
        else:
            self.__stage = self.FINISHED
        
        return self.__stage
       
    def hasStarted(self):
        '''
        API to get if the simulation has started yet. Returns True (it has) or 
        False (it hasn't).
        '''
        return self.__stage != self.NOT_STARTED
        
    def inPreRun(self):
        '''
        API to get if the simulation is in prerun. Returns True or False
        '''
        return self.__stage == self.PRE_RUN
    
    def inMain(self):
        '''
        API to get if the simulation is in the main part of the run. Returns True
        or False
        '''
        return self.__checkStatus() == self.MAIN_SEQ
        
    def isFinished(self):
        '''
        API to get if the simulation is finished. Returns True of False
        '''
        return self.__checkStatus() == self.FINISHED
    
    def getStage(self):
        '''
        API to get a copy of self.__stage
        '''
        return self.__checkStatus()
    
    def __editStr(self, runme_str):
        '''
        API to edit the runme string.
        '''
        t_fmt = "%Y:%m:%d:%H:%M:%S"
        start_time = self.start_time.strftime(t_fmt)
        end_time = self.end_time.strftime(t_fmt)
        
        params = self.parameters.copy()
        convert_dict = {'dims' : ['xx1', 'xx2', 'yy2', 'zz2'],
                        'procs': ['npx', 'npy', 'npz'],
                        'res'  : ['nx', 'ny', 'nz']}
        for bunch, bits in convert_dict.iteritems():
            params.update(zip(bits, params.pop(bunch)))
        
        # This is where all the inputs to the runme are defined
        values_dict = {"setenv OPENGGCMDIR" : self.sim.full,
                       "STARTTIME"          : start_time,
                       "ENDTIME"            : end_time,
                       "OUTDIR"             : self.target.full}
        
        values_dict.update(params)
        
        # If the values were retrieved by "get_swdata". The satellite position 
        # doesn't change much, so it's not a catastrophe if we don't update it 
        # all the time.
        if len(self.ace_loc): 
            values_dict["MOX"] = str(self.ace_loc['x'])
            values_dict["MOY"] = str(self.ace_loc['y'])
            values_dict["MOZ"] = str(self.ace_loc['z'])
        
        # Get the output times.
        for obj in self.out_obj_dict.itervalues():
            values_dict[obj.runme_entry] = "-%d" % obj.f_out
        
        # For the time being, I am not going to support direct 2d outputs. 2d plots can
        # be made from the 3d data files.
        ## Get the outplanes
        #for key, value in OUT_PLANES_2D.iteritems():
        #    if value == None:
        #        values_dict['outplane%s' % key] = '100000' #None
        #    else:
        #        values_dict['outplane%s' % key] = str(value)
        
        # Make the changes to the file string
        for location, value in values_dict.iteritems():
            if not My_Re.find("(%s\s+%s)\s" % (location, value), runme_str):
                old_str = My_Re.retrieve("(%s\s+\S+)\s" % location, runme_str)[0]
                new_str = "%-16s" % str(location)
                if isinstance(value, list):
                    for item in value:
                        new_str += (item + " ")
                else:
                    new_str += str(value)
                        
                # Replace the entries on the runme string.
                self.debug("Making replacement in runme: old '%s' and new '%s'" % (old_str, new_str))
                runme_str = runme_str.replace(old_str, new_str)
        
        return runme_str
    
    def __updateAndRunRunme(self):
        'Function to update and run the runme.'
        name = "/runme"
        f_tmplt = open(LOCAL.full + '/ref' + name, "r")
        runme_str = f_tmplt.read()
        f_tmplt.close()
        runme_str = self.__editStr(runme_str)
        
        # Write the modified contents to the file.
        self.debug("Writing runme.")
        f = open(self.dir.full + name, "w")
        f.write(runme_str)
        f.close()
        
        # Run the runme
        self.__runRunme(name)
        
        return
    
    def __runRunme(self, name):
        '''
        Internal API to run the runme file that compiles the simulation code.
        '''
        # If I am only pretending to have a simulation running, the runme is not run 
        # (that's the entire point of emulating the simulation), so the target directory
        # will not be created. Thus I create it (unless it already exists from a 
        # previous run, which doesn't happend often).
        if self.emulate:
            self.info("Using Openggcm_Mask, not running runme. Simulation will not be run, but its behavior will be imitated.")
            self.target.makeIfNeeded()
            return
        
        # make the runme executable
        self.debug("chmod-ing runme")
        os.chmod(self.dir.full + name, (stat.S_IRWXU or stat.S_IRWXG or stat.S_IRWXO))
        
        # The command is the name of the file. (The file name string should begin with a /)
        command = ".%s" % name

        # Run the command, and let me know if I keyboard interupt while executing.
        # The Shell_Command function object (which executes when called with now=True)
        # will look for expressions containing any of the words/phrases/regexes in 
        # `catches` in the runme's output. If it finds any, it will except.
        try:
            self.info("Running runme.")
            Shell_Command('runme', command, loc=self.dir.full, now=True, catches=["FATAL", "Error"])
        except KeyboardInterrupt:
            self.ERROR("Keyboard interrupt during runme!")
            raise

        return
    
    def start(self):
        '''
        Execute all the prerequisites, compile the simulation code (this takes a while)
        and then begin the simulation.
        '''
        # Run the runme
        self.__stage = self.PRIMING
        
        # Retrieve and write the solar wind data
        self.swd.startSwdata()
        
        # Updata and run the runme
        self.__updateAndRunRunme()
        
        if self.emulate:
            # Start the simulation emulator in a separate process
            sample_out = LOCAL.getSubDirectory("sample_out")
            self.__proc = Process(target=self.__simulation_emulator, args=[sample_out])
            self.__proc.daemon = True
        else:
            # Start the simulation in a separate thread (to be clear, the other
            # thread is not actually running the simulation, it is simply sending
            # the shell command to run it.
            p = self.parameters['procs']
            np = p[0] * p[1] * p[2] + 2
            cmd = ["nice", "-n", "19", "mpirun", "-np", str(np), "./openggcm"]
            print self.target.full
            print self.target.getParent().name
            self.__proc = Shell_Command('mpirun', cmd,
                                        loc=self.target.full,
                                        catches=['entering death'])
        # Start the simulation (or emulator)
        self.__proc.start()
        self.__stage = self.PRE_RUN
        return
    
    def join(self):
        '''
        API to check that the simulation ran without problems.
        '''
        self.debug("Waiting for simulation to complete")
        # Wait for the solar wind data to end
        self.swd.joinSwdata()
        print "Joined swdata monitor"
        
        # Wait for the simulation to end
        self.__proc.join()
        
        # We'll assume the emulator was OK.
        if self.emulate:
            return
        
        # Check the output of the simulation for keywords implying the simulation
        # failed to run all the way through
        self.__proc.checkOutputs()
        return
    
    def __simulation_emulator(self, sample_out):
        '''
        This function is for testing purposes. Testing this python code does not really
        require that the simulation be run. Running the simulation takes a LONG time.
        This allows me to (almost*) fully test my code without having to wait so long.
        
        *This does NOT test whether the runme and simulation work. Erroneous entries
        to the runme will not necessarily be caught, nor will simulation failures.
        Nevertheless, this will allow you to catch most of your errors.
        '''
        from shutil import copy as cp
        
        wait_time = 30
        def copyOver(fname, num=None):
            cp("%s/%s" % (sample_out.full, fname), self.target.full)
            new_name = fname.replace("sample_out", self.name)
            if num != None:
                new_name = new_name.replace(".%06d" % 0, ".%06d" % num)
            os.rename("%s/%s" % (self.target.full, fname), '%s/%s' % (self.target.full, new_name))
            print "Copied", new_name
        
        ti = self.start_time
        tf = self.end_time
        h5_dict = {}
        for key, obj in self.out_obj_dict.iteritems():
            fname_patt = 'sample_out\.%s\.000000_p000000.h5' % (obj.key)
            if sample_out.find(fname_patt):
                h5_dict[key] = _HDF5(sample_out.full + '/' + sample_out.getFullName(fname_patt, only_one = True))
            else:
                # If this exception is raised, you probably need to update the 
                # contents of sample_out directory.
                raise Exception('Could not find file %s' % fname_patt)
            
            # Copy over the base xdmf files.
            xdmf = sample_out.getFullName("sample_out\.%s\.xdmf" % obj.key, only_one=True)
            copyOver(xdmf)
        
        t = dict.fromkeys(h5_dict.keys(), ti)
        while min(t.values()) < tf:
            sleep(wait_time)
            for key in t.iterkeys():
                obj = self.out_obj_dict[key]
                t[key] += timedelta(seconds = obj.f_out)
                
                h5_obj = h5_dict[key]
                
                diff = int((t[key] - ti).seconds)
                
                # Edit the contents of the file to reflect the appropriate time
                h5_obj.open('a')
                h5_obj.setTime(t[key])
                fname = h5_obj.getFilename()
                h5_obj.close()
                
                # Copy it over to the Target directory
                copyOver(fname, diff)
                
                # Copy over the associated xdmf file
                xdmf = sample_out.getFullName("sample_out\.%s\.\d+\.xdmf" % obj.key, only_one=True)
                copyOver(xdmf, diff)
        
        return

#TODO: I should probably move all this into the Openggcm object.
#================================================================================
# Solar Wind data retrieval object
#================================================================================
class SwDataError(Exception):
    pass

class swdata(Logged_Object):
    '''
    Object to monitor the ACE solar wind data website, retrieve the latest data,
    and write that data to the swdata file (or possibly files...haven't decided
    yet).
    '''
    keys = ('Bx', 'By', 'Bz', 'Vx', 'Vy', 'Vz', 'N', 'P', 'Nx', 'Ny', 'Nz')
    BAD = 'BAD'
    OLD = 'OLD'
    def __init__(self, address, out_file, start_time, end_time):
        self.address = "http://" + address
        self.start_time = start_time
        self.end_time = end_time
        self.out_file = out_file
        
        self.used_data_type_list = ["swepam", "mag"]
        
        self.__num_omitted = 0
        
        # Initialize the logger
        Logged_Object.__init__(self, "swdata")
        
        self.format_str = "%(time)10.0f"
        for key in self.keys:
            self.format_str += " %%(%s)5.3f" % key
        self.format_str += "\n"
        
        self.monitor = None
        return
    
    def __openSite(self, fname):
        '''
        Internal API to open a website given by `fname`. Retry in case of lag or random
        transitory problems.
        '''
        self.debug("Openning website")
        retry = 3
        while retry:
            try:
                ret = urlopen(self.address + '/' + fname)
                break
            except URLError:
                retry -= 1
                if not retry:
                    raise
                self.WARNING("Failed to open website %s...retrying." % fname)
                sleep(2)
        self.debug("successfully opened site.")
        return ret
    
    def __getNewData(self, start_time, end_time):
        '''
        API to get the latest solar wind data. This runs using the 2-hour files
        produce by ACE, updating the swdata file as new data becomes available.
        There will generally be a roughtly 2-5 minute lag between real time and
        the latest available file that has good data.
        '''
        # Replace the input start and end times with times zeroed to the
        # nearest hour
        start_time = start_time.replace(second = 0, microsecond = 0)
        end_time = end_time.replace(second = 0, microsecond = 0)
        
        # For ease of use later
        m = timedelta(minutes = 1)
        fmt = "ace_%s_1m.txt"
        
        # I am storing the raw data in python memory as a dictionary (line_lists) 
        # indexed by data type name strings (d_type).
        line_lists = {}
        def updateLists():
            for d_type in self.used_data_type_list:
                fname = fmt % d_type
                line_lists[d_type] = self.__getList(fname)
        updateLists() # Run the function to actually get a copy of the lists
        
        # Because the different datatypes to process a given time, but the lists
        # don't update all at the same time, I need to keep track of how far off
        # they are from eachother. This will need to be done every time new
        # lists are successfully acquired.
        offset = {}
        def getOffset():
            times = {}
            for d_type in self.used_data_type_list:
                l = My_Re.retrieveNums(line_lists[d_type][-1])
                times[d_type] = (l[3]/100)*60 + l[3]%100 # minutes
            
            # it's a pretty safe bet that any differences will only be by one
            lesser = min(times.values())
            for d_type, time in times.iteritems():
                if time > lesser:
                    offset[d_type] = -1
                else:
                    offset[d_type] = 0
        
        # Define a sub-function that processes the data from the i-th line (indexed
        # by whichever list whose most recent good value is the oldest).
        def getData(i, check = True):
            raw_data = {}
            for d_type, di in offset.iteritems():
                raw_data[d_type] = My_Re.retrieveNums(line_lists[d_type][i + di])
            return self.__parseLine(raw_data, check)
        
        # Sub-function used to write the data.
        def write(data):
            s = self.__makeString(data)
            self.__writeFile(s, 'a')
            return
        
        # Put it all together in a loop: while the currently searched for time (time) 
        # is less than the requested end time, keep searching.
        time = start_time.replace()
        try:
            while time < end_time:
                # Read the lates file from ACE, and get the list of lines
                updateLists()
                
                # The files are not updated at EXACTLY the same time, so the indexing 
                # may not match
                getOffset()
                
                # Go back until the data is good
                i = -1
                data = getData(-1)
                try:
                    while data == self.BAD:
                        i -= 1
                        data = getData(i)
                except IndexError:
                    # If I can't find any good data, then go back far enough that the 
                    # data should at least be permanently bad, not just transitorily bad.
                    i = -5
                    data = getData(i, check = False) # Don't check for clean data.
                 
                if data['t'] < time:
                    sleep(15) # either the time isn't available or it's bad
                elif data['t'] > time:
                    i -= (data['t'] - time).seconds/60
                    if i <= -120: # This would excede the range of the list
                        print time, data['t']
                    data = getData(i)
                    if data != self.BAD:
                        write(data)
                    else:
                        msg = "Omitting data for %s due to faulty data." % str(time)
                        self.WARNING(msg)
                    
                    # If the data was bad, at this point it will likely never be good. 
                    # So just skip it.
                    time += m
                else:
                    write(data)
                    time += m
                
                sleep(0.1)
        except:
            print "Ending loop at time %s and index %d" % (time, i)
            raise
            
        return
    
    def __parseLine(self, raw_data, check = True):
        '''
        Function to parse a line of data.
        Input:
        raw_data -- one line from each type of file in a dict indexed by the data type 
                    name strings. The lines should both apply to the same time. If they
                    are not, this function will except.

        Output:
        data_dict -- dict of values keyed using the keys in `self.keys` (e.g. Bx for 
                     the x-comp of the B-field)
        '''
        # Make sure the times from the different file types agree
        time = None
        for data in raw_data.itervalues():
            this_time = datetime(*data[:3], hour = data[3]/100, minute = data[3]%100)
            if time == None:
                time = this_time
            elif this_time != time:
                msg = "Can only parse data when line times from files agree."
                msg += "Got times %s and %s" % (str(time), str(this_time))
                raise SwDataError(msg)
        
        # Prime the output dictionary using the keys and setting them to a default of 0
        data_dict = dict.fromkeys(self.keys, 0.0)
        data_dict['Nx'] = -1.0  # non-zero default
        
        # If we're supposed to check that the data's good...
        if check:
            # None of the values here should be negative
            neg_data = False
            for j in range(7, 10):
                if float(raw_data['swepam'][j]) < 0:
                    neg_data = True
                    break
        
            # Check for certain types of problems in the data. If the data has one of these
            # issues, return the `self.BAD` string. Otherwise, continue.
            if raw_data['mag'][6] > 0 or raw_data['swepam'][6] > 2 or neg_data:
                return self.BAD
        
        # Retrieve the magnetic field components.
        data_dict.update(zip(['Bx', 'By', 'Bz'], raw_data['mag'][7:10]))
        
        # ACE gives PROTON density. Assuming neutral plasma, there would be 2x as many 
        # charged particles.
        n = raw_data['swepam'][7]*2
        if n < 2.5:
            n = 2.5
        P = 1.3807e-5 * n * raw_data['swepam'][9]  # ~ k*n*T
        data_dict.update(zip(['P', 'N', 'Vx'], [P, n, -raw_data['swepam'][8]]))
        data_dict['t'] = time
        return data_dict
    
    def __getList(self, fname):
        '''
        Internal function to get a list of lines from one of the ACE swdata files.
        '''
        # Open the file and read the contents
        s = self.__openSite(fname)
        s_str = s.read()
        s.close()
        del s
        
        # Break the string into a list of lines
        l = s_str.splitlines()
        
        # Remove the lines that don't contain data (the header)
        while My_Re.find('[^0-9 .+-e]+', l[0]):
            l.remove(l[0])
        
        # return the result
        return l
    
    def __getOldData(self, start_time, end_time):
        '''
        API to get the solar wind data from the website starting at start_time and 
        ending at end_time (floored to nearest second and microsecond). This function,
        unlike __getNewData, only writes to the swdata file ONCE, as it is not
        continuously looking for the new data.
        '''
        start_time = start_time.replace(second = 0, microsecond = 0)
        
        # Define a general fromat for the ace data files
        fmt = "%%Y%%m%%d_ace_%s_1m.txt"
        
        # Set some datetime values for convenience
        day = timedelta(days=1)
        m = timedelta(minutes = 1)
        
        # Prime this for writing.
        string = ''
        raw_data = {}
        time = start_time - timedelta(hours = 2)
        
        # Subfunction to get a list index from a time    
        def getIndex(time):
            return (time - time.replace(hour = 0, minute = 0, second = 0, microsecond = 0)).seconds/60
        
        # I am storing the raw data in python memory as a dictionary (line_lists) 
        # indexed by data type name strings (d_type). I also keep track of how long
        # each file is.
        line_lists = {}
        N_dict = {}
        def getList(time):
            for d_type in self.used_data_type_list:
                fname = time.strftime(fmt % d_type)
                line_lists[d_type] = self.__getList(fname)
                N_dict[d_type] = len(line_lists[d_type])
        
        # Subfunction to create the input dict for the `self.__parseLine` function and
        # get then pass along the parsed data.
        def processData(i):
            for d_type in self.used_data_type_list:
                raw_data[d_type] = My_Re.retrieveNums(line_lists[d_type][i])
            return self.__parseLine(raw_data)
        
        # Get the index for the pre-start primer.
        i_p = getIndex(time)
        getList(time)
        data = processData(i_p)
        
        # In case that was a bad value, try again...
        i = i_p
        i_have_gone_back = False
        while data == self.BAD:
            i -= 1
            
            # In the rare even that the first acquisition fails and I end up going into
            # the previous day, go to the previous day's file. This will only work for
            # one day. It seems very unlikely that there would be more bad data than that. 
            if i < 0 and not i_have_gone_back:
                i_have_gone_back = True
                getList(time - day)
            
            data = processData(i)
        
        # Add the result to the string
        string += self.__makeString(data)
        
        # In case I had to go back a day to get the primer value (i.e. if it is <2 in 
        # the morning UTC).
        if start_time.day != time.day:
            getList(start_time)
        
        # Now I get to loop through the rest of the times.
        time = start_time
        this_day = time.day
        n_omitted = 0
        n_total = 0
        start_new = False
        while time < end_time:
            # Check to see if I have to switch days.
            if time.day != this_day:
                this_day = time.day
                getList(time)
            
            i = getIndex(time)
            try:
                data = processData(i)
            except IndexError: 
                if (datetime.utcnow() - time).seconds < 2*3600:
                    # Occasionally the server takes a long time to update the daily files,
                    # but if not, they have often updated the files containing the last
                    # two hours.
                    start_new = True
                    break
                else:
                    raise
            
            # Check to see if the data is bad. If not, parse and reformat the data and 
            # add it to the string. If so, omit it.
            if data != self.BAD:
                string += self.__makeString(data)
            else:
                n_omitted += 1
                self.WARNING("Omitting data for %s due to faulty data." % str(time))
            
            # Increment time and the total
            n_total += 1
            time += m
        
        # Write the file
        self.__writeFile(string, 'w')
        self.debug("%d/%d lines omitted due to bad data." % (n_omitted, n_total))
        
        # If the daily files were not quite up to date, get the remaining data from the
        # two hour data files.
        if start_new:
            self.__getNewData(time, end_time)
        return
    
    def __makeString(self, data):
        '''
        API that builds the swdata file line from a line's worth of data.
        '''
        time = data['t']
        
        # This is annoying, but differences of datetimes are never negative, 
        # and negative is what I want.
        if time > self.start_time:
            t = int((time - self.start_time).seconds / 60)
        else:
            t = -int((self.start_time - time).seconds / 60)
        
        # One last value to go in the data_dict
        data['time'] = t
        
        # Add to the string
        return self.format_str % data
    
    def __writeFile(self, s, m):
        '''
        Write the data to the file(s)
        '''
        f = open(self.out_file, m)
        f.write(s)
        f.close()
        return
    
    def startSwdata(self):
        '''
        Function to retrieve solar wind data from ACE Satellite Daily and Minutely data
        files. If the end time defined in the object initiation is beyond the present,
        this function will spawn a thread to monitor the minutely files and then 
        return. To join with that thread, call `joinSwdata` defined below.
        '''
        # Get the current time (floored to the nearest minute) and check to see if all
        # of the requested data is in the past or if it will be necessary to wait for
        # data from the minute-ly data files. In both cases get any data that I need
        # from the daily data files.
        now = datetime.utcnow().replace(second = 0, microsecond = 0)
        continuous = True
        if self.start_time < now:
            if self.end_time > now:
                end_old = now
            else:
                end_old = self.end_time
                continuous = False # The data is encompassed in old data
            
            self.__getOldData(self.start_time, end_old)
        else:
            secs = (now - self.start_time).seconds
            self.info("Start time is %d seconds away. Sleeping until then..." % (secs))
            sleep(secs)
            self.info("Now beginning")
            self.__getOldData(self.start_time, self.start_time)
        
        # If I need to get new data, then start a thread to do so.
        if continuous:
            print "starting continuous observation"
            self.monitor = Thread(target = self.__getNewData, 
                                  args = (now, self.end_time), 
                                  name = "observer_thread")
            self.monitor.daemon = True
            self.monitor.start()
        
        return
    
    def joinSwdata(self, timeout = None):
        '''
        Join with the monitor thread spawned by `startSwdata`
        '''
        if self.monitor != None:
            self.monitor.join(timeout)
        return

#================================================================================
# Queuing object
#================================================================================

class QueueError(Exception):
    pass

class Plot_Queuer(object):
    '''
    This object acts as a pool of a set number of processors to which
    to send tasks. Each new task is in fact a newly spawned process,
    however there is a limit to the number of processes that may be
    created. The purpose of this object is to maintain that limit.

    All access to the multiprocessing module should go through this
    function, as it implement thread safety procedures that are lacking
    in multiprocessing. If you want to create a single process, simply
    create a 1-process pool and add your task to the queue.
    
    Along with the above, and somewhat in explanation, this object
    should NEVER BE SHARED BY MULTIPLE PROCESSES. EVER. Just don't
    do it. It is designed to be shared by multiple threads.
    '''
    
    # It's really a pain, but multipocessing is NOT thread
    # safe (kind of stupid, right?) so I need to lock any
    # time anything with multiprocessing is done.
    __GLOBAL_LOCK = Lock()
    
    def __init__(self, np):
        # The name of the object (the only public attribute)
        self.name = "Plot_Queuer"

        # The total number of processes that may be spawned
        # at any given time.
        self.__total = np
        
        # A lock for this object
        self.__lock = Lock()

        # Dictionary of all tasks ever created
        self.__proc_dict = {}

        # List of ids of tasks waiting to run
        self.__queue = []
        
        # List of ids of tasks currently running
        self.__running = []
        
        # List of ids of tasks that have already run
        self.__done = []

        # The most recent task id
        self.__n = 0

        # Signals for the monitor
        self.__ending = False
        self.__closing = False
        self.__open = False

        return
    
    def changeNum(self, np):
        '''
        API to change the number of processes allowed to spawn.
        '''
        self.__total = np
        return
    
    def getQueue(self):
        return self.__queue

    def getRunning(self):
        return self.__running
        
    def getDone(self):
        return self.__done

    def isOpen(self):
        return self.__open
    
    def getNumProcs(self):
        return self.__total
    
    def addToQueue(self, target, *args, **kwargs):
        '''
        API to add a task to the queue. Arguments are the arguments for process 
        spawning. The monitor MUST be running, or this method will except.
        '''
        fmt = "'%s' - Sorry, we're %s. No tasks are being accepted."
        if not self.__open:
            raise QueueError(fmt % ("not open yet", self.name))

        if self.__ending or self.__closing:
            raise QueueError(fmt % ("closing", self.name))
        
        # Create the process object. These locks may not be strictly necessary,
        # however, as a precaution I am locking anytime I interract with the
        # multiprocessing module, because it is not thread safe.
        self.__GLOBAL_LOCK.acquire()
        try:
            p = Process(name = str(self.__n + 1), target = target, args = args, 
                        kwargs = kwargs)
        finally:
            self.__GLOBAL_LOCK.release()
        
        # Locking here because different threads could be attempting to modify __n, and
        # I need the proc ids to be unique.
        self.__lock.acquire()
        try:
            self.__n += 1
            qid = self.__n
            self.__proc_dict[qid] = p
            self.__queue.append(qid)
        finally:
            self.__lock.release()

        return qid
        
    def startMonitor(self):
        '''
        API to spawn the threads that monitor and handle the queue
        '''
        if self.__ending or self.__closing:
            print "WARNING: '%s' has already been openned and closed." % self.name
            print "Are you sure you want to open again?"
            self.__ending = False
            self.__closing = False
        
        # Create the thread objects
        self.__queue_thread = Thread(target = self.__queueMonitor, 
                                     name = self.name + "_queue")
        self.__running_thread = Thread(target = self.__runningMonitor, 
                                       name = self.name + "_running")
        self.__queue_thread.daemon = True
        self.__running_thread.daemon = True

        # Start the threads
        self.__queue_thread.start()
        self.__running_thread.start()
        
        # We are now open!
        self.__open = True
        return

    def __runningMonitor(self):
        '''
        Internal API to monitor the processes that are running. This
        is run as a thread (see 'startMonitors').
        '''
        # Define a method for checking '__running' (list of tasks currently running)
        def checkRunning():
            for qid in self.__running[:]:
                proc = self.__proc_dict[qid]
                
                # Check if process is done and join if it is. 
                self.__GLOBAL_LOCK.acquire()
                try:
                    proc_is_dead = False
                    if not proc.is_alive():
                        proc.join()
                        proc_is_dead = True
                finally:
                    self.__GLOBAL_LOCK.release()
                
                if proc_is_dead:
                    # Update the lists
                    self.__running.remove(qid)
                    self.__done.append(qid)
            
            sleep(1)
            return
        
        # While we're open, check __running no matter what, when we're done or
        # we except, still keep checking.
        try:
            while not self.__closing and not self.__ending:
                checkRunning()
        finally:
            while len(self.__running):
                checkRunning()
        
    def __queueMonitor(self):
        '''
        Internal API to monitor the number of processes and handle queing. This is run
        as a thread (see 'startMonitors').
        '''
        # Define a method for checking '__queue' (list of tasks waiting to run)
        def checkQueue():
            if len(self.__running) < self.__total:
                for qid in self.__queue[:]:
                    # Start the process
                    self.__GLOBAL_LOCK.acquire()
                    try:
                        self.__proc_dict[qid].start()
                    finally:
                        self.__GLOBAL_LOCK.release()
                    
                    # Update the lists
                    self.__queue.remove(qid)
                    self.__running.append(qid)
                    
                    # Check if we're at (or over) our limit.
                    if len(self.__running) >= self.__total:
                        return
            sleep(1)
            return
        
        # Until the end or closing value is set, keep monitoring
        while not self.__ending and not self.__closing:
            checkQueue()
        else:
            if self.__closing:
                print "We're closing! Waiting for %d task(s)." % len(self.__queue)
                while len(self.__queue):
                    checkQueue()
        return
    
    def waitForAll(self, timeout = None):
        '''
        API to block until the processes in the queue run and complete. A timeout may
        be specified in seconds; default is None.
        '''
        self.__closing = True
        self.__queue_thread.join(timeout)
        self.__running_thread.join(timeout)
        self.__open = False
        return len(self.__queue)
    
    def waitForRunning(self, timeout = None):
        '''
        API to block until the processes currently running complete. A timeout may be
        specified in seconds; default is None.
        '''
        self.__ending = True
        self.__queue_thread.join(timeout)
        self.__running_thread.join(timeout)
        self.__open = False
        return len(self.__running)
    
    def waitFor(self, qid, timeout = None):
        '''
        API to block until the specified pid is completed. Timeout may be
        specified in seconds if desired. If timeout is exceded, this function
        will raise.
        '''
        class TimeoutError(Exception):
            pass
        
        def wait(t):
            sleep(1)
            t += 1
            if timeout != None:
                if t > timeout:
                    msg = "Timeout exceded while waiting for process '%s'." % self.__proc_dict[qid].name
                    raise TimeoutError(msg)
            return t
        
        t = 0
        # Wait for the process to start
        while qid in self.__queue:
            if self.__ending:
                print "Monitor is ending, process '%s' will not be run." % self.__proc_dict[qid].name
                return
            t = wait(t)

        # Wait for the process to complete
        while qid in self.__running:
            t = wait(t)
        
        return

#================================================================================
# Overlord Object:
# The Run object is the master of all. This is what users will directly interact
# with.
#================================================================================
Dataset_List = '''
Available Datasets:

FROM THE SIMULATION:
pp          - Plasma pressure [pPa]
rr          - Plasma number density [cm**-3]
V           - Plasma velocity [km/s]
B           - Magnetic field [nT]
XJ          - Current density [mu-A/m**2]
pot         - Potential [V]
pacurr      - Field Aligned Current (FAC) density [micro-A/m**2] (positive down)
sigh        - Hall conductance [S]
sigp        - Pedersen conductance [S]
prec_e_fe_1 - Diffuse auroral e- precipitation energy flux [W/m**2].
prec_e_fe_2 - Discrete auroral e- precipication energy flux [W/m**2].
prec_e_e0_1 - Diffuse auroral e- precipitation mean energy [eV].
prec_e_e0_2 - Discrete auroral e- precipitation mean energy [eV].
delphi      - Knight potential [V]
ppio        - mapped pressure [pPa]
rrio        - mapped density [cm**-3]
ttio        - mapped temperature [K]
fac_dyn     - dynamo FAC from CTIM [micro-A/m**2]
fac_tot     - total FAC [micro-A/m**2]
xjh         - Joule heating rate [W/m**2]
delbt       - Ground magnetic H perturbation [nT]
EIO         - azimuthal electric field [mV/m]

CALCULATED:
vms         - magnetosonic speed
beta        - Plasma beta
temp        - plasma temperature
cs          - sound speed
vd          - ion-electron drift speed
va          - alfven speed
ent         - specific entropy
JCB         - magnetic force density
jpar        - parallel current density
JFL         - parallel velocity
E           - electric field
aurora      - A (dubious) measure of the number of light emissions in the ionosphere
'''

class Run(object):
    '''
    This is a class to use if you are running the simulation. This is the top-most
    interface to the user.
    '''
    __doc__ = __doc__ + Dataset_List
    def __init__(self, n_plot_procs, *sim_opts, **data_set_times):
        # Initialize simulation(s)
        sim_dict = {}
        if isinstance(sim_opts, tuple):
            for opts in sim_opts:
                new_sim = Openggcm(**opts)
                sim_dict[new_sim.name] = new_sim
        else:
            new_sim = Openggcm(**sim_opts)
            sim_dict[new_sim.name] = new_sim
        
        self.sim_dict = sim_dict
        self.np = n_plot_procs
        
        out_type_dict = {'iof':Iono, '3d':Mag3d}
        
        # Process datasets
        self.ds_dict = {} # This will hold this object's references to the datasets
        ftype_ds_ref = {} # This will sort the refs for the regions to the datasets
        for ds, f_out in data_set_times.iteritems():
            ds_class = DS_INIT_DICT[ds]
            ftype = ds_class.file_type
            ot = out_type_dict[ftype]
            if ot.f_out != None and ot.f_out > f_out:
                ot.f_out = f_out
            if not self.ds_dict.has_key(ds):
                self.ds_dict[ds] = ds_class(ds)
            
            # Add this to the dictionary for the regions.
            if not ftype_ds_ref.has_key(ftype):
                ftype_ds_ref[ftype] = {}
            ftype_ds_ref[ftype][ds] = self.ds_dict[ds]
            
            print 'Initialized', ds
        
        # Collect all the useful simulation info. This is used in the output type
        # objects. Note that you can use this even if there is no simulation running.
        # That's really the point of this, is that you can make plots from a ouptuts of
        # a simulation that has already run. 
        sim_info_dict = {}
        for name, sim in self.sim_dict.iteritems():
            sim_info_dict[name] = (sim.target, sim.start_time)
        
        # Initialize out-type objects
        out_obj_dict = {}
        for key, out_type in out_type_dict.iteritems():
            if out_type.f_out < 600000:
                out_obj_dict[key] = out_type(sim_info_dict)
        
        # Give the datasets references to their slave output object.
        for ds, ds_obj in self.ds_dict.iteritems():
            ds_obj.setOutObj(out_obj_dict[ds_obj.file_type])
        
        # Give each simulation references to the slave ouptut objects.
        # Also, check for the smallest dimensions that contain both simulations.
        smallest_sim_dims = None
        for sim in self.sim_dict.itervalues():
            if smallest_sim_dims == None:
                smallest_sim_dims = sim.parameters['dims']
            else:
                other_sim_dims = sim.parameters['dims']
                for i in range(len(smallest_sim_dims)):
                    if other_sim_dims[i] < smallest_sim_dims[i]:
                        smallest_sim_dims[i] = other_sim_dims[i]
            
            sim.addOutObj(**out_obj_dict)
        
        # Initialize the plot queuer. This is the object that controls the
        # multiprocessing of the plotting. See Region object for its use.
        self.plot_queue = Plot_Queuer(self.np)
        
        # Initialize the regions
        self.regions = {}
        reg_init_dict = {'earth':Earth, 'space':Magnetosphere}
        for name, region in reg_init_dict.iteritems():
            # Only take references to the necessary regions.
            if not ftype_ds_ref.has_key(region.file_type):
                continue
            this_ds_dict = ftype_ds_ref[region.file_type]
            if name == 'space':
                default_view = tuple(['y', 0] + smallest_sim_dims)
            else:
                default_view = None
            self.regions[name] = region(name, this_ds_dict, self.plot_queue, 
                                        default_view)
        
        return
    
    def start(self):
        '''
        Function called to make initializations for running the simulation
        '''
        # Start the plot queuer
        self.plot_queue.startMonitor()
        
        # Start the simulation(s)
        for sim in self.sim_dict.itervalues():
            sim.start()
        
        return
        
    def finish(self):
        "Function to wait for things to finish"
        for sim in self.sim_dict.itervalues():
            sim.join()
        
        n_remaining = len(self.plot_queue.getQueue())
        print "Wrapping up"
        if n_remaining > 6:
            print "\nIncreasing number of plot processes\n"
            self.plot_queue.changeNum(6)
        return self.plot_queue.waitForAll()
    
    def getDatasetLabels(self, *datasets):
        "Function to get the description associated with each dataset"
        labels = dict.fromkeys(datasets)
        for ds in datasets:
            labels[ds] = self.ds_dict[ds].label
        return labels
    
    def isTimeAvailable(self, time):
        '''
        Method to determine if outputs are available for a given time.
        '''
        ret = True
        for region in self.regions.itervalues():
            if not region.isTimeAvailable(time):
                ret = False
                break
        return ret
