import numpy as np
import os
if 'DISPLAY' not in os.environ:
    # This is necessary if there is no access to a screen (X-windows)
    from matplotlib import use
    use("agg", warn = False)
from matplotlib.image import imread
from mpl_toolkits.basemap import *
from mpl_toolkits.basemap import _cylproj
from urllib import urlretrieve
from scipy.misc import imresize, imsave
    
class Img_Array(np.ndarray):
    '''
    This is an object for use representing images as numpy arrays. It is often useful
    to know what file the image came from, so this object stores that information.
    It also provides a method for resizing (using scipy.misc.imresize), which differs
    significantly from normal array resizing.
    
    Inheriting from ndarrays is a bit tricky because of the many ways that
    an array can be instantiated. For a more detailed explanation:
    http://docs.scipy.org/doc/numpy/user/basics.subclassing.html#basics-subclassing
    '''
    def __new__(self, arg):
        '''
        This object is instantiated with the name of an image file. That file is read
        and the name of the file saved. __new__ is used instead of __init__ because
        ndarray cannot be initialized with an existing array.
        '''
        if isinstance(arg, str):
            obj = np.asarray(imread(arg)).view(self)
            obj.fname = arg
        elif isinstance(arg, np.ndarray):
            obj = np.asarray(arg).view(self)
            obj.fname = None
        return obj
    
    def __array_finalize__(self, obj):
        '''
        This would be part of the quirks of inheriting from np.ndarray
        '''
        if obj is None:
            return
        
        self.fname = getattr(obj, 'info', None)
        return
        
    def resize(self, shape, interp='bilinear', mode=None):
        '''
        This is an API to resize the array as an image. 
        
        Parameters
        ----------
        shape: int, float, or tuple
            * int   - Percent of current size
            * float - Fraction of current size
            * tuple - New dimensions of image
        
        interp: string
            interpolation to use for re-sizing ('nearest', 'bilinear', 'bicubic' or 'cubic')
        
        mode:
            mode is the PIL image mode ('P', 'L', etc.)
        
        Returns
        -------
        New Img_Array object of the specified size
        '''
        new_array = Img_Array(imresize(self, shape, interp, mode))
        new_array.fname = self.fname
        return new_array
    
    def save(self, new_name = None):
        '''
        API to save an image array as an image using scipy.misc.imsave
        '''
        if new_name == None:
            return imsave(self.fname, self)
        else:
            return imsave(new_name, self)
        
    def flipVertical(self):
        '''
        API to flip an image vertically (inplace)
        '''
        H, W = self.shape[:-1]
        
        tmp_self = np.ones((H+1, W, 3))
        tmp_self[1:,:,:] = self[:,:,:]
        
        for i in range(W):
            self[:,i,:] = tmp_self[-1:0:-1,i,:]
        
        return
    
    def flipHorizontal(self):
        '''
        API to flip an image horizontally (inplace)
        '''
        H, W = self.shape[:-1]
        
        tmp_self = np.ones((H, W+1, 3))
        tmp_self[:,1:,:] = self[:,:,:]
        
        for i in range(H):
            self[i,:,:] = tmp_self[i,-1:0:-1,:]
        
        return
    
class ggcm_Basemap(Basemap):
    def warpimage(self, image="bluemarble", scale=None, **kwargs):
        """
        Display an image (filename given by ``image`` keyword) as a map background.
        If image is a URL (starts with 'http'), it is downloaded to a temp
        file using urllib.urlretrieve.

        Default (if ``image`` not specified) is to display
        'blue marble next generation' image from http://visibleearth.nasa.gov/.

        Specified image must have pixels covering the whole globe in a regular
        lat/lon grid, starting and -180W and the South Pole.
        Works with the global images from
        http://earthobservatory.nasa.gov/Features/BlueMarble/BlueMarble_monthlies.php.

        The ``scale`` keyword can be used to downsample (rescale) the image.
        Values less than 1.0 will speed things up at the expense of image
        resolution.

        Extra keyword ``ax`` can be used to override the default axis instance.

        \**kwargs passed on to :meth:`imshow`.

        returns a matplotlib.image.AxesImage instance.
        """
        ax = kwargs.pop('ax', None) or self._check_ax()
        if isinstance(image, Img_Array):
            file = image.fname
            is_object = True
            dtype = image.dtype
        elif isinstance(image, str):
            is_object = False
            dtype = np.uint8
            
            # default image file is blue marble next generation
            # from NASA (http://visibleearth.nasa.gov).
            if image == "bluemarble":
                file = os.path.join(basemap_datadir, 'bmng.jpg')
            
            # display shaded relief image (from
            # http://www.shadedreliefdata.com)
            elif image == "shadedrelief":
                file = os.path.join(basemap_datadir, 'shadedrelief.jpg')
            
            # display etopo image (from
            # http://www.ngdc.noaa.gov/mgg/image/globalimages.html)
            elif image == "etopo":
                file = os.path.join(basemap_datadir, 'etopo1.jpg')
            else:
                file = image
        else:
            raise TypeError('Expected Img_Array or str, got %s instead' % type(image))
        
        # if image is same as previous invocation, used cached data.
        # if not, regenerate rgba data.
        if not hasattr(map,'_bm_file') or self._bm_file != file:
            newfile = True
        else:
            newfile = False
        
        if file.startswith('http'):
            self._bm_file, _ = urlretrieve(file)
        else:
            self._bm_file = file
        
        # bmproj is True if map projection region is same as
        # image region.
        bmproj = self.projection == 'cyl' and \
                 self.llcrnrlon == -180 and self.urcrnrlon == 180 and \
                 self.llcrnrlat == -90 and self.urcrnrlat == 90
        
        # read in jpeg image to rgba array of normalized floats.
        if not hasattr(map,'_bm_rgba') or newfile:
            # get the image (if necessary)
            if is_object:
                self._bm_rgba = image
            else:
                self._bm_rgba = Img_Array(self._bm_file)
            
            # scale the image
            if scale is not None:
                self._bm_rgba = self._bm_rgba.resize(scale)
            
            # define lat/lon grid that image spans.
            nlons = self._bm_rgba.shape[1]; nlats = self._bm_rgba.shape[0]
            delta = 360./float(nlons)
            self._bm_lons = np.arange(-180. + 0.5*delta,180., delta)
            self._bm_lats = np.arange(-90. + 0.5*delta,90., delta)
            
            # is it a cylindrical projection whose limits lie
            # outside the limits of the image?
            cylproj =  self.projection in _cylproj and \
                      (self.urcrnrlon > self._bm_lons[-1] or \
                       self.llcrnrlon < self._bm_lons[0])
            
            # if pil_to_array returns a 2D array, it's a grayscale image.
            # create an RGB image, with R==G==B.
            if self._bm_rgba.ndim == 2:
                tmp = np.ones(self._bm_rgba.shape+(3,),dtype)
                for k in range(3):
                    tmp[:,:,k] = self._bm_rgba
                self._bm_rgba = tmp
            
            if cylproj and not bmproj:
                # stack grids side-by-side (in longitiudinal direction), so
                # any range of longitudes may be plotted on a world self.
                self._bm_lons = \
                    np.concatenate((self._bm_lons, self._bm_lons + 360), 1)
                self._bm_rgba = \
                    np.concatenate((self._bm_rgba, self._bm_rgba), 1)
            
            # convert to normalized floats if they are not already.
            if self._bm_rgba.mean() > 1.:
                self._bm_rgba = self._bm_rgba.astype(np.float32)/255.
        
        if not bmproj: # interpolation necessary.
            if newfile or not hasattr(map,'_bm_rgba_warped'):
                # transform to nx x ny regularly spaced native
                # projection grid.
                # nx and ny chosen to have roughly the
                # same horizontal res as original image.
                if self.projection != 'cyl':
                    dx = 2.*np.pi*self.rmajor/float(nlons)
                    nx = int((self.xmax - self.xmin)/dx) + 1
                    ny = int((self.ymax - self.ymin)/dx) + 1
                else:
                    dx = 360./float(nlons)
                    nx = int((self.urcrnrlon - self.llcrnrlon)/dx) + 1
                    ny = int((self.urcrnrlat - self.llcrnrlat)/dx) + 1
                
                self._bm_rgba_warped = np.ones((ny,nx,4), np.float64)
                
                # interpolate rgba values from geographic coords (proj='cyl')
                # to map projection coords.
                # if masked=True, values outside of
                # projection limb will be masked.
                for k in range(3):
                    self._bm_rgba_warped[:,:,k], x, y = \
                        self.transform_scalar(self._bm_rgba[:,:,k], self._bm_lons, 
                                              self._bm_lats, nx, ny, returnxy = True)
                # for ortho,geos mask pixels outside projection limb.
                if self.projection in ['geos','ortho','nsper'] or \
                   (self.projection == 'aeqd' and self._fulldisk):
                    lonsr, latsr = self(x, y, inverse = True)
                    mask = ma.zeros((ny,nx,4), np.int8)
                    mask[:,:,0] = np.logical_or(lonsr>1.e20, latsr>1.e30)
                    
                    for k in range(1,4):
                        mask[:,:,k] = mask[:,:,0]
                    
                    self._bm_rgba_warped = \
                        ma.masked_array(self._bm_rgba_warped, mask = mask)
                    
                    # make points outside projection limb transparent.
                    self._bm_rgba_warped = self._bm_rgba_warped.filled(0.)
                # treat pseudo-cyl projections such as mollweide, robinson and sinusoidal.
                elif self.projection in _pseudocyl:
                    lonsr,latsr = map(x, y, inverse = True)
                    mask = ma.zeros((ny,nx,4), np.int8)
                    lon_0 = self.projparams['lon_0']
                    lonright = lon_0 + 180.
                    lonleft = lon_0 - 180.
                    x1 = np.array(ny*[0.5*(self.xmax + self.xmin)], np.float)
                    y1 = np.linspace(self.ymin, self.ymax, ny)
                    lons1, lats1 = map(x1, y1, inverse = True)
                    lats1 = np.where(lats1 < -89.999999, -89.999999, lats1)
                    lats1 = np.where(lats1 > 89.999999, 89.999999, lats1)
                    
                    for j,lat in enumerate(lats1):
                        xmax, ymax = map(lonright, lat)
                        xmin, ymin = map(lonleft, lat)
                        mask[j,:,0] = np.logical_or(x[j,:] > xmax, x[j,:] < xmin)
                        
                    for k in range(1,4):
                        mask[:,:,k] = mask[:,:,0]
                        
                    self._bm_rgba_warped = \
                        ma.masked_array(self._bm_rgba_warped, mask = mask)
                    
                    # make points outside projection limb transparent.
                    self._bm_rgba_warped = self._bm_rgba_warped.filled(0.)
            
            # plot warped rgba image.
            im = self.imshow(self._bm_rgba_warped, ax = ax, **kwargs)
        else:
            # bmproj True, no interpolation necessary.
            im = self.imshow(self._bm_rgba, ax = ax, **kwargs)
        #print '+++', self._bm_file, self._bm_rgba
        return im

