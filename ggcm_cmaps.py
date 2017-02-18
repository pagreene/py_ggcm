from matplotlib.cm import *
ggcm_cdicts = {}
ggcm_cdicts["Blue_faid"] = {'blue' : ((0.0, 0.0, 0.0),
                                      (0.25, 0.6, 0.6),
                                      (0.5, 0.8, 0.8),
                                      (1.0, 1.0, 1.0)),
                            'green': ((0.0, 0.0, 0.0),
                                      (0.25, 0.2, 0.2),
                                      (1.0, 1.0, 1.0)),
                            'red'  : ((0.0, 0.0, 0.0),
                                      (0.75, 0.1, 0.1),
                                      (1.0, 1.0, 1.0))}
ggcm_cdicts["Green_faid"] = {'green' : ((0.0, 0.0, 0.0),
                                       (0.75, 0.8, 0.8),
                                       (1.0, 1.0, 1.0)),
                             'blue': ((0.0, 0.0, 0.0),
                                       (0.75, 0.0, 0.0),
                                       (1.0, 1.0, 1.0)),
                             'red'  : ((0.0, 0.0, 0.0),
                                       (0.75, 0.0, 0.0),
                                       (1.0, 1.0, 1.0))}
ggcm_cdicts["Red_faid"] = {'red' : ((0.0, 0.0, 0.0),
                                     (0.5, 0.8, 0.8),
                                     (1.0, 1.0, 1.0)),
                           'green': ((0.0, 0.0, 0.0),
                                     (0.75, 0.0, 0.0),
                                     (1.0, 1.0, 1.0)),
                           'blue'  : ((0.0, 0.0, 0.0),
                                     (0.75, 0.0, 0.0),
                                     (1.0, 1.0, 1.0))}
ggcm_cdicts["Purple_faid"] = {'blue' : ((0.0, 0.0, 0.0),
                                        (0.5, 0.8, 0.8),
                                        (1.0, 1.0, 1.0)),
                              'green': ((0.0, 0.0, 0.0),
                                        (0.75, 0.0, 0.0),
                                        (1.0, 1.0, 1.0)),
                              'red'  : ((0.0, 0.0, 0.0),
                                        (0.75, 0.6, 0.6),
                                        (1.0, 1.0, 1.0))}

ggcm_cdicts["Aurora1"] = {'blue': ((0.0, 0.0, 0.0),
                                   (0.4, 0.5, 0.5),
                                   (0.6, 0.5, 0.5),
                                   (1.0, 0.0, 0.0)),
                         'green': ((0.0, 1.0, 1.0),
                                   (1.0, 0.0, 0.0)),
                         'red'  : ((0.0, 0.0, 0.0),
                                   (1.0, 1.0, 1.0))}

ggcm_cdicts["Aurora2"] = {'blue' : ((0.0, 0.2, 0.2),
                                   (0.4, 0.7, 0.7),
                                   (0.6, 0.7, 0.7),
                                   (1.0, 0.2, 0.2)),
                         'green': ((0.0, 1.0, 1.0),
                                   (1.0, 0.2, 0.2)),
                         'red'  : ((0.0, 0.2, 0.2),
                                   (1.0, 1.0, 1.0))}

for color, cdict in ggcm_cdicts.iteritems():
    register_cmap(name = color, data = cdict, lut  = 1000)
    
import pylab as pl
from My_Re import find
def viewChoices(choice_list = []):
    '''
    A function to plot all of the colormaps. Intended for use in iPython.
    '''
    pl.rc('text', usetex=False)
    maps = []
    if not len(choice_list):
        for m in datad:
            if not find('_r', m):
                maps.append(m)
        maps += ggcm_cdicts.keys()
    else:
        for m in choice_list:
            maps.append(m)
    maps.sort()
    l = len(maps)
    pl.figure(figsize = (l/2, 5))
    a = np.outer(np.arange(0,1,0.001), np.ones(10))
    pl.subplots_adjust(top=0.8,bottom=0.05,left=0.01,right=0.99)
    for i, m in enumerate(maps):
        pl.subplot(1,l,i)
        pl.axis('off')
        pl.imshow(a, aspect='auto', cmap = get_cmap(m), origin='lower')
        pl.title(m, rotation=90)
        
    pl.show()

if __name__ == "__main__":
    from sys import argv
    maps = []
    for i, arg in enumerate(argv):
        if i > 0:
            maps.append(arg)
    viewChoices(maps)
