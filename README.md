#PY_GGCM
##Python Interfac to OpenGGCM

--------------------------------------------------------------------------
###Contributors 
Patrick Greeene
patrick.anton.greene@gmail.com

-------------------------------------------------------------------------------
###Prupose 
The purpose of this code is to run the openggcm simulation and convert the output data 
files into plots that are used by the spaceweather website hosted on fishercat. It is 
hoped that the system may be extendable to other purposes. Ideas that I've had for 
extensions or other improvements are contained in the TODO text file.

---------------------------------------------------------------------------------
###Usage 
To run a standard update of the website, simply run:

```
python update.py
```

For an example of a basic run in the Magnetosphere, look at/read through `sample.py`.

---------------------------------------------------------------------------------
###Files 
Here is a list of the different files and a general summary of their purpose and
contents. THIS IS BY NO MEANS COMPREHENSIVE. This should serve the function of helping
you figure out where to look to answer any detailed questions, as well as giving you a
feel for the overall structure of the code.

- `ggcm_simulation.py` is the primary beast. This file contains most of the code 
    involved in running the simulation and plotting the outputs.
    
    Contents:
    
    - *Plotting Utilities*:  Provides objects with methods to ease plotting. The Region object
            lays out a common interface, so that the user doesn't need to know about
            the details of how the plots are made. The child implement the specifics
            of the plots.
      - Parent:   `Region`
      - Children: `Earth`, `Magnetosphere`
    
    - *Output File Type Objects*: Provides low level methods of interfacing with the hdf5 files 
            output by the simulation. There are 3 types of file that are currently
            available: iof, 3df, and p<dim>_<plane>. The formats of the files differ
            according to the type of data they contain, so each output type object
            defines the methods needed to get the data from that type of file. This
            includes determining the coordinates.
      - Parent:   `_Output_Type`
      - Children: `Iono`, `Mag3d`, `Mag2d`(depricated)
      - Other:    `_HDF5` (lowest level interface) 
    
    
    - *Simulation Interface*: These are low level objects and methods used to begin and monitor
            the simulation. Openggcm is the main class. The Shell_Command object is
            used to run the simulation command in a separate process. Openggcm directly
            handles the initialization of the simulation by modifying and running the
            runme file. The `__simulation_emulator' method is also provided so that the
            behavior of the simulation may be superficially imitated for testing 
            purposes.
      - Main: `Openggcm`
      - Other: `Shell_Command`
    
    - *Solar Wind Data Retrieval*: (Master: `Openggcm`) Provides the methods needed to retrieve data from the ACE solar wind
            satellite and write it to the solar wind data file. This should really be
            included in `openggcm'.
      - Main: `swdata`
    
    - *Queueing Object*: (Master: `Region`) This object provides the methods needed to maintain a running queue
            of plots to be plotted at most `np' number of processes at a time. New
            requests are added to the queue and plotted in the order requested when 
            space opens up.
      - Main: `Plot_Queuer`
    
    - *Main*: (Overlord Object) This is the highest level object which is imported and instantiated
            by the user. All other objects are slaves to this object, and they are
            coordinated by this object. All other object instantiations occur 
            internally to this object.
      - Main: `Run`


- `ggcm_datasets.py` contains all the class definitions of the datasets. For
    datasets that come directly from the simulation outputs, the default plotting info
    is defined as class properties, while for calculated datasets (calculated using
    data from other raw datasets), these default plotting properties as well as the
    calculation method are defined. The two types of dataset correspond to two classes
    from which the individual dataset classes inherit. These two classes themselves
    inherit from one base class that standardizes the higher level interface.
    
    CONTENTS:
    
    - *Base Dataset Objects*: The Children are the classes from which all the particular dataset
            classes will inherit from. Average_Dataset is used for datasets that are 
            taken directly from the simulation(s). If there are multiple simulations,
            the outputs are averaged. Ave_Calc_Ds is for datasets that are calculated
            from other datasets. As with Average_Datasets, if there are multiple
            simulations, the outputs of the simulations are averaged.
      - Parent:   `_Base_Dataset`
      - Children: `Average_Dataset`, `Ave_Calc_Ds`
    
    - *Helpful Functions*: These are some functions to help define the calculations for
            calculated datsets.
    
    - *Dataset Definitions*: Here is where the properties of each dataset are laid out. Currently
            they are constructed by hand, which is dreadfully time consuming, and it is
            fairly systematic, so the definitions should really be automatically
            generated. Sadly, it is not systematic enough for this to be trivial.

- `ggcm_logger.py` contains the specifications for logging.

- `ggcm_cmaps.py` contains some definitions of Matplotlib colormaps used for 
    plotting.

- `my_basemap.py` contains a relatively minor yet curcial modification of the
    matplotlib basemap toolkit that allows for better pasting of images onto earth
    maps.

- `sim_defaults.py` defines some basic default parameters for the simulation.

The following files should be phased out because they are unnecessary:

- `My_Re.py` contains some wrappers around some of the standard python RegEx utilities.

- `ggcm_dir.py` contains some special objects and structure for handling
    directories. It is implemented widely throughout the code, and although it works
    very well and is very convenient, much of the same functionality could be achieved
    with standard utilities.

- `ggcm_exceptions.py` contains some special exceptions that do not require their own file.
