from ggcm_simulation import Run
from datetime import datetime, timedelta

# I need to set my starting and ending times. These MUST be datetime objects.
# In this case, I am setting my end time to be the start of whatever hour this file is
# run, and the start time is one hour before that.
end = datetime.utcnow().replace(second = 0, microsecond = 0, minute = 0)
start = end - timedelta(hours = 1)

# Here is where I set my simulation options. There are a lot of possible options (see
# `simulation.py' for more details), but I am showing a very basic selection. 
# `start_time' and `end_time' are required, and `emulate' tells the code to simply copy
# hdf5 files from a local directory into a target directory, which saves time and 
# computation while still testing most aspects of the code. The default target directory
# is ~/run/run<YR><MO><DA><HR>_<n>/target, where this run is the n-th run in an hour.
sim_opts = {}
sim_opts.update(start_time = start, 
                end_time   = end, 
                emulate    = True)

# Here, I specify WHAT I will plot. I am only doing plots in the magnetosphere for this
# example. What this means is that `pp' (plasma pressure) should be plotted every 
# 5*60 (simulated) seconds = 5 (simulated) minutes and `temp' (plasma temperature)
# should plot every 2.5*60 (simulated) seconds = 2.5 (simulated) minutes.
datasets = {'pp':5*60, 'temp':2.5*60}

# Now I simply initialze the simulation. The arguments indicate the number of processes
# (4), and then the simulation options `sim_opts' and the datasets given as keyword 
# arguments.
# This is a very useful shorthand for python:
#   func(**{'a':1, 'b':2}) = func(a = 1, b = 2)
# Similarly,
#   func(*[1,2]) = func(*(1,2)) = func(1,2)
# In this case, it would have been just as easy to write
#   run = Run(4, sim_opts, pp = 5*60, temp = 2.5*60),
# but in general you will want to assign the datasets to be plotted more systematically
# so it's worth demonstrating.
run = Run(4, sim_opts, **datasets)
run.start()

# Now I simply loop through the output times and make the plots as they become 
# available. Note that the simulation framework doesn't actually make the plots as it
# goes, it simply makes them available. You still have to tell it to make the plots.
# This is because (as you can see in `update.py') you often need to do other things 
# around the plotting that are not handled by the simulation.
DT = timedelta(seconds = 2.5*60)
T = start + DT
while T < end:
    # Check to see if the time is available to be plotted. This will check to see that
    # EVERY dataset requested is available. If you wanted, there is a way to go dataset
    # by dataset and check individually.
    if run.isTimeAvailable(T):
        # Go through all the datasets and plot them for the given time.
        for ds in datasets.iterkeys():
            # The `regions' attribute is set by default based on what datasets are
            # given. Every dataset is associated with one of two regions: the 
            # magnetosphere (far from earth) or the ionosphere (close to earth). In
            # this case, because we only selected datasets from the magnetosphere 
            # region we simply select whichever region is first.
            run.regions.values()[0].plot(T, ds)
        
        # Now step in time.
        T = T + DT

# Now we tell the simulation to wrap up and wait for it to finish. It should be done or
# at least nearly done (unless we messed up on our time stepping in the loop). This
# also allow any plots that are still running to wrap up before we close everything.
run.finish()

# And that's it!
