import orbitize
from orbitize import driver
import matplotlib.pyplot as plt
import multiprocessing as mp
import warnings
warnings.filterwarnings("ignore")

# print python version
from platform import python_version

print('printing python version:', python_version())

# MCMC parameters
num_walkers = 50
num_threads = mp.cpu_count() # or a different number if you prefer

my_driver = orbitize.driver.Driver('{}/B2-B3-orbitizetab.csv'.format(orbitize.DATADIR), # path to data file
                                  'MCMC', # name of algorithm for orbit-fitting
                                  1, # number of secondary bodies in system
                                  5.5, # total system mass [M_sun]
                                  2.41, # total parallax of system [mas]
                                  mass_err=0.5, # mass error [M_sun]
                                  plx_err=0.03, # parallax error [mas]
                                  mcmc_kwargs={'num_walkers': num_walkers, 'num_threads': num_threads})

# RUN MCMC
total_orbits = 100 # number of steps x number of walkers (at lowest temperature)
burn_steps = 10 # steps to burn in per walker
thin = 1 # only save every 2nd step

my_driver.sampler.run_sampler(total_orbits, burn_steps=burn_steps, thin=thin)
