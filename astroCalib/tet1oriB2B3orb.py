'''
A script to fit a posterior distribution of orbits to
measurements of theta1ori b2b3 using 'orbitize!'
orbitize documentation:
- https://ui.adsabs.harvard.edu/abs/2020AJ....159...89B/abstract
- https://orbitize.readthedocs.io/en/latest/
- https://github.com/sblunt/orbitize

inputs are:
total_orbits (int or float) - how many orbits to fit
burn_steps (int or float) - number of orbits to fit before total orbits (discarded)
thin - how many orbits to save to disk (thin 5 will save every 5th orbit)
'''

total_orbits = 2#e5
burn_steps = 1#e5
thin = 1#0

if __name__=="__main__":
    # input file
    filename = "all-B2-B3.csv"

    # orbitize imports
    import orbitize
    from orbitize import read_input, system, priors, sampler, driver, DATADIR
    # general imports
    import numpy as np
    import multiprocessing as mp
    import matplotlib.pyplot as plt
    import sys
    import warnings

    # ignore negative in log warning bc it happens a lot and fills up terminal:
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'RuntimeWarning: invalid value encountered in log')


    # system parameters
    num_secondary_bodies = 1
    system_mass = 5.5 # [Msol]
    plx = 2.654763675820139 # [mas]
    mass_err = 0.5 # [Msol]
    plx_err = 0.042361263 # [mas]

    # MCMC parameters
    num_temps = 10
    num_walkers = 50
    num_threads = int(mp.cpu_count()*2/3)

    # init the driver class that will run orbitize
    my_driver = driver.Driver(
        filename, 'MCMC', num_secondary_bodies, system_mass, plx, mass_err=mass_err, plx_err=plx_err,
        system_kwargs = {'fit_secondary_mass':False, 'tau_ref_epoch':0},
        mcmc_kwargs={'num_temps': num_temps, 'num_walkers': num_walkers, 'num_threads': num_threads}
                              )
    print('Driver Initialized...')
    # assign the sampler
    m = my_driver.sampler

    # getting the system object:
    sys = my_driver.system
    # get parameter labels
    lab = sys.param_idx

    # set priors on parameters
    # sys.sys_priors[lab['m0']] = priors.GaussianPrior(0.986, 0.027) # Maire 2020
    # sys.sys_priors[lab['m1']] = priors.GaussianPrior(0.067,0.1)
    # sys.sys_priors[lab['m2']] = priors.GaussianPrior(1.1, 0.1) # Anderson & Francis 2012
    # sys.sys_priors[lab['sma1']] = priors.GaussianPrior(6.5,1)
    # sys.sys_priors[lab['inc1']] = priors.GaussianPrior(np.deg2rad(56),np.deg2rad(5))
    # sys.sys_priors[lab['pan1']] = priors.GaussianPrior(np.deg2rad(-12),np.deg2rad(20))
    # sys.sys_priors[lab['pan1']] = priors.GaussianPrior(np.deg2rad(252),np.deg2rad(20))
    # sys.sys_priors[lab['ecc1']] = priors.GaussianPrior(0.5,0.2)
    print('Priors set...')
    # make sure inputs are int
    total_orbits = int(total_orbits)*num_walkers # number of steps x number of walkers (at lowest temperature)
    burn_steps = int(burn_steps) # steps to burn in per walker
    thin = int(thin) # only save every 2 steps

    # run the MCMC
    print('Running MCMC...')
    m.run_sampler(total_orbits, burn_steps=burn_steps, thin=thin)

    # assign results
    myResults = m.results
    # save posterior to disk
    savefile = 'b2b3_orbitize_posterior.hdf5'
    myResults.save_results(savefile)
    print('Saved Posterior!')
    from datetime import datetime
    from astropy.time import Time

    # myResults.load_results('wbalmer_orbitize_posterior.hdf5', append=True)

    import seaborn as sns

    plt.rcParams['font.family'] = 'monospace'   # Fonts
    plt.rcParams['font.monospace'] = 'DejaVu Sans Mono'
    sns.set_context("talk")
    plt.rcParams.update({
    "text.usetex": True,
    })

    starttime = Time(datetime.strptime('1990 January 1', '%Y %B %d')).to_value('mjd', 'long')
    # orb = myResults.plot_orbits(
    #                             object_to_plot = 1, # Plot orbits for the first (and only, in this case) companion
    #                             num_orbits_to_plot= 1, # Will plot 50 randomly selected orbits of this companion
    #                             start_mjd=starttime, # Minimum MJD for colorbar (here we choose first data epoch)
    #                             show_colorbar = True,
    #                             rv_time_series = False,
    #                             plot_astrometry_insts=True
    # )
    # orb.savefig('b2b3_panelplot.pdf')
    # print('Save orbit plot!')
    # posterior plot
    median_values = np.median(myResults.post,axis=0) # Compute median of each parameter
    range_values = np.ones_like(median_values)*0.95 # Plot only 95% range for each parameter
    corner_figure_median_95 = myResults.plot_corner(
        range=range_values,
        truths=median_values
    )
    corner_figure_median_95.savefig('b2b3_postplot.pdf')
    print('Saved basic corner plot!')

    # seaborn posterior plot
    params = ['a$_{1}$ [au]','e$_{1}$','i$_{1}$ [$^\\circ$]', '$\\omega_{0}$ [$^\\circ$]', '$\\Omega_{0}$ [$^\\circ$]','$\\tau_{1}$','$\pi$ [mas]','M$_T$ [M$_{{\\odot}}$]']
    import pandas as pd
    postframe = pd.DataFrame(myResults.post, columns=params)
    g = sns.PairGrid(postframe,
                 vars = params,
                 diag_sharey=False, corner=True,
                 )
    g.map_lower(sns.kdeplot, levels=5, lw=1, color='black')
    g.map_diag(sns.kdeplot, lw=3, color="black")
    g.add_legend()
    plt.savefig('b2b3_seaborn_corner.pdf',dpi=300)
    print('Script done :)')
