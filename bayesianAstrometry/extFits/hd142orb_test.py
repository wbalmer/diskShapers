'''
A script to fit a posterior distribution of orbits to
measurements of HD 142527B using 'orbitize!'
orbitize documentation:
- https://ui.adsabs.harvard.edu/abs/2020AJ....159...89B/abstract
- https://orbitize.readthedocs.io/en/latest/
- https://github.com/sblunt/orbitize

inputs are:
total_orbits (int or float) - how many orbits to fit
burn_steps (int or float) - number of orbits to fit before total orbits (discarded)
thin - how many orbits to save to disk (thin 5 will save every 5th orbit)
'''

hip_burn_steps = 10
hip_mcmc_steps = 10

total_orbits = int(10)
burn_steps = int(10)
thin = 1

if __name__=="__main__":
    # input file
    filename = "hd142-allast-orbitizelike.csv"

    # orbitize imports
    import orbitize
    from orbitize import read_input, system, priors, sampler, driver, DATADIR, hipparcos, gaia
    from orbitize.hipparcos import nielsen_iad_refitting_test
    # general imports
    import numpy as np
    import multiprocessing as mp
    import matplotlib.pyplot as plt
    import sys
    import warnings
    from datetime import datetime
    from astropy.time import Time
    import seaborn as sns

    plt.rcParams['font.family'] = 'monospace'   # Fonts
    plt.rcParams['font.monospace'] = 'DejaVu Sans Mono'
    sns.set_context("talk")

    # ignore negative in log warning bc it happens a lot and fills up terminal:
    warnings.filterwarnings("ignore")

    ### HIP reformatting ###

    # The Hipparcos ID of your target. Available on Simbad.
    hip_num = '078092'

    # Name/path for the plot this function will make
    saveplot = 'HD142527_IADrefit.png'

    # Location of the Hipparcos IAD file.
    IAD_file = '{}H{}.d'.format('./', hip_num)

    start = datetime.now()

    # run the fit
    print('Go get a coffee. This will take a few mins! :)')
    nielsen_iad_refitting_test(
        IAD_file,
        hip_num=hip_num,
        saveplot=saveplot,
        burn_steps=hip_burn_steps,
        mcmc_steps=hip_mcmc_steps
    )

    end = datetime.now()
    duration_mins = (end - start).total_seconds() / 60

    print("Done! This fit took {:.1f} mins on my machine.".format(duration_mins))

    ### ACTUAL ORBITIZE ###

    data_table = read_input.read_file(filename)
    print(data_table)

    # system parameters
    num_secondary_bodies = 1
    hipparcos_number = hip_num
    hipparcos_filename = IAD_file

    # HIP logprob
    HD142527_Hip = hipparcos.HipparcosLogProb(
        hipparcos_filename, hipparcos_number, num_secondary_bodies
    )
    # gaia logprob
    HD142527_edr3_number = 5994826707951507200
    HD142527_Gaia = gaia.GaiaLogProb(
        HD142527_edr3_number, HD142527_Hip, dr='edr3'
    )

    # more system parameters
    m0 = 2.05 # [Msol]
    plx = 6.35606723729484 # [mas]
    fit_secondary_mass = True

    mass_err = 0.5 # [Msol]
    plx_err = 0.04714455423 # [mas]

    HD142527_system = system.System(
        num_secondary_bodies, data_table, m0, plx, hipparcos_IAD=HD142527_Hip,
        gaia=HD142527_Gaia, fit_secondary_mass=fit_secondary_mass, mass_err=mass_err,
        plx_err=plx_err
    )

    # set uniform primary mass prior
    m0_index = HD142527_system.param_idx['m0']
    HD142527_system.sys_priors[m0_index] = priors.GaussianPrior(2.05, 0.3)
    # set uniform primary mass prior
    m1_index = HD142527_system.param_idx['m1']
    HD142527_system.sys_priors[m1_index] = priors.GaussianPrior(0.25, 0.2)

    # MCMC parameters
    num_temps = 20
    num_walkers = 50
    num_threads = int(mp.cpu_count()*2/3)

    # init driver
    HD142527_sampler = sampler.MCMC(
        HD142527_system, num_threads=num_threads, num_temps=num_temps,
        num_walkers=num_walkers
    )
    print('Driver Initialized! Running MCMC...')

    HD142527_sampler.run_sampler(total_orbits, burn_steps=burn_steps, thin=thin)

    myResults = HD142527_sampler.results

    # save posterior to disk
    savefile = 'paperdraft_posterior.hdf5'
    myResults.save_results(savefile)
    print('Saved Posterior!')

    starttime = Time(datetime.strptime('1990 January 1', '%Y %B %d')).to_value('mjd', 'long')

    orb = myResults.plot_orbits(
                                object_to_plot = 1, # Plot orbits for the first (and only, in this case) companion
                                num_orbits_to_plot= 1, # Will plot 50 randomly selected orbits of this companion
                                start_mjd=starttime, # Minimum MJD for colorbar (here we choose first data epoch)
                                show_colorbar = True,
                                rv_time_series = True,
                                plot_astrometry_insts=True
    )
    orb.savefig('HD72946_rvtimeseries_panelplot.pdf')
    print('Save orbit plot!')

    # posterior plot
    median_values = np.median(myResults.post,axis=0) # Compute median of each parameter
    range_values = np.ones_like(median_values)*0.95 # Plot only 95% range for each parameter
    corner_figure_median_95 = myResults.plot_corner(
        range=range_values,
        truths=median_values
    )
    corner_figure_median_95.savefig('hd142_postplot.png')
    print('Saved basic corner plot!')

    # seaborn posterior plot
    params = ['a$_{1}$ [au]','e$_{1}$','i$_{1}$ [$^\\circ$]', '$\\omega_{0}$ [$^\\circ$]', '$\\Omega_{0}$ [$^\\circ$]','$\\tau_{1}$','$\\pi$ [mas]', '$\\mu_\\alpha$', '$\\mu_\\delta$', '$\\alpha_{0}$', '$\\delta_{0}$','M$_B$ [M$_{{\\odot}}$]','M$_A$ [M$_{{\\odot}}$]']
    import pandas as pd
    postframe = pd.DataFrame(myResults.post, columns=params)
    g = sns.PairGrid(postframe,
                 vars = params,
                 diag_sharey=False, corner=True,
                 )
    g.map_lower(sns.kdeplot, levels=5, lw=1, color='black')
    g.map_diag(sns.kdeplot, lw=3, color="black")
    g.add_legend()
    plt.savefig('hd142_seaborn_corner_full.pdf',dpi=300)

    g2 = sns.PairGrid(postframe,
                 vars = ['a$_{1}$ [au]','e$_{1}$','i$_{1}$ [rad]','M$_A$ [M$_{{\\odot}}$]','M$_B$ [M$_{{\\odot}}$]'],
                 diag_sharey=False, corner=True,
                 )
    g2.map_lower(sns.kdeplot, levels=5, lw=1, color='black')
    g2.map_diag(sns.kdeplot, lw=3, color="black")
    g2.add_legend()
    plt.savefig('hd142_seaborn_corner_small.pdf',dpi=300)
    print('Script done :)')
