import os
import time
import pickle
import warnings
import numpy as np
import astropy.io.fits as fits
import pyklip.fitpsf as fitpsf
import matplotlib.pylab as plt
import pyklip.instruments.MagAO as MagAO
from forwardModel import forwardModel


warnings.filterwarnings('ignore')

if __name__ == '__main__':  # This is a very important precaution for Windows
    __spec__ = None  # Important for ipynb compatibility

    # plotting functions
    def create_circular_mask(h, w, center=None, radius=None, leq=False):

        if center is None:  # use the middle of the image
            center = (int(w/2), int(h/2))
        if radius is None:  # use the smallest distance between the center and image walls
            radius = min(center[0], center[1], w-center[0], h-center[1])

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
        if leq is False:
            return dist_from_center >= radius
        else:
            return dist_from_center <= radius


    def domask(img, inn=70, outt=180):
        outermask = create_circular_mask(451, 451, radius=outt, leq=True)
        img[~outermask] = np.nanmedian(img)
        mask = create_circular_mask(451, 451, radius=inn)
        masked_img = img.copy()
        masked_img[~mask] = np.nanmedian(img)
        return masked_img

    # loop inputs
    date = '18May15'  # str(input('enter date in format ##Month##: '))
    fwhm = 4.49  # float(input('enter dataset fwhm: '))
    ann1 = 6  # int(input('enter IWA: '))
    ann2 = 20  # int(input('enter OWA: '))
    sep = 10  # int(input('enter sep: '))
    pa = 110  # int(input('enter pa: '))

    # dataset
    # date = '16May15'
    prefixes = ['Cont', 'Ha']
    psfplans = [['doGaussian', fwhm], ['doMoffat', fwhm],
                ['doGhost', fwhm]]
    residuals = []
    filepaths = []
    outputdirs = []
    for pref in prefixes:
        filepath = f'data\\{date}\\{pref}\\*fits'
        outputdir = f'output\\{date}\\{pref}'
        filepaths.append(filepath)
        outputdirs.append(outputdir)

    for instpsf in psfplans:
        PSFpath = instpsf[0]
        FWHM = fwhm
        # parameters
        KLmode = 20
        contrast = 1e-2
        an = [ann1, ann2]
        move = 4
        scale = 1

        cores = 4
        highpass = 1.5*FWHM
        numbasis = KLmode

        # run
        for i in range(len(filepaths)):
            filepath = filepaths[i]
            outputdir = outputdirs[i]
            prefix = prefixes[i]
            if i == 0:
                fm = forwardModel(filepath, outputdir, prefix, KLmode, sep, pa, contrast, an, move, scale, ePSF=PSFpath, FWHM=FWHM, cores=cores, highpass=highpass, numbasis=numbasis)
            else:
                fm.update_paths(filepath, outputdir, prefix)
            fm.prep_KLIP()
            fm.run_KLIP()

        output_prefix = outputdir+'\\'
        print('output_prefix: '+output_prefix)

        # move image
        try:
            from shutil import move
            src = output_prefix + prefix + "-fmpsf-KLmodes-all.fits"
            dest = src.replace(prefix + "-fmpsf-KLmodes-all.fits", PSFpath+'\\'+prefix+'-fmpsf-KLmodes-all.fits')
            move(src, dest)
            src = output_prefix + prefix + "-klipped-KLmodes-all.fits"
            dest = src.replace(prefix + "-klipped-KLmodes-all.fits", PSFpath+'\\'+prefix+'-klipped-KLmodes-all.fits')
            move(src, dest)
        except FileNotFoundError:
            src = output_prefix + prefix + "-fmpsf-KLmodes-all.fits"
            os.mkdir(src.replace(prefix + "-fmpsf-KLmodes-all.fits", '\\'+PSFpath+'\\'))
            dest = src.replace(prefix + "-fmpsf-KLmodes-all.fits", PSFpath+'\\'+prefix+'-fmpsf-KLmodes-all.fits')
            os.rename(src, dest)
            src = output_prefix + prefix + "-klipped-KLmodes-all.fits"
            dest = src.replace(prefix + "-klipped-KLmodes-all.fits", PSFpath+'\\'+prefix+'-klipped-KLmodes-all.fits')
            os.rename(src, dest)

        output_prefix = os.path.join(output_prefix, PSFpath)
        output_prefix = output_prefix+'\\'+prefix


        plt.savefig(output_prefix+'-FM-psfs-.png', dpi=300)
        img = fits.getdata(output_prefix + "-fmpsf-KLmodes-all.fits")[0]
        n = np.nanmax(img)

        plt.savefig('test.png')
        plt.imshow(domask(img, inn=an[0], outt=an[1]), origin='lower', vmin=-n, vmax=n, cmap='magma')
        plt.xlim(200, 250)
        plt.ylim(200, 250)
        plt.colorbar()
        plt.title(PSFpath.replace('do', '')+' FWHM='+str(FWHM))
        plt.savefig(output_prefix+'-FM-img-'+PSFpath.replace('do', '')+'.png', dpi=300)

        # Your variables here
        # some basics to point towards your model
        sep = sep  # only needs a guess
        pa = pa  # guess here too
        length = 2.5  # guess here also

        # set some boundaries for your MCMC
        x_range = 2.5  # in pixels, anywhere from 1.5-5 is reasonable
        y_range = 2.5  # same as x
        flux_range = 10  # [2e3,2e4] # flux can vary by an order of magnitude
        corr_len_range = 3  # between 0.3 and 30

        # and finally some parameters for the MCMC run
        nwalkers = 8

        nburn = 1000
        nsteps = 10000

        # output to save chain to
        pklout = outputdir+'\\'+prefix+PSFpath.replace('do', '')+'_chain.pkl'

        # get FM frame
        fm_frame = fits.getdata(output_prefix + "-fmpsf-KLmodes-all.fits")[0]
        fm_header = fits.getheader(output_prefix + "-fmpsf-KLmodes-all.fits")
        fm_centx = fm_header['PSFCENTX']
        fm_centy = fm_header['PSFCENTY']

        # get data_stamp frame
        data_frame = fits.getdata(output_prefix + "-klipped-KLmodes-all.fits")[0]
        data_header = fits.getheader(output_prefix + "-klipped-KLmodes-all.fits")
        data_centx = data_header['PSFCENTX']
        data_centy = data_header['PSFCENTY']

        # get initial guesses. Should be in the header but aren't?
        guesssep = sep
        guesspa = pa

        # create FM Astrometry object - 13 is fitboxsize
        fma = fitpsf.FMAstrometry(guesssep, guesspa, 13)

        # generate FM stamp
        # padding should be greater than 0 so we don't run into interpolation problems
        fma.generate_fm_stamp(fm_frame, [fm_centx, fm_centy], padding=5)

        # generate data_stamp stamp
        # note that dr=4 means we are using a 4 pixel wide annulus to sample the noise for each pixel
        # exclusion_radius excludes all pixels less than that distance from the estimated location of the planet
        fma.generate_data_stamp(data_frame, [data_centx, data_centy], dr=4, exclusion_radius=3)

        # set kernel, no read noise
        corr_len_guess = length
        corr_len_label = r"$l$"
        fma.set_kernel("matern32", [corr_len_guess], [corr_len_label])

        # set bounds based on given boundaries
        fma.set_bounds(x_range, y_range, flux_range, [corr_len_range])

        t0 = time.time()

        # run MCMC fit
        fma.fit_astrometry(nwalkers=nwalkers, nburn=nburn, nsteps=nsteps, numthreads=4, chain_output=pklout)

        t1 = time.time()
        print("time taken: ", str(np.round(t1-t0)), " seconds")

        fma.propogate_errs(star_center_err=0.1, platescale=MagAO.MagAOData.lenslet_scale*1000, platescale_err=0.000015, pa_offset=-0.59, pa_uncertainty=0.3)

        # We load in the results of the MCMC from the compressed pickle file
        # the pyklip code has generated for us

        chain_info = pickle.load(open(pklout, "rb"))

        # First, let's plot the chains

        fig = plt.figure(figsize=(10, 8))
        # plot RA offset
        ax1 = fig.add_subplot(411)
        ax1.plot(chain_info[:, :, 0].T, '-', color='k', alpha=0.3)
        ax1.set_xlabel("Steps")
        ax1.set_ylabel(r"$\Delta$ RA")

        # plot Dec offset
        ax2 = fig.add_subplot(412)
        ax2.plot(chain_info[:, :, 1].T, '-', color='k', alpha=0.3)
        ax2.set_xlabel("Steps")
        ax2.set_ylabel(r"$\Delta$ Dec")

        # plot flux scaling
        ax3 = fig.add_subplot(413)
        ax3.plot(chain_info[:, :, 2].T, '-', color='k', alpha=0.3)
        ax3.set_xlabel("Steps")
        ax3.set_ylabel(r"$\alpha$")

        # plot hyperparameters.. we only have one for this example: the correlation length
        ax4 = fig.add_subplot(414)
        ax4.plot(chain_info[:, :, 3].T, '-', color='k', alpha=0.3)
        ax4.set_xlabel("Steps")
        ax4.set_ylabel(r"$l$")

        plt.savefig(output_prefix+'_BKA_chain'+PSFpath.replace('do',str(FWHM))+'.png')

        # the other two figures are easier to make, because there are methods
        # already written

        # Second is the corner plot
        fig = fma.make_corner_plot()
        plt.savefig(output_prefix+'_BKA_corner'+PSFpath.replace('do',str(FWHM))+'.png', transparent=True, dpi=300)

        # And third is the model comparison and residuals
        fig, resids = fma.best_fit_and_residuals()
        plt.savefig(output_prefix+'_BKA_residuals'+PSFpath.replace('do',str(FWHM))+'.png', transparent=True, dpi=300)

        residuals.append(resids)

    for resid in residuals:
        total = np.nansum(resid)
        print('residual: '+str(total))
