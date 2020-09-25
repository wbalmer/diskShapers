# imports
import os
import glob
import pyklip
import numpy as np
# import gaplanets as gp
from astropy.io import fits
import pyklip.fakes as fakes
import matplotlib.pyplot as plt
import pyklip.instruments.MagAO as MagAO
from gaussian import Gaussian
import warnings
warnings.filterwarnings('ignore')

class planetHole():
    '''
    A class to perform negative planet injection in order to determine
    companion astrometry.
    Initialized: William B. 9/22/2020
    '''

    def __init__(self, filepath, prefix, outputdir, contrast, sep, theta,
                 fwhm=None, ghostpath='ghost.fits', highpass=False,
                 klipparams=[6, 3, [1, 5, 50]], usegaussian=False):
        '''
        Initializes the class, injecting hole as specified and prepared for
        KLIP reduction via "run_KLIP" function
        Initialized: William B. 9/22/2020
        '''
        self.filelist = glob.glob(filepath+'\\*.fits')
        self.head = fits.getheader(self.filelist[0])
        self.highpass = highpass
        self.dataset = MagAO.MagAOData(self.filelist, highpass=self.highpass)
        self.contrast = contrast
        self.sep = sep
        self.theta = theta
        if fwhm is None:
            self.fwhm = self.head["0PCTFWHM"]
        else:
            self.fwhm = fwhm
        self.ghostpath = ghostpath
        self.usegaussian = usegaussian
        self.psf = self.instrPSF(self.ghostpath)

        self.inpflux = np.zeros((self.dataset.input.shape[0],
                                 self.psf.shape[0], self.psf.shape[1]))

        for i in range(self.dataset.input.shape[0]):
            self.inpflux[i] = self.contrast*self.psf

        fakes.inject_planet(self.dataset.input, self.dataset.centers,
                            self.inpflux, self.dataset.wcs, self.sep,
                            self.theta, fwhm=self.fwhm)

        self.outputdir = outputdir
        self.pfx = prefix
        self.numann, self.movm, self.KLlist = klipparams

    def run_KLIP(self):
        '''
        Runs a parallelized klip reduction using input parameters on the
        negative planet injection dataset.
        Initialized: William B. 9/22/2020
        '''
        pyklip.parallelized.klip_dataset(self.dataset,
                                         outputdir=self.outputdir,
                                         fileprefix=self.pfx,
                                         algo='klip', annuli=self.numann,
                                         subsections=1, movement=self.movm,
                                         numbasis=self.KLlist,
                                         calibrate_flux=False,
                                         mode="ADI", highpass=self.highpass,
                                         save_aligned=False,
                                         time_collapse='median')

        result = self.outputdir+'\\'+self.pfx+'-KLmodes-all.fits'
        print('KLIP result is saved to: '+result)
        resultdata = fits.getdata(result)
        self.resultdata = resultdata
        return

    def instrPSF(self, ghostpath):
        '''
        Grabs instrumental psf for negative planet injection.
        Intended to be expanded to include default gaussian
        Initialized: William B. 9/22/2020
        '''
        if self.usegaussian is True:
            gauss = Gaussian(31, self.fwhm)
            psf = gauss.g
        else:
            psf = fits.getdata(ghostpath)
        return psf

    def showit(self, image, save='n', lims='n', cmap='magma', name='showit.png', vmin=None, vmax=None):
        '''
        Displays image data cleanly with attractive colorbar and limits
        Initialized: William B. 9/22/2020
        '''
        plt.figure(figsize=(7, 7))
        plt.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')  # plots image
        plt.colorbar()  # plots colorbar
        if lims != 'n':
            plt.xlim(lims[0][0], lims[0][1])
            plt.ylim(lims[1][0], lims[1][1])
        if save == 'y':
            plt.savefig(name, dpi=150)
        return

    def showresult(self, KLmode=2, save='n', name='negplanetinj.png', vmin=None, vmax=None):
        '''
        Runs showit function on result of KLIP reduction
        Initialized: William B. 9/22/2020
        '''
        self.showit(self.resultdata[KLmode], lims=[[205, 225], [210, 230]],
                    save='n', name=name, vmin=vmin, vmax=vmax)
        return
