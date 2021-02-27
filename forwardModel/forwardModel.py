# Class implementation of bayesian klip astrometry foward modeling
# original implementation written by Alex Watson circa 2017
# expanded into script by William Balmer Sept. 2020
# expanded into current class implementation by William Balmer 9/22/2020
# based on pyklip documentation found here:
# https://pyklip.readthedocs.io/en/latest/bka.html

# original implementation written by Alex Watson circa 2017
# expanded by William Balmer
# last edited 9/18/2020
# based on pyklip documentation found here:
# https://pyklip.readthedocs.io/en/latest/bka.html

# import statements
import os
import glob
import warnings
import numpy as np
from astropy.io import fits

import pyklip.instruments.MagAO as MagAO
import pyklip.fmlib.fmpsf as fmpsf
import pyklip.fm as fm

import GhostIsolation as ghost
from gaussian import Gaussian

warnings.filterwarnings('ignore')


class forwardModel():
    '''
    A class that uses pyKLIP's forward model functionality to generate and
    forward model MagAO data for a BKA MCMC fit.
    '''

    def __init__(self, filepaths, output, prefix, KLmode, sep, pa, contrast,
                 annuli, move, scale, ePSF=None, FWHM=None, cores=1,
                 highpass=True, **kwargs):
        if __name__ == '__main__':  # An important precaution for Windows
            __spec__ = None  # Important for ipynb compatibility
            # all the stuff goes here
        if ePSF is None:
            print('You have not provided a path to your instrumental psf')
            cubepath = input('Enter the path to your MagAO image cube to generate one, or enter \'Gaussian\' to use a simple gaussian psf: ')
            if cubepath == 'Gaussian':
                ePSF = 'doGaussian'
            else:
                ePSF = 'ghost.fits'

        # set paths to sliced dataset, call dataset into KLIP format
        # set up variables needed for KLIP calls
        self.filepaths = filepaths
        self.filelist = glob.glob(self.filepaths)
        self.dataset = MagAO.MagAOData(self.filelist)
        self.head = fits.getheader(self.filelist[0])
        self.pre = prefix
        try:
            a = len(annuli)
            self.annulus_bounds = [annuli]  # annulus centered on the planet
        except TypeError:
            self.annulus_bounds = annuli
        self.move = move
        self.fwhm = FWHM
        self.cores = cores
        self.ePSF = ePSF

        # setup FM guesses
        if 'numbasis' in kwargs:
            self.numbasis = np.array([kwargs['numbasis']])  # KL basis cutoffs you want
        else:
            self.numbasis = np.array([KLmode])
        self.guesssep = sep  # estimate of separation in pixels
        self.guesspa = pa  # estimate of position angle, in degrees
        self.guessflux = contrast  # estimated contrast
        self.dn_per_contrast = np.zeros((self.dataset.input.shape[0]))
        for i in range(self.dn_per_contrast.shape[0]):
            self.dn_per_contrast[i] = scale  # factor to scale PSF to star
        self.guessspec = np.array([1])  # our data is 1D in wavelength

        # PSF subtraction parameters
        self.outputdir = output  # where to write the output files
        self.prefix = prefix  # fileprefix for the output files
        self.subsections = 1  # we are not breaking up the annulus
        self.padding = 0  # we are not padding our zones
        self.movement = move
        self.hpf = highpass

        print('Parameters set, ready to begin forward modeling... ')

    def prep_KLIP(self):
        '''
        Sets up instrumental psf by running the construct_inst_PSF method,
        and then initializing the pyklip fm_class object which is required
        to forward model through KLIP (via the run_KLIP method)
        '''
        # make sure the outputdir exists and if not create it
        try:
            root = os.getcwd()
            os.makedirs(root+'\\'+self.outputdir)
            print('saving files to: .\\'+self.outputdir)
        except OSError:
            print('saving files to: .\\'+self.outputdir)
        self.construct_inst_PSF()  # sets self.psf2 == instrumental psf
        if self.fwhm is None:
            print('instrumental PSF FWHM is: '+str(self.head["0PCTFWHM"]))
        else:
            print('instrumental PSF FWHM is: '+str(self.fwhm))
        # initialize the FM Planet PSF class
        self.fm_class = fmpsf.FMPlanetPSF(self.dataset.input.shape,
                                          self.numbasis, self.guesssep,
                                          self.guesspa, self.guessflux,
                                          self.psf2,
                                          np.unique(self.dataset.wvs),
                                          self.dn_per_contrast,
                                          star_spt='A6',
                                          spectrallib=[self.guessspec])
        print('fm_class ready for KLIP')

    def run_KLIP(self):
        '''
        Runs klip on the dataset with the fm_class object given all input
        parameters. Last step before MCMC
        '''
        # run KLIP-FM
        fm.klip_dataset(self.dataset, self.fm_class, mode="ADI",
                        outputdir=self.outputdir, fileprefix=self.prefix,
                        numbasis=self.numbasis, annuli=self.annulus_bounds,
                        subsections=self.subsections, padding=self.padding,
                        movement=self.movement, numthreads=self.cores,
                        highpass=self.hpf)
        print('Done constructing forward model! You are ready to MCMC.')

    def construct_inst_PSF(self):
        '''
        Constructs the instrumental psf to use as the forwarded model, either
        from an instrumental ghost or a gaussian. FWHM of gaussian can be
        input or taken from header of files in filelist
        '''
        if self.fwhm is None:
            fwhm = self.head["0PCTFWHM"]
        else:
            fwhm = self.fwhm
        if self.ePSF == 'doGaussian':
            gauss = Gaussian(31, fwhm)
            psf = gauss.g
        elif self.ePSF == 'doMoffat':
            ghostdata, moffat = ghost.ghostIsolation(self.filepaths, 380, 220, 10, fwhm, 1)
            psf = moffat
        elif self.ePSF == 'doGhost':
            ghostdata, moffat = ghost.ghostIsolation(self.filepaths, 380, 220, 10, fwhm, 1)
            psf = ghostdata
        else:
            psf = fits.getdata(self.ePSF)
        # shape instrumental psf how pyklipFM wants
        psf2 = np.zeros((1, psf.shape[0], psf.shape[1]))
        psf2[0] = psf
        self.psf2 = psf2
