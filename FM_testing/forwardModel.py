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
import glob
import warnings
import numpy as np
from astropy.io import fits

import pyklip.instruments.MagAO as MagAO
import pyklip.fmlib.fmpsf as fmpsf
import pyklip.fm as fm

import GhostIsolation as Ghost
from gaussian import Gaussian

warnings.filterwarnings('ignore')


class forwardModel():
    '''
    A class that uses pyKLIP's forward model functionality to generate and
    forward model MagAO data for a BKA MCMC fit.
    '''

    def __init__(self, filepaths, output, prefix, KLmode, sep, pa, contrast,
                 annulus1, annulus2, move, scale, PSFpath=None, FWHM=None):
        if __name__ == '__main__':  # An important precaution for Windows
            __spec__ = None  # Important for ipynb compatibility
            # all the stuff goes here
        if PSFpath is None:
            print('You have not provided a path to your instrumental psf')
            cubepath = input('Enter the path to your MagAO image cube to generate one, or enter \'Gaussian\' to use a simple gaussian psf: ')
            if cubepath == 'Gaussian':
                PSFpath = 'doGaussian'
            else:
                Ghost.ghostIsolation(cubepath, 380, 220, 10, 10, 10)
            PSFpath = 'ghost.fits'

        # set paths to sliced dataset, call dataset into KLIP format
        # set up variables needed for KLIP calls
        self.filelist = glob.glob(filepaths)
        self.dataset = MagAO.MagAOData(self.filelist)
        self.head = fits.getheader(self.filelist[0])
        self.output = output
        self.pre = prefix
        annulus = [int(annulus1), int(annulus2)]
        self.move = move
        self.fwhm = FWHM

        self.PSFpath = PSFpath

        self.psf2 = self.construct_inst_PSF()

        # setup FM guesses
        self.numbasis = np.array([KLmode])  # KL basis cutoffs you want
        self.guesssep = sep  # estimate of separation in pixels
        self.guesspa = pa  # estimate of position angle, in degrees
        self.guessflux = contrast  # estimated contrast
        self.dn_per_contrast = np.zeros((self.dataset.input.shape[0]))
        for i in range(self.dn_per_contrast.shape[0]):
            self.dn_per_contrast[i] = scale  # factor to scale PSF to star
        self.guessspec = np.array([1])  # our data is 1D in wavelength

        # initialize the FM Planet PSF class

        self.fm_class = fmpsf.FMPlanetPSF(self.dataset.input.shape,
                                          self.numbasis, self.guesssep,
                                          self.guesspa, self.guessflux,
                                          self.psf2,
                                          np.unique(self.dataset.wvs),
                                          self.dn_per_contrast,
                                          star_spt='A6',
                                          spectrallib=[self.guessspec])

        # PSF subtraction parameters
        self.outputdir = output  # where to write the output files
        self.prefix = prefix  # fileprefix for the output files
        self.annulus_bounds = [annulus]  # annulus centered on the planet
        self.subsections = 1  # we are not breaking up the annulus
        self.padding = 0  # we are not padding our zones
        self.movement = move

        print('Parameters set, beginning forward modeling... ')
        self.run_KLIP()
        return

    def run_KLIP(self):
        # run KLIP-FM
        fm.klip_dataset(self.dataset, self.fm_class, mode="ADI",
                        outputdir=self.outputdir, fileprefix=self.prefix,
                        numbasis=self.numbasis, annuli=self.annulus_bounds,
                        subsections=self.subsections, padding=self.padding,
                        movement=self.movement)
        print('Done constructing forward model! You are ready to MCMC.')

    def construct_inst_PSF(self):
        '''
        Constructs the instrumental psf to use as the forwarded model, either
        from an instrumental ghost or a gaussian. FWHM of gaussian can be
        input or taken from header of files in filelist
        '''
        if self.PSFpath == 'doGaussian':
            if self.fwhm is None:
                fwhm = self.head["0PCTFWHM"]
            else:
                fwhm = self.fwhm
            psf = Gaussian(31, fwhm)
        else:
            psf = fits.getdata(self.PSFpath)
        # shape instrumental psf how pyklipFM wants
        psf2 = np.zeros((1, psf.shape[0], psf.shape[1]))
        psf2[0] = psf
        return(psf2)
