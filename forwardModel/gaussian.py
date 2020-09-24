# Create a gaussian and save as a fits file
# Used if ghost model doesn't match point source
# Initialized William Balmer 9/18/2020
# gaussian function inspired by gist #4635563 by user andrewgiessel

# Converted from script to class by William Balmer 9/22/2020

# imports
import numpy as np
from astropy.io import fits


class Gaussian():
    '''
    A class to call when you need a gaussian generated
    '''

    def __init__(self, shape, FWHM, save='n'):
        '''
        initialize the class
        '''
        self.gaussian = 1

        # run script
        self.shape = shape
        self.FWHM = FWHM
        self.g = self.Gaussian()
        if save == 'y':
            self.saveGaussian()

    def Gaussian(self):
        """
        Generate a 2D gaussian within a square array
        """

        x = np.arange(0, self.shape, 1, float)
        y = x[:, np.newaxis]

        x0 = y0 = self.shape // 2

        return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / self.FWHM**2)

    def saveGaussian(self):
        fits.writeto('gausFWHM'+str(self.FWHM)+'.fits', self.g, overwrite=True)
        return
