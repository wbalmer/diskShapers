# Create a gaussian and save as a fits file
# Used if ghost model doesn't match point source
# William Balmer 9/18/2020
# gaussian function inspired by gist #4635563 by user andrewgiessel

# this script takes sys inputs in the following order:
# "shape" as one integer, "fwhm" as one float

# imports
import sys
import numpy as np
from astropy.io import fits


# function
def Gaussian(shape, FWHM=5):
    """
    Generate a 2D gaussian within a square array
    """

    x = np.arange(0, shape, 1, float)
    y = x[:, np.newaxis]

    x0 = y0 = shape // 2

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / FWHM**2)


# run script
shape = int(sys.argv[1])
FWHM = float(sys.argv[2])

g = Gaussian(shape, FWHM=FWHM)
fits.writeto('gausFWHM'+str(FWHM)+'.fits', g, overwrite=True)
