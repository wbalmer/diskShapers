# imports
from scipy import ndimage
import numpy as np


# functions
def linearizeClio2(rawimage):
    '''
    linearizeClio2
    --------------
    Linearizes Clio2 raw image according to information in document on Clio wiki
    https://magao-clio.github.io/zero-wiki/6d927/attachments/69b81/linearity_clio.pdf
    Coefficients were measured on Comm2 March-April 2013

    Modification History:
    Function originally written in IDL 2013/04/10 by Katie Morzinski (ktmorz@arizona.edu)
    Ported into python 10/13/2020 by William Balmer (williamobalmer@gmail.com)

    inputs
    --------------
    rawimage      : un-linearized image, usually gather via astropy.io.fits

    outputs
    --------------
    im           : linearized image, ready for further use
    '''
    im = rawimage
    coeff3 = [112.575, 1.00273, -1.40776e-06, 4.59015e-11]
    # decoadd step


    # Only apply to pixels above 27,000 counts.
    coeff3 = [112.575, 1.00273, -1.40776e-06, 4.59015e-11]
    new = np.zeros((im.shape[0],im.shape[1]))
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if im[i,j] > 2.7e4:
                new[i,j] = coeff3[0] + coeff3[1]*im[i,j] + coeff3[2]*im[i,j]**2. + coeff3[3]*im[i,j]**3.
            else:
                new[i,j] = im[i,j]

    return new


def badpixelcorrect(data_arr, badpixelmask, speed='fast'):
    '''
    badpixelcorrect
    -----------------
    Performs a simple bad pixel correction, replacing bad pixels with image median.
    Input image and bad pixel mask image must be the same image dimension.

    Modification History:
    Provided circa March 2020 by Kim Ward-Duong
    Reproduced here 10/13/2020 by William Balmer

    inputs
    ----------------
    data_arr      : (matrix of floats) input image
    badpixelmask  : (matrix of floats) mask of values 1.0 or 0.0, where 1.0 corresponds to a bad pixel
    speed         : (str) whether to calculate the median filtered image (computationally intensive), or simply take the image median. Default = 'fast'

    outputs
    ---------------
    corr_data     : (matrix of floats) image corrected for bad pixels
    '''
    corr_data = data_arr.copy()
    if speed == 'slow':
        # smooth the science image by a median filter to generate replacement pixels
        median_data = ndimage.median_filter(data_arr, size=(30, 30))
        # replace the bad pixels with median of the local 30 pixels
        corr_data[badpixelmask == 1] = median_data[badpixelmask == 1]
    else:
        corr_data[badpixelmask == 1] = np.nanmedian(data_arr)

    return corr_data
