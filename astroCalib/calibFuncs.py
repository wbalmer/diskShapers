# imports
import os
import glob
import numpy as np
import scipy.signal
from scipy.signal import peak_widths
from scipy import ndimage
from scipy.ndimage.interpolation import shift
from astropy.io import fits
import warnings
from photutils import DAOStarFinder, CircularAperture
from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
import matplotlib.pyplot as plt
from astropy.nddata import NDData
from photutils.psf import extract_stars
from astropy.table import Table
from photutils import EPSFBuilder

from photutils.psf import IterativelySubtractedPSFPhotometry
from photutils import MMMBackground
from photutils.psf import IntegratedGaussianPRF, DAOGroup
from astropy.modeling.fitting import LevMarLSQFitter

import astropy.units as u


# functions
def calcDistance(x0, y0, x1, y1):
    return np.sqrt((x1 - x0)**2 + (y1-y0)**2)


def linearizeClio2(rawimage):
    '''
    linearizeClio2
    --------------
    Linearizes Clio2 raw image according to information in document on wiki
    https://magao-clio.github.io/zero-wiki/6d927/attachments/69b81/linearity_clio.pdf
    Coefficients were measured on Comm2 March-April 2013
    IMPORTANT NOTE: the clio image must be decoadded (i.e. divide by header
    keyword 'COADD' before linearized)

    Example Usage: linearized_image = linearizeClio2(decoadded_image)

    Modification History:
    Written in IDL 2013/04/10 by Katie Morzinski (ktmorz@arizona.edu)
    Ported into python 10/13/2020 by William Balmer (williamobalmer@gmail.com)

    inputs
    ------
    rawimage      : un-linearized image, usually gather via astropy.io.fits

    outputs
    -------
    im           : linearized image, ready for further use
    '''
    im = rawimage
    coeff3 = [112.575, 1.00273, -1.40776e-06, 4.59015e-11]

    # Only apply to pixels above 27,000 counts.
    coeff3 = [112.575, 1.00273, -1.40776e-06, 4.59015e-11]
    new = np.zeros((im.shape[0], im.shape[1]))
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            if im[i, j] > 2.7e4:
                new[i, j] = (coeff3[0] + coeff3[1]*im[i, j]
                             + coeff3[2]*im[i, j]**2. + coeff3[3]*im[i, j]**3.)
            else:
                new[i, j] = im[i, j]

    return new


def badpixelcorrect(data_arr, badpixelmask, speed='fast'):
    '''
    badpixelcorrect
    ---------------
    Performs a simple bad pixel correction, replacing bad pixels with image
    median. Input image and bad pixel mask image must be the same image
    dimension.

    Example Usage: corrected_img = badpixelcorrect(image, badpixelmask)

    Modification History:
    Provided to author by Kim Ward-Duong circa March 2020
    Reproduced here 10/13/2020 by William Balmer

    inputs
    ------
    data_arr      : (matrix of floats) input image
    badpixelmask  : (matrix of floats) mask of values 1.0 or 0.0, where
                    1.0 corresponds to a bad pixel
    speed         : (str) whether to calculate the median filtered image
                    (computationally intensive), or simply take the image
                    median. Default = 'fast'

    outputs
    -------
    corr_data     : (matrix of floats) image corrected for bad pixels
    '''
    corr_data = data_arr.copy()
    if speed == 'slow':
        # smooth the science image by a median filter to generate replacement
        median_data = ndimage.median_filter(data_arr, size=(30, 30))
        # replace the bad pixels with median of the local 30 pixels
        corr_data[badpixelmask == 1] = median_data[badpixelmask == 1]
    else:
        corr_data[badpixelmask == 1] = np.nanmedian(data_arr)

    return corr_data


def centroid(data_arr, xcen, ycen, nhalf=5, derivshift=1.):
    '''
    centroid
    -----------------
    based on dimension-independent line minimization algorithms
    implemented in IDL cntrd.pro

    Example Usage:

    Modification History:
    Original implementation by M. Petersen, J. Lowenthal, K. Ward-Duong.
    Updated, provided to author by S. Betti. circa March 2020
    Reproducted here 10/15/2020 by William Balmer

    inputs
    ----------------
    data_arr      : (matrix of floats) input image
    xcen          : (int) input x-center guess
    ycen          : (int) input y-center guess
    nhalf         : (int, default=5) the excised box of pixels to use
                    recommended to be ~(2/3) FWHM (only include star pixels).
    derivshift    : (int, default=1) degree of shift to calculate derivative.
                     larger values can find shallower slopes more efficiently



    outputs
    ---------------
    tuple
    xcenf         : the centroided x value
    ycenf         : the centroided y value

    dependencies
    ---------------
    numpy         : imported as np




    also see another implementation here:
    https://github.com/djones1040/PythonPhot/blob/master/PythonPhot/cntrd.py

    '''
    # input image requires the transpose to
    # find the maximum value near the given point
    data = data_arr[int(ycen-nhalf):int(ycen+nhalf+1),
                    int(xcen-nhalf):int(xcen+nhalf+1)]

    yadjust = nhalf - np.where(data == np.max(data))[0][0]
    xadjust = nhalf - np.where(data == np.max(data))[1][0]

    xcen -= xadjust
    ycen -= yadjust

    # now use the adjusted centers to find a better square
    data = data_arr[int(ycen-nhalf):int(ycen+nhalf+1),
                    int(xcen-nhalf):int(xcen+nhalf+1)]

    # make a weighting function
    ir = (nhalf-1) > 1

    # sampling abscissa: centers of bins along each of X and Y axes
    nbox = 2*nhalf + 1
    dd = np.arange(nbox-1).astype(int) + 0.5 - nhalf

    # #Weighting factor W unity in center, 0.5 at end, and linear in between
    w = 1. - 0.5*(np.abs(dd)-0.5)/(nhalf-0.5)
    sumc = np.sum(w)

    #
    # fancy comp sci part to find the local maximum
    #
    # this uses line minimization using derivatives
    # (see text such as Press' Numerical Recipes Chapter 10),
    # treating X and Y dimensions as independent (generally safe for stars).
    # In this sense the method can be thought of as a two-step gradient descent

    # find X centroid

    # shift in Y and subtract to get derivative
    deriv = np.roll(data, -1, axis=1) - data.astype(float)
    deriv = deriv[nhalf-ir:nhalf+ir+1, 0:nbox-1]
    deriv = np.sum(deriv, 0)  # Sum X derivatives over Y direction

    sumd = np.sum(w*deriv)
    sumxd = np.sum(w*dd*deriv)
    sumxsq = np.sum(w*dd**2)

    dx = sumxsq*sumd/(sumc*sumxd)

    xcenf = xcen - dx

    # find Y centroid

    # shift in X and subtract to get derivative
    deriv = np.roll(data, -1, axis=0) - data.astype(float)
    deriv = deriv[0:nbox-1, nhalf-ir:nhalf+ir+1]
    deriv = np.sum(deriv, 1)  # Sum X derivatives over Y direction

    sumd = np.sum(w*deriv)
    sumxd = np.sum(w*dd*deriv)
    sumxsq = np.sum(w*dd**2)

    dy = sumxsq*sumd/(sumc*sumxd)

    ycenf = ycen - dy

    return xcenf, ycenf


def cross_image(im1, im2, centerx, centery, boxsize=400, **kwargs):
    '''
    cross_image
    ---------------
    calcuate cross-correlation of two images in order to find shifts

    Example Usage: xshift, yshift = cross_image(reference_image,
                                                image_to_be_shifted,
                                                centerx,
                                                centery,
                                                boxsize=400)

    Modification History:
    Original implementation by M. Petersen, J. Lowenthal, K. Ward-Duong.
    Updated, provided to author by S. Betti. circa March 2020
    Reproducted here 10/15/2020 by William Balmer

    inputs
    ---------------
    im1                      : (matrix of floats)  first input image
    im2                      : (matrix of floats) second input image
    centerx                  : (float) x-center of subregion in reference image
    centery                  : (float) y-center of subregion in reference image
    boxsize                  : (integer) subregion of image to cross-correlate


    returns
    ---------------
    xshift                   : (float) x-shift in pixels
    yshift                   : (float) y-shift in pixels


    dependencies
    ---------------
    scipy.signal.fftconvolve : two-dimensional fourier convolution
    centroid                 : a centroiding algorithm of your definition
    numpy                    : imported as np

    todo
    ---------------
    -add more **kwargs capabilities for centroid argument

    '''

    # The type cast into 'float' is to avoid overflows:
    im1_gray = im1.astype('float')
    im2_gray = im2.astype('float')

    # Enable a trimming capability using keyword argument option.
    im1_gray = im1_gray[centery-boxsize:centery+boxsize,
                        centerx-boxsize:centerx+boxsize]
    im2_gray = im2_gray[centery-boxsize:centery+boxsize,
                        centerx-boxsize:centerx+boxsize]

    # Subtract the averages (means) of im1_gray and 2 from their respective arr
    im1_gray -= np.nanmean(im1_gray)
    im2_gray -= np.nanmean(im2_gray)

    # guard against extra nan values
    im1_gray[np.isnan(im1_gray)] = np.nanmedian(im1_gray)
    im2_gray[np.isnan(im2_gray)] = np.nanmedian(im2_gray)

    # Calculate the correlation image using fast Fourier Transform (FFT)
    # Note the flipping of one of the images (the [::-1]) to act as a high-pass
    corr_image = scipy.signal.fftconvolve(im1_gray, im2_gray[::-1, ::-1],
                                          mode='same')

    # Find the peak signal position in the cross-correlation,
    # which gives the shift between the images
    corr_tuple = np.unravel_index(np.nanargmax(corr_image), corr_image.shape)

    try:  # try to use a centroiding algorithm to find a better peak
        xcenc, ycenc = centroid(corr_image.T, corr_tuple[0], corr_tuple[1],
                                nhalf=10, derivshift=1.)
    except:  # if centroiding algorithm fails, use just peak pixel
        xcenc, ycenc = corr_tuple
    finally:
        # Calculate shifts (distance from central pixel of cross-correlated im)
        xshift = xcenc - corr_image.shape[0]/2.
        yshift = ycenc - corr_image.shape[1]/2.

    return xshift, yshift


def shift_image(image, xshift, yshift):
    '''
    shift_image
    -------------
    wrapper for scipy's implementation that shifts images according to values
    from cross_image

    Example Usage: output_image = shift_image(image_to_be_shifted, xsh, ysh)

    Modification History:
    Original implementation by M. Petersen, J. Lowenthal, K. Ward-Duong.
    Updated, provided to author by S. Betti. circa March 2020
    Reproducted here 10/15/2020 by William Balmer

    inputs
    ------------
    image           : (matrix of floats) image to be shifted
    xshift          : (float) x-shift in pixels
    yshift          : (float) y-shift in pixels

    outputs
    ------------
    shifted image   : shifted, interpolated image.
                      same shape as input image, with zeros filled where
                      the image is rolled over


    '''
    return shift(image, (xshift, yshift))


def createMasterDark(darklist, inttime, save='n', savepath='./'):
    '''
    Takes a list of darks, scales them to some common integration time
    and returns the result
    '''
    # length of cube
    n = len(darklist)

    # gather first image info
    first_frame_data = fits.getdata(darklist[0])
    first_frame_head = fits.getheader(darklist[0])

    # saves shape of images as variable
    imsize_y, imsize_x = first_frame_data.shape

    # creates empty stack of depth n
    fits_stack = np.zeros((imsize_y, imsize_x, n))

    # adds images to stack
    for ii in range(0, n):
        im = fits.getdata(darklist[ii])
        hdr = fits.getheader(darklist[ii])
        intT = hdr['INT']
        im2 = inttime*(im/intT)
        fits_stack[:, :, ii] = im2

    # takes median of stack, saves as var
    med_frame = np.nanmedian(fits_stack, axis=2)

    if save == 'y':
        savefile = savepath+'/medDark'+str(inttime)+'sInt.fits'
        fits.writeto(savefile, med_frame, first_frame_head, overwrite=True)
    else:
        return med_frame


def darkSub(image, imgint, masterdark, masterdark_int):
    '''
    Takes in an image array, a masterdark array, the integration time
    for that masterdark, and returns the dark subtraction of that
    image array
    '''
    dark = imgint*(masterdark/masterdark_int)
    # some imgs are like 300 pix tall and need to be padded to 512
    if dark.shape[0] > image.shape[0]:
        newim = np.zeros((dark.shape[0], dark.shape[1]))
        numbery = dark.shape[0] - image.shape[0]
        if dark.shape[1] > image.shape[1]:
            numberx = dark.shape[1] - image.shape[1]
            newim[numbery:newim.shape[0], numberx:newim.shape[1]] = image
        else:
            newim[numbery:newim.shape[0], :newim.shape[1]] = image
        sub = newim-dark
    else:
        sub = image-dark

    return sub


def runSubtraction(imlist, masterdark, masterdark_int, badpixelmask):
    '''

    '''
    for im in imlist:
        hdr = fits.getheader(im)
        imgint = hdr['INT']
        imgdata = fits.getdata(im)
        # check if cube:
        if len(imgdata.shape) > 2:
            for i in range(imgdata.shape[0]):
                # decoadd
                imcoadd = imgdata[i]/hdr['COADDS']
                # linearize
                imlin = linearizeClio2(imcoadd)
                # dark subtract
                imsub = darkSub(imlin, imgint, masterdark, masterdark_int)
                # bad pixel correct
                imbpcorr = badpixelcorrect(imsub, badpixelmask)

                newpath = im.replace('.fit', '_LDBP_im'+str(i+1)+'.fit')

                fits.writeto(newpath, imbpcorr, hdr, overwrite=True)
        else:
            # decoadd
            imcoadd = imgdata/hdr['COADDS']
            # linearize
            imlin = linearizeClio2(imcoadd)
            # dark subtract
            imsub = darkSub(imlin, imgint, masterdark, masterdark_int)
            # bad pixel correct
            imbpcorr = badpixelcorrect(imsub, badpixelmask)

            newpath = im.replace('.fit', '_LDBP.fit')

            fits.writeto(newpath, imbpcorr, hdr, overwrite=True)

    return ('Data is linearized, dark subtracted, and bad pixel corrected!')


def crosscube(imcube, cenx, ceny, box=50, returnmed='y', returncube='n'):
    '''

    '''
    im1 = imcube[0]

    xshifts = {}
    yshifts = {}
    cube = np.zeros([imcube.shape[0], im1.shape[0], im1.shape[1]])

    for index in range(imcube.shape[0]):
        im = imcube[index]
        xshifts[index], yshifts[index] = cross_image(im1, im, centerx=cenx,
                                                     centery=ceny, boxsize=box)
        cube[index, :, :] = shift_image(im, xshifts[index], yshifts[index])

    # Calculate trim edges of new median stacked images so all stacked images
    # of each target have same size
    max_x_shift = np.max(np.abs([xshifts[x] for x in xshifts.keys()]))
    max_y_shift = np.max(np.abs([yshifts[x] for x in yshifts.keys()]))

    print('Max x-shift={0}, max y-shift={1} (pixels)'.format(max_x_shift,
                                                             max_y_shift))
    median_image = np.nanmedian(cube, axis=0)

    print('\n Done stacking!')
    if returnmed == 'y' and returncube != 'y':
        return median_image
    elif returnmed != 'y' and returncube == 'y':
        return cube
    elif returnmed == 'y' and returncube == 'y':
        return median_image, cube


def nodSubtraction(imlist, path='nodsubcube.fits'):
    '''

    '''
    # get first image for shape of cubes
    im1 = imlist[0]
    im1data = fits.getdata(im1)
    im1head = fits.getheader(im1)

    # create list of nods
    nod1 = []
    nod2 = []
    for im in imlist:
        hdr = fits.getheader(im)
        if hdr['BEAM'] == 0:
            nod1.append(im)
        elif hdr['BEAM'] == 1:
            nod2.append(im)
        else:
            print('there are more than two nods!')
    # prepare median nods for subtraction
    nod_A_cube = np.zeros((len(nod1), im1data.shape[0], im1data.shape[1]))
    nod_B_cube = np.zeros((len(nod2), im1data.shape[0], im1data.shape[1]))

    j = 0
    k = 0
    for i in range(len(imlist)):
        img = imlist[i]
        imdata = fits.getdata(img)
        if img in nod1:
            nod_A_cube[j] = imdata
            j += 1
        elif img in nod2:
            nod_B_cube[k] = imdata
            k += 1

    a_nod_med = np.nanmedian(nod_A_cube, axis=0)
    b_nod_med = np.nanmedian(nod_B_cube, axis=0)

    # perform nod subtraction on each image
    nodsubs = np.zeros((len(imlist), im1data.shape[0], im1data.shape[1]))
    j = 0
    k = 0
    for i in range(len(imlist)):
        img = imlist[i]
        if img in nod1:
            # subtract opposite nod median from single image
            imdata = nod_A_cube[j]
            bg_sub_imdata = b_nod_med - imdata  # imdata - b_nod_med
            nodsubs[i] = bg_sub_imdata
            j += 1
        elif img in nod2:
            imdata = nod_B_cube[k]
            bg_sub_imdata = a_nod_med - imdata  # imdata - a_nod_med
            nodsubs[i] = bg_sub_imdata
            k += 1

    # write the cube of unshifted nod subtracted images to disk
    fits.writeto(path, nodsubs, header=im1head, overwrite=True)

    return print('Cube of nod subtracted data writen to '+path)


def shiftNoddedData(cubepath, ret='y', cube='n', med='n'):
    '''

    '''
    nodsubcube = fits.getdata(cubepath)
    head = fits.getdata(cubepath)
    # run cross correlation shift, get cube and median image
    shiftnodsubmed, shiftnodsubcube = crosscube(nodsubcube, cenx=250,
                                                ceny=250, box=250,
                                                returnmed='y',
                                                returncube='y')

    if cube == 'y':
        fits.writeto(cubepath.replace('.fits', '_shiftcube.fits'),
                     shiftnodsubcube, header=head, overwrite=True)

    if med == 'y':
        fits.writeto(cubepath.replace('.fits', '_shiftmed.fits'),
                     shiftnodsubcube, header=head, overwrite=True)

    if ret == 'y':
        return shiftnodsubmed, shiftnodsubcube


def filelister(filelist, day, band, targ, datedict, wavedict, objdict):
    '''
    Cross checks dictionaries of day, band, and target names and compiles
    a list of the images in a filelist that match the day, band, and targ
    specified.
    '''
    thelist = []
    for file in filelist:
        if datedict[file] == day:
            if wavedict[file] == band:
                if objdict[file] == targ:
                    thelist.append(file)
    return thelist


def sortData(datadir, instrument='CLIO2', filesufx='*.fit*'):
    '''
    Sorts through a data directory filled with Clio data, sorts it by epoch,
    target, and passband and returns lists of sorted data, as well as a list of
    the darks in all the folders as a separate list.
    '''
    # ignore warnings raised about NIRC2 header contents
    warnings.filterwarnings("ignore")

    names = []
    bands = []
    days = []
    uniques = []
    darks = []
    ao_on_images = []

    wavedict = {}
    objdict = {}
    datedict = {}

    images = glob.glob(datadir+'\\*\\'+filesufx)

    if images is []:
        raise FileNotFoundError("Empty data directory!")

    for im in images:
        hdr = fits.getheader(im)
        if instrument == 'CLIO2':
            name = hdr['CID']
            passband = hdr['PASSBAND']
            date = hdr['DATE']
            day = date.split('T')[0]
            unique = day+'&'+passband+'&'+name
            if passband == 'Blocked':  # if passband is 'blocked' then it is a dark
                darks.append(im)
            elif hdr['LOOP'] == 1:  # ensure AO is on for images used
                ao_on_images.append(im)

        elif instrument == 'NIRC2':
            name = hdr['TARGNAME']
            passband = hdr['FILTER']
            date = hdr['DATE-OBS']
            day = date.split('T')[0]
            unique = day+'&'+passband+'&'+name
            # currently using all NIRC2 images, so no need to filter for AO
            ao_on_images.append(im)
        else:
            raise ValueError('the currently supported instruments are NIRC2 and CLIO2')

        if name not in names:
            names.append(name)
        if passband not in bands:
            bands.append(passband)
        if day not in days:
            days.append(day)
        if unique not in uniques:
            uniques.append(unique)
        wavedict.update({im: passband})
        objdict.update({im: name})
        datedict.update({im: day})

    datasets = []
    for unique in uniques:
        day, band, name = unique.split('&')
        liste = filelister(ao_on_images, day, band, name, datedict=datedict,
                           wavedict=wavedict, objdict=objdict)
        datasets.append(liste)

    if instrument == 'NIRC2':
        return datasets
    else:
        return datasets, darks


def runtheReduction(datadir, badpixelpath, intTime=300):
    '''
    '''
    # get data
    datasets, darks = sortData(datadir)
    # get bad pixel map
    badpixelmap = fits.getdata(badpixelpath)
    # create master dark
    med_dark = createMasterDark(darks, intTime)
    # loop through datasets and correct them
    for dataset in datasets:
        runSubtraction(dataset, med_dark, intTime, badpixelmap)
    # get new subtracted data
    reduced_data, darks2 = sortData(datadir, filesufx='*_LDBP*.fit*')
    # run nodSubtraction
    for dataset in reduced_data:
        savepath = datadir+'/reduced/'
        try:
            os.makedirs(savepath)
        except OSError:
            print('putting files in an existing folder.')
        head = fits.getheader(dataset[0])
        band, name, datetime = head['PASSBAND'], head['CID'], head['DATE']
        date = datetime.split('T')[0]
        name = name.split(' * ')[-1]
        filename = name+'_'+band+'_'+date+'_nodsub.fits'
        nodSubtraction(dataset, path=savepath+filename)

    return print('Done reducing the files')


def starLocate(imagepath, thresh, fwhmguess, bright, stampsize=None,
               plot=True, roundness=0.5, crit_sep=15):
    '''
    starLocate
    ---------
    User selects reference psf source, then target source from image
    which is called via fits.getdata from imagepath. Returns the iterative
    photometry result ran on the target, which yields accurate position

    Example Usage: targx, targy, fwhm = findFirst('path/to/img.fits')

    Modification History:
    Initialized by William Balmer 10/30/2020

    inputs
    ------------
    imagepath           : (string) path to fits image
    thresh              : (float) threshold parameter for DAOStarFinder
    fwhmguess           : (float) fwhm parameter for DAOStarFinder
    bright              : (int) number of found stars to display
    stampsize           : (optional, default=None) size of subregion of image to analyse
    plot                : (optional, default=True) plot results or not
    roundness           : (optional, default=0.5) roundlo, roundhi parameter for DAOStarFinder
    crit_sep            : (optional, default=15)

    outputs
    ------------
    phot_results        : (astropy.table object)
    '''
    print('first, find a reference star to create a reference PSF')
    targx, targy, fwhm = findFirst(imagepath, thresh=thresh,
                                   fwhmguess=fwhmguess, bright=bright)
    if stampsize is None:
        stampsize = int(input('input the size of stamp: '))
    elif stampsize is not int:
        stampsize = int(stampsize)

    # read in img header to get pixel scale
    head = fits.getheader(imagepath)
    pixel_scale = float(head['PIXSCALE'])
    date = head['DATE-OBS']

    data = fits.getdata(imagepath)
    x0 = int(targx-(stampsize/2))
    x1 = int(targx+(stampsize/2))
    y0 = int(targy-(stampsize/2))
    y1 = int(targy+(stampsize/2))
    ref_stamp = data[y0:y1, x0:x1]

    epsf = makeEPSF(ref_stamp, plot=plot)

    print('next, select your target system to fit positions to')
    targx, targy, fwhm = findFirst(imagepath, thresh=thresh,
                                   fwhmguess=fwhmguess, bright=bright)

    x0 = int(targx-(stampsize/2))
    x1 = int(targx+(stampsize/2))
    y0 = int(targy-(stampsize/2))
    y1 = int(targy+(stampsize/2))
    stamp = data[y0:y1, x0:x1]

    daogroup = DAOGroup(crit_separation=crit_sep)
    mmm_bkg = MMMBackground()
    finder = DAOStarFinder(thresh, fwhm, roundlo=-roundness, roundhi=roundness,
                           sigma_radius=3)
    fitter = LevMarLSQFitter()
    phot_obj = IterativelySubtractedPSFPhotometry(finder=finder,
                                                  group_maker=daogroup,
                                                  bkg_estimator=mmm_bkg,
                                                  psf_model=epsf,
                                                  fitter=fitter,
                                                  fitshape=int(stampsize/2),
                                                  niters=1,
                                                  aperture_radius=7)

    phot_results = phot_obj(stamp)
    norm = ImageNormalize(stretch=LogStretch())
    pos = phot_results['x_fit', 'y_fit']
    positions = np.transpose((pos['x_fit'], pos['y_fit']))
    apertures = CircularAperture(positions, r=fwhm)

    if plot is True:
        plt.figure(figsize=(9, 9))
        plt.subplot(1, 2, 1)
        plt.imshow(stamp, origin='lower', norm=norm)
        apertures.plot(color='red', lw=1.5, alpha=0.7)
        plt.colorbar(orientation='horizontal')

        plt.subplot(1, 2, 2)
        plt.imshow(phot_obj.get_residual_image(), cmap='viridis',
                   origin='lower', interpolation='nearest', aspect=1)
        apertures.plot(color='red', lw=1.5, alpha=0.7)
        plt.colorbar(orientation='horizontal')

    phot_results['pixscale'] = pixel_scale
    phot_results['date'] = date

    return phot_results


def findFirst(imagepath, thresh=100, fwhmguess=5, bright=5):
    '''
    findFirst
    ---------
    Allows user to select target star in a fits image with a coarse
    DAOStarFinder search, and returns the centroid and FWHM of that target

    Example Usage: targx, targy, fwhm = findFirst('path/to/img.fits')

    Modification History:
    Initialized by William Balmer 10/30/2020

    inputs
    ------------
    imagepath           : (string) path to fits image
    thresh              : (optional, float) threshold parameter for DAOStarFinder
    fwhmguess           : (optional, float) fwhm parameter for DAOStarFinder
    bright              : (optional, int) number of found stars to display

    outputs
    ------------
    targx               : (int) x centroid of selected target star
    targy               : (int) y centroid ''
    fwhm                : (float) fwhm of target star
    '''
    # plot results of quick starfinder to grab rough pos of target
    firstim = fits.getdata(imagepath)
    firstSF = DAOStarFinder(thresh, fwhmguess, brightest=bright)
    table1 = firstSF.find_stars(firstim)

    norm = ImageNormalize(stretch=LogStretch())
    positions1 = np.transpose((table1['xcentroid'], table1['ycentroid']))
    apertures1 = CircularAperture(positions1, r=5)
    plt.figure()
    plt.imshow(firstim, origin='lower', norm=norm)
    apertures1.plot()
    plt.show()
    print(table1['xcentroid', 'ycentroid', 'roundness1'])

    # select index of target central star from table printed above
    targind = int(input('input the 0 indexed integer of your target from the table above: '))
    targx = int(table1['xcentroid'][targind])
    targy = int(table1['ycentroid'][targind])

    # get fwhm of targ from first img
    hist = firstim[targy]
    fwhm, count_at_fwhm, left, right = peak_widths(hist, [np.argmax(hist)])
    fwhm = fwhm[0]

    print('target star is at ', str(targx), ',', str(targy), ' at FWHM', fwhm)
    return targx, targy, fwhm


def makeEPSF(image, mit=10, verb=True, plot=True):
    '''
    makeEPSF
    ---------
    Creates EPSF object from source in image

    Example Usage: epsf = makeEPSF(reference)

    Modification History:
    Initialized by William Balmer 10/30/2020

    inputs
    ------------
    image       : (string) np.array image, centered on reference star psf
    mit         : (int) max iteration parameter for EPSFBuilder
    verbo       : (Bool) progress_bar parameter for EPSFBuilder
    plot        : (Bool) plot EPSF to check

    outputs
    ------------
    epsf        : (object) photutils.psf.EPSFModel object

    todo
    ------------
    - add ability to deal with rectangular image arrays, i.e. len(x)!=len(y)
    - assess necessity for oversampling parameter (not useful for Keck, can only be 1)
    '''
    # norm to 1
    image = image/np.nanmax(image)
    size = image.shape[0]
    # do photutils ePSF
    nddata = NDData(data=image)

    stars_tbl = Table()
    stars_tbl['x'] = [int(size/2)]
    stars_tbl['y'] = [int(size/2)]
    stars = extract_stars(nddata, stars_tbl, size=(size-10))

    epsf_builder = EPSFBuilder(oversampling=1, maxiters=mit, progress_bar=verb)
    epsf, fitted_stars = epsf_builder(stars)

    if plot is True:
        # Plot epsf to check
        norm = ImageNormalize(stretch=LogStretch())
        plt.figure(figsize=(5,5))
        plt.imshow(epsf.data, cmap='inferno', origin='lower', norm=norm,
                   interpolation='nearest')
        plt.colorbar()

    return epsf


def calcBinDist(phot_results):
    '''
    '''
    pos = phot_results['x_fit', 'y_fit']
    unc = phot_results['x_0_unc', 'y_0_unc']

    pixel_scale = phot_results['pixscale'][0]

    seps = []
    errs = []
    for i in range(len(pos)-1):
        x, y = pos[i]
        x1, y1 = pos[i+1]
        s = calcDistance(x, y, x1, y1)*pixel_scale*u.arcsec
        dx, dy = unc[i]
        dx1, dy1 = unc[i+1]
        ds = calcDistance(dx, dy, dx1, dy1)
        ds = (ds*pixel_scale*u.arcsec).to(u.mas)

        print(s.to(u.mas), '+/-', ds)
        seps.append(s.value)
        errs.append(ds.value)
    result = np.asarray([seps,errs])
    return result
