# imports
import os
import glob
import tqdm
import datetime
import numpy as np
import scipy.signal
from scipy.signal import peak_widths
from scipy import ndimage
from scipy.ndimage.interpolation import shift
from astropy.io import fits
import warnings
from photutils import CircularAperture, IRAFStarFinder
from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
import matplotlib.pyplot as plt
from astropy.nddata import NDData
from photutils.psf import extract_stars
from astropy.table import Table
from photutils import EPSFBuilder
from astropy.modeling.functional_models import Moffat2D, AiryDisk2D
from photutils.psf import IterativelySubtractedPSFPhotometry, FittableImageModel#
from photutils import MMMBackground
from photutils.psf import IntegratedGaussianPRF, DAOGroup, prepare_psf_model
from astropy.modeling.fitting import LevMarLSQFitter

import astropy.units as u


# functions
def calcDistance(x0, y0, x1, y1):
    '''
    calcDistance
    ------------
    Calculates the euclidian straight line distance between two points

    Example Usage: ds = calcDistance(x, y, x+dx, y+dy)

    Modification History:
    Written by William Balmer c.2020
    Docstring added 12/3/2020
    '''
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
        median_data = ndimage.median_filter(data_arr, size=(10, 10))
        # replace the bad pixels with median of the local 10 pixels
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
    Reproduced here 10/15/2020 by William Balmer

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

    if either center is a nan, returns an error

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

    # not sure if this will work with the crossimage except statement...
    if xcenf or ycenf is np.nan:
        raise ValueError

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
    except ValueError:  # if centroiding algorithm fails, use just peak pixel
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
    med_fullframe = np.nanmedian(fits_stack, axis=2)

    # need to account for different FOVs
    # full frame: 1024/512
    # strip: 1024/300
    # stamp: 400/200
    # substamp: 100/50
    med_strip = med_fullframe[212:512]
    med_stamp = med_fullframe[312:512, 0:400]
    med_substamp = med_fullframe[462:512, 0:100]

    return med_fullframe, med_strip, med_stamp, med_substamp


def darkSub(image, imgint, masterdark, masterdark_int):
    '''
    Takes in an image array, a masterdark array, the integration time
    for that masterdark, and returns the dark subtraction of that
    image array
    '''
    dark = imgint*(masterdark/masterdark_int)
    return image-dark


def runClioSubtraction(imlist, darks, masterdark_int, badpixelmaskspath):
    '''
    linearizes, dark subtracts, and bad pixel corrects clio data from list
    '''
    # need to account for different FOVs
    # full frame: 1024/512
    bpff = badpixelmaskspath+'\\badpix_fullframe.fit'
    # strip: 1024/300
    bpstrp = badpixelmaskspath+'\\badpix_strip.fit'
    # stamp: 400/200
    bpstmp = badpixelmaskspath+'\\badpix_stamp.fit'
    # substamp: 100/50
    bpsub = badpixelmaskspath+'\\badpix_substamp.fit'
    # bad pixel dict
    bps = {512: bpff, 300: bpstrp, 200: bpstmp, 50: bpsub}
    # darks dict
    dark_fullframe, dark_strip, dark_stamp, dark_substamp = darks
    darkdict = {512: dark_fullframe, 300: dark_strip, 200: dark_stamp, 50: dark_substamp}

    # begin subtraction loop
    for im in tqdm.tqdm(imlist):
        hdr = fits.getheader(im)
        imgint = hdr['INT']
        imgdata = fits.getdata(im)
        # check if cube:
        if len(imgdata.shape) > 2:
            newcube = np.zeros((imgdata.shape[0], imgdata.shape[1], imgdata.shape[2]))
            for i in range(imgdata.shape[0]):
                ylen = imgdata.shape[1]
                if ylen in bps:
                    badpixelmask = fits.getdata(bps[ylen])
                    masterdark = darkdict[ylen]
                else:
                    errstrng = 'Image in list does not have bad pixel mask: '+im
                    raise ValueError(errstrng)
                # decoadd
                imcoadd = np.divide(imgdata[i], int(hdr['COADDS']))
                # linearize
                imlin = linearizeClio2(imcoadd)
                # dark subtract
                imsub = darkSub(imlin, imgint, masterdark, masterdark_int)
                # bad pixel correct
                imbpcorr = badpixelcorrect(imsub, badpixelmask)
                # add to newcube
                newcube[i] = imbpcorr

            newpath = im.replace('.fit', '_LDBP.fit')

            newcoadded = np.nanmedian(newcube, axis=0)
            newhdr = hdr
            newhdr['COADDS'] = str(len(imgdata.shape))
            newhdr['HISTORY'] = '--------------------------'
            newhdr['COMMENT'] = '    Reduction History     '
            newhdr['COMMENT'] = '--------------------------'
            now = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            history = 'Linearized, Dark subtracted, and bad pixel corrected '+ now
            newhdr['COMMENT'] = history

            fits.writeto(newpath, newcoadded, newhdr, overwrite=True)
        else:
            ylen = imgdata.shape[0]
            if ylen in bps:
                badpixelmask = fits.getdata(bps[ylen])
                masterdark = darkdict[ylen]
            else:
                errstrng = 'Image in list does not have bad pixel mask: '+im
                raise ValueError(errstrng)
            # decoadd
            imcoadd = np.divide(imgdata, int(hdr['COADDS']))
            # linearize
            imlin = linearizeClio2(imcoadd)
            # dark subtract
            imsub = darkSub(imlin, imgint, masterdark, masterdark_int)
            # bad pixel correct
            imbpcorr = badpixelcorrect(imsub, badpixelmask)

            newpath = im.replace('.fit', '_LDBP.fit')

            hdr['HISTORY'] = '--------------------------'
            hdr['COMMENT'] = '    Reduction History     '
            hdr['COMMENT'] = '--------------------------'
            history = 'Linearized, Dark subtracted, and bad pixel corrected '+ datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            hdr['COMMENT'] = history

            fits.writeto(newpath, imbpcorr, hdr, overwrite=True)

    return ('Data is linearized, dark subtracted, and bad pixel corrected!')


def clioNodSub(reduced_data, savepath):
    '''
    performs nod subtraction on reduced clio data
    '''
    for dataset in tqdm.tqdm(reduced_data):
        # ignoring stamps, the data isn't great and they're a pain
        stamps = []

        for im in dataset:
            imgdata = fits.getdata(im)
            if imgdata.shape[1] < 1024:
                stamps.append(im)

        imlist = [im for im in dataset if im not in stamps]
        im1 = imlist[0]
        im1data = fits.getdata(im1)

        # gather info on the nods, make dictionaries to hold data
        nods = []
        nod_dict = {}
        nod_cubes = {}
        nod_meds = {}
        img_dict = {}

        for im in imlist:
            imgdata = fits.getdata(im)
            # this is to match a header to the final nod subtracted image
            # assumes that each image will sum to a unique value (ok assumption I think)
            img_dict[np.sum(imgdata)] = im
            # this is to determine the nod from the header
            hdr = fits.getheader(im)
            nod_dict[im] = hdr['BEAM']
            if hdr['BEAM'] not in nods:
                nods.append(hdr['BEAM'])

        for nod in nods:
            ims = [key for (key, value) in nod_dict.items() if value == nod]
            nod_cube = np.zeros((len(ims), im1data.shape[0], im1data.shape[1]))
            for i in range(len(ims)):
                im = ims[i]
                imdata = fits.getdata(im)
                # add image to cube
                nod_cube[i] = imdata
            # add nod_cube to dictionary for later
            nod_cubes[nod] = nod_cube
            nod_meds[nod] = np.nanmedian(nod_cube, axis=0)

        for nod in nods:
            nod_cube = nod_cubes[nod]
            othernods = [n for n in nods if n != nod]
            opposing_nod_meds = np.zeros((len(othernods), im1data.shape[0], im1data.shape[1]))
            for i in range(len(othernods)):
                opposing_nod_meds[i] = nod_meds[othernods[i]]
            opposing_med = np.nanmedian(opposing_nod_meds, axis=0)
            for im in nod_cube:
                # nod subtract
                nodsub = im-opposing_med
                # fill values 1sigma below median with median (gets rid of ugly negative blobs)
                nodsub[nodsub < (np.nanmedian(nodsub)-np.std(nodsub))] = np.nanmedian(nodsub)
                # find original image path
                orig_str = img_dict[np.sum(im)]
                # edit header
                hdr = fits.getheader(orig_str)
                morehistory = 'Nod subtracted '+ datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                hdr['COMMENT'] = morehistory
                # new filename
                new_str = orig_str.replace('.fit', '_nodsub.fit')
                new_str = savepath+new_str.split('\\')[-1]
                fits.writeto(new_str, nodsub, hdr, overwrite=True)


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


def sortData(datadir, instrument='CLIO2', filesufx='*.fit*', returntab=False):
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

    if returntab is True:
        return uniques
    elif instrument == 'NIRC2':
        return datasets
    else:
        return datasets, darks


def runClioReduction(datadir, badpixelpath, intTime=300):
    '''
    reduces (linearize, dark, bad pix, nod sub) all clio data in datadir
    '''
    # get data
    datasets, darks = sortData(datadir)
    # create master dark
    med_dark = createMasterDark(darks, intTime)
    # loop through datasets and correct them
    for dataset in datasets:
        runClioSubtraction(dataset, med_dark, intTime, badpixelpath)
    # get new subtracted data
    reduced_data, darks2 = sortData(datadir, filesufx='*_LDBP*.fit*')
    # run nodSubtraction
    savepath = datadir+'/reduced/'

    try:
        os.makedirs(savepath)
    except OSError:
        print('putting files in an existing folder.')

    clioNodSub(reduced_data, savepath)

    return print('Done reducing the files')


def starLocate(imagepath, thresh, fwhmguess, bright, stampsize=None,
               epsfstamp=None, plot=True, roundness=0.5, crit_sep=15,
               iterations=1, setfwhm=False):
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
    thresh              : (float) threshold parameter for IRAFStarFinder
    fwhmguess           : (float) fwhm parameter for IRAFStarFinder
    bright              : (int) number of found stars to display
    stampsize           : (optional, int, default=None) size of subregion of image to analyse
    plot                : (optional, bool, default=True) plot results or not
    roundness           : (optional, float, default=0.5) roundlo, roundhi parameter for IRAFStarFinder
    crit_sep            : (optional, float, default=15)
    iterations          : (optional, int, default=1) iterations for IterativelySubtractedPSFPhotometry
    setfwhm             : (optional, bool, default=False) use fwhmguess as fwhm

    outputs
    ------------
    phot_results        : (astropy.table object)
    '''
    # read in data
    data = fits.getdata(imagepath)
    # read in img header to get pixel scale
    head = fits.getheader(imagepath)
    date = head['DATE-OBS']
    date_time_obj = datetime.datetime.strptime(date, '%Y-%m-%d')
    pixturnover = datetime.datetime.strptime('2015-5-13', '%Y-%m-%d')
    if date_time_obj.date() > pixturnover.date():
        pixel_scale = 0.009971
    else:
        pixel_scale = float(head['PIXSCALE'])

    epsf = nircEPSF(imagepath, epsfsize=epsfstamp)

    print('Select your target system to fit positions to')
    targx, targy, fwhm = findFirst(imagepath, thresh=thresh,
                                   fwhmguess=fwhmguess, bright=bright)

    if stampsize is None:
        stampsize = int(input('input the size of stamp: '))
    elif stampsize is not int:
        stampsize = int(stampsize)

    x0 = int(targx-(stampsize/2))
    x1 = int(targx+(stampsize/2))
    y0 = int(targy-(stampsize/2))
    y1 = int(targy+(stampsize/2))
    stamp = data[y0:y1, x0:x1]

    daogroup = DAOGroup(crit_separation=crit_sep)
    mmm_bkg = MMMBackground()
    if setfwhm is True:
        fwhm = fwhmguess
    finder = IRAFStarFinder(thresh, fwhm, roundlo=-roundness, roundhi=roundness,
                           sigma_radius=3)
    fitter = LevMarLSQFitter()
    phot_obj = IterativelySubtractedPSFPhotometry(finder=finder,
                                                  group_maker=daogroup,
                                                  bkg_estimator=mmm_bkg,
                                                  psf_model=epsf,
                                                  fitter=fitter,
                                                  fitshape=int(stampsize/2),
                                                  niters=iterations,
                                                  aperture_radius=7)

    phot_results = phot_obj(stamp)
    norm = ImageNormalize(stretch=LogStretch())
    pos = phot_results['x_0', 'y_0']
    positions = np.transpose((pos['x_0'], pos['y_0']))
    apertures = CircularAperture(positions, r=fwhm)

    if plot is True:
        plt.figure(figsize=(9, 9))
        plt.subplot(1, 2, 1)
        plt.imshow(stamp, origin='lower', norm=norm)
        apertures.plot(color='red', lw=1.5, alpha=0.7)
        plt.colorbar(orientation='horizontal')

        plt.subplot(1, 2, 2)
        plt.imshow(phot_obj.get_residual_image(), cmap='viridis', norm=norm,
                   origin='lower', interpolation='nearest', aspect=1)
        apertures.plot(color='red', lw=1.5, alpha=0.7)
        plt.colorbar(orientation='horizontal')

    phot_results['pixscale'] = pixel_scale
    phot_results['date'] = date

    result = calcBinDist(phot_results)

    return result


def findFirst(imagepath, thresh=100, fwhmguess=5, bright=5):
    '''
    findFirst
    ---------
    Allows user to select target star in a fits image with a coarse
    IRAFStarFinder search, and returns the centroid and FWHM of that target

    Example Usage: targx, targy, fwhm = findFirst('path/to/img.fits')

    Modification History:
    Initialized by William Balmer 10/30/2020

    inputs
    ------------
    imagepath           : (string) path to fits image
    thresh              : (optional, float) threshold parameter for IRAFStarFinder
    fwhmguess           : (optional, float) fwhm parameter for IRAFStarFinder
    bright              : (optional, int) number of found stars to display

    outputs
    ------------
    targx               : (int) x centroid of selected target star
    targy               : (int) y centroid ''
    fwhm                : (float) fwhm of target star
    '''
    # plot results of quick starfinder to grab rough pos of target
    firstim = fits.getdata(imagepath)
    firstSF = IRAFStarFinder(thresh, fwhmguess, brightest=bright)
    table1 = firstSF.find_stars(firstim)

    norm = ImageNormalize(stretch=LogStretch())
    positions1 = np.transpose((table1['xcentroid'], table1['ycentroid']))
    apertures1 = CircularAperture(positions1, r=5)
    plt.figure()
    plt.imshow(firstim, origin='lower', norm=norm)
    apertures1.plot()
    plt.show()
    print(table1['xcentroid', 'ycentroid', 'roundness'])

    # select index of target central star from table printed above
    try:
        targind = int(input('input the 0 indexed integer of your target from the table above: '))
        a = table1['xcentroid'][targind]
    except IndexError:
        targind = int(input('remember to 0 index your target index: '))
    targx = int(table1['xcentroid'][targind])
    targy = int(table1['ycentroid'][targind])

    # get fwhm of targ from first img
    hist = firstim[targy]
    fwhm, count_at_fwhm, left, right = peak_widths(hist, [np.argmax(hist)])
    fwhm = fwhm[0]

    print('target star is at ', str(targx), ',', str(targy), ' at FWHM', fwhm)
    return targx, targy, fwhm


def nircEPSF(imgpath, epsfsize=None, thresh=100, fwhmguess=5, bright=5,
             plot=True):
    print('Choose a reference star image to create a reference PSF from')
    targx, targy, fwhm = findFirst(imgpath, thresh=thresh,
                                   fwhmguess=fwhmguess, bright=bright)

    if epsfsize is None:
        epsfsize = int(input('input the size of reference psf stamp: '))
    elif epsfsize is not int:
        epsfsize = int(epsfsize)

    data = fits.getdata(imgpath)
    x0 = int(targx-(epsfsize/2))
    x1 = int(targx+(epsfsize/2))
    y0 = int(targy-(epsfsize/2))
    y1 = int(targy+(epsfsize/2))
    ref_stamp = data[y0:y1, x0:x1]

    epsf = makeEPSF(ref_stamp, plot=plot)
    if epsf is None:
        print('EPSF fitting failed, using gaussian PRF')
        epsf = IntegratedGaussianPRF()

    return epsf


def makeEPSF(image, verb=False, plot=True):
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
    verb       : (Bool) Ask if psf is good
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
    # construct psf from image
    epsf = FittableImageModel(image)

    if plot is True:
        # Plot epsf to check
        norm = ImageNormalize(stretch=LogStretch())
        plt.figure(figsize=(5, 5))
        plt.imshow(epsf.data, cmap='inferno', origin='lower', norm=norm,
                   interpolation='nearest', vmin=0)
        plt.colorbar()
        plt.show()

    if verb is True:
        take = input('accept this epsf? (y/n): ')
        if take == 'n':
            epsf = None

    return epsf


def angle_between(p1, p2):
    deltaX = p1[0]-p2[0]
    deltaY = p1[1]-p2[1]
    return 180 - np.arctan2(deltaX, deltaY)/np.pi*180


def calcBinDist(phot_results):
    '''
    '''
    phot_results = phot_results[phot_results['x_0_unc'] < 1]
    pos = phot_results['x_fit', 'y_fit']
    unc = phot_results['x_0_unc', 'y_0_unc']

    pixel_scale = phot_results['pixscale'][0]

    seps = []
    errs = []
    PAs = []
    dPAs = []
    for i in range(len(pos)-1):
        x, y = pos[i]
        x1, y1 = pos[i+1]
        s = calcDistance(x, y, x1, y1)*pixel_scale*u.arcsec
        s = s.to(u.mas)
        phi = angle_between((x, y), (x1, y1))
        PA = phi

        dx, dy = unc[i]
        dx1, dy1 = unc[i+1]
        ds = calcDistance(dx, dy, dx1, dy1)
        ds = (ds*pixel_scale*u.arcsec).to(u.mas)

        dphi_max = angle_between((x-dx, y-dy), (x1+dx1, y1+dy1))
        dphi_min = angle_between((x+dx, y+dy), (x1-dx1, y1-dy1))

        dPA = dphi_max - dphi_min

        print(s, '+/-', ds)
        print(PA, '+/-', dPA)
        seps.append(s.value)
        errs.append(ds.value)
        PAs.append(PA)
        dPAs.append(dPA)
    result = np.asarray([seps[0], errs[0], PAs[0], dPAs[0]])
    return result
