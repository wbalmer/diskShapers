# imports
import os
import glob
import numpy as np
import scipy.signal
from scipy import ndimage
from scipy.ndimage.interpolation import shift
from astropy.io import fits


# functions
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
    max_x_shift = int(np.max(np.abs([xshifts[x] for x in xshifts.keys()])))
    max_y_shift = int(np.max(np.abs([yshifts[x] for x in yshifts.keys()])))

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
    '''
    thelist = []
    for file in filelist:
        if datedict[file] == day:
            if wavedict[file] == band:
                if objdict[file] == targ:
                    thelist.append(file)
    return thelist


def sortCliodata(datadir, filesufx='*.fit*'):
    '''
    Sorts through a data directory filled with Clio data, sorts it by epoch,
    target, and passband and returns lists of sorted data, as well as a list of
    the darks in all the folders as a separate list.
    '''
    names = []
    bands = []
    days = []
    uniques = []
    darks = []
    ao_on_images = []

    wavedict = {}
    objdict = {}
    datedict = {}

    images = glob.glob(datadir+'clio_astro/*/'+filesufx)
    for im in images:
        hdr = fits.getheader(im)
        name = hdr['CID']
        passband = hdr['PASSBAND']
        date = hdr['DATE']
        day = date.split('T')[0]
        unique = day+'&'+passband+'&'+name
        if passband == 'Blocked':  # if passband is 'blocked' then it is a dark
            darks.append(im)
        elif hdr['LOOP'] == 1:  # ensure AO is on for images used
            ao_on_images.append(im)
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

    clio_datasets = []
    for unique in uniques:
        day, band, name = unique.split('&')
        liste = filelister(ao_on_images, day, band, name, datedict=datedict,
                           wavedict=wavedict, objdict=objdict)
        clio_datasets.append(liste)

    return clio_datasets, darks


def runtheReduction(datadir, badpixelpath, intTime=300):
    '''
    '''
    # get data
    datasets, darks = sortCliodata(datadir)
    # get bad pixel map
    badpixelmap = fits.getdata(badpixelpath)
    # create master dark
    med_dark = createMasterDark(darks, intTime)
    # loop through datasets and correct them
    for dataset in datasets:
        runSubtraction(dataset, med_dark, intTime, badpixelmap)
    # get new subtracted data
    reduced_data, darks2 = sortCliodata(datadir, filesufx='*_LDBP*.fit*')
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
