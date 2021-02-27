import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.modeling import models, fitting
from regions import CirclePixelRegion, PixCoord
from photutils import centroid_2dg
from photutils import source_properties, detect_sources, deblend_sources
import math
import scipy
import scipy.ndimage.interpolation as sci

# code written by Sarah Betti c2020
# edited by William Balmer Feb 2021


def circlesym(datadir, filname, output, method='median', box=451, data=None, save=True, **kwargs):
    '''
    finds center of circular symmetry of median combinations of registered images, and shifts this center to the center of the image cube

    INPUTS:
    datadir:str -  directory to location of file
    filname:str - name of FITS file
    output:str - location and name to save outputted aligned FITS image
    method:str - (median/individual) method to determine circular symmetry.  Use either a median image of cube or find circular symmetry for each individual frame in the cube / optional
    **kwargs:
        center_only:boolean (T/F) - use center 1.3" to find circular symmetry
        mask:boolean (T/F) - create mask and use it to cover oversaturated pixels
        rmax:float - maximum radius to search for circular symmetry

    OUTPUT:
    Data_cent:ndarray - aligned image cube

    '''
    print('running circular symmetry on: ', filname)
    if data is not None:
        Data = data
        hdr = fits.getheader(datadir + filname)
    else:
        Data = fits.getdata(datadir + filname)
        hdr = fits.getheader(datadir + filname)

    if len(Data.shape) > 2:
        segm = detect_sources(Data[3,:,:], 10, npixels=35)
        cat = source_properties(Data[3,:,:], segm)
    else:
        segm = detect_sources(Data, 10, npixels=35)
        cat = source_properties(Data, segm)

    xind_cen, yind_cen = int(cat[0].xcentroid.value), int(cat[0].ycentroid.value)

    box_size = box
    radius_size = int(box_size // 2)

    if xind_cen < radius_size:
        xind_cen = radius_size+1
    if yind_cen < radius_size:
        yind_cen = radius_size+1

    print(xind_cen, yind_cen, radius_size)

    Data = Data[:, yind_cen-radius_size:yind_cen+radius_size, xind_cen-radius_size:xind_cen+radius_size]
    print('dimensions of Data:', Data.shape)

    if len(Data.shape) > 2:
        Datamed = np.nanmedian(Data[3:-1], axis=0)
    else:
        Datamed = Data

    if 'center_only' in kwargs:
        centerrad = kwargs['center_only']
        box_size = int(centerrad)
        radius_size = int(50)
        cenx, ceny = int(Datamed.shape[1] /2), int(Datamed.shape[0]/2)
        Data_circsym = Data[:, ceny-radius_size:ceny+radius_size, cenx-radius_size:cenx+radius_size]

        Datamed_circsym = Datamed[ceny-radius_size:ceny+radius_size, cenx-radius_size:cenx+radius_size]
    else:
        Data_circsym = Data
        Datamed_circsym = Datamed

    dimy = Datamed_circsym.shape[0]  ## y
    dimx = Datamed_circsym.shape[1]  ## x

    xr = np.arange(dimx / 2 + 1.0) - dimx / 4
    yr = np.arange(dimy / 2 + 1.0) - dimy / 4

    if 'rmax' in kwargs:
        lim = kwargs['rmax']
    else:
        lim = dimx/2. # xx limit

    Data_xc = np.array([])
    Data_yc = np.array([])
    if kwargs.get('mask'):
        mask_choice = 'True'
        if method == 'individual':
            for j in np.arange(len(Data_circsym)):
                print('constructing mask for oversaturated pixels for image', j+1, '/', len(Data_circsym))
                xs, ys = np.shape(Data_circsym[j,:,:])[1], np.shape(Data_circsym[j,:,:])[0]
                segm = detect_sources(Data_circsym[j,:,:], 12, npixels=10)
                segm_deblend = deblend_sources(Datamed_circsym, segm, npixels=10, nlevels=30, contrast=0.001)
                cat = source_properties(Data_circsym[j,:,:], segm)

                cenx, ceny = cat.xcentroid, cat.ycentroid
                radius = cat.equivalent_radius.value
                mask = makeMask(xs, ys, radius, cenx, ceny)
                print('calculating center of circular symmetry for median line image with mask')
                Dxc, Dyc = center_circlesym(Data_circsym[j,:,:], xr, yr, lim, mask)
                Data_xc = np.append(Data_xc, Dxc)
                Data_yc = np.append(Data_yc, Dyc)

        elif method == 'median':
            print('constructing mask for oversaturated pixels for image')
            xs, ys = np.shape(Datamed_circsym)[1], np.shape(Datamed_circsym)[0]
            segm = detect_sources(Datamed_circsym, 5, npixels=5)
            segm_deblend = segm #deblend_sources(Datamed_circsym, segm, npixels=10, nlevels=15, contrast=0.001)
            cat = source_properties(Datamed_circsym, segm_deblend)
            cenx, ceny = cat.xcentroid, cat.ycentroid
            radius = cat.equivalent_radius.value
            mask = makeMask(xs, ys, radius, cenx, ceny)

            print('calculating center of circular symmetry for median line image with mask')
            Dxc, Dyc = center_circlesym(Datamed_circsym, xr, yr, lim, mask)
            Data_xc = np.append(Data_xc, Dxc)
            Data_yc = np.append(Data_yc, Dyc)
        else:
            return ValueError('chose method: individual images or median')


    else:
        mask_choice='False'
        if method == 'individual':
            for j in np.arange(len(Data_circsym)):
                print('calculating center of circular symmetry for median line image')
                Dxc, Dyc = center_circlesym(Data_circsym[j,:,:], xr, yr, lim)
                Data_xc = np.append(Data_xc, Dxc)
                Data_yc = np.append(Data_yc, Dyc)
        elif method == 'median':
            Dxc, Dyc = center_circlesym(Datamed_circsym, xr, yr, lim)
            Data_xc = np.append(Data_xc, Dxc)
            Data_yc = np.append(Data_yc, Dyc)

        else:
            return ValueError('chose method: individual images or median')

    print()
    Data_xc_shift = ((dimx-1)/2.) - Data_xc
    Data_yc_shift = ((dimy-1)/2.) - Data_yc

    print('median center of circular symmetry is: ', np.median(Data_xc), np.median(Data_yc))
    print('median shift all images by: ', np.median(Data_xc_shift), np.median(Data_yc_shift))

    Data_cent = np.zeros(Data.shape)
    if len(Data.shape) > 2:
        nims = Data.shape[0]
        for i in np.arange(nims):
            if len(Data_yc_shift) == nims:
                temp = sci.shift(Data[i,:,:],(Data_yc_shift[i], Data_xc_shift[i]))
            else:
                temp = sci.shift(Data[i,:,:],(Data_yc_shift[0], Data_xc_shift[0]))
            Data_cent[i,:,:] = temp

    else:
        Data_cent = sci.shift(Data,(Data_yc_shift[0], Data_xc_shift[0]))

    print()
    print('----------- o -------------')
    print()

    if save is True:
        print(f'writing centered image cube: {output}')
        hdr.append(('COMMENT', f'aligned with method: {method} using mask:{mask_choice}'), end=True)
        fits.writeto(output, Data_cent, header=hdr, overwrite=True )
        return
    else:
        return Data_cent

def center_circlesym(im, xr, yr, rmax, mask=None):
    '''Finds the center of a star image assuming circular symmetry
    PARAMETERS:
    im   :  the input image
    xr   : vector of x-offsets from the center of the image
    yr   : vector of y-offsets from the center of the image
    rmax : maximum radius to consider
    mask : optional 1/0 mask, 1 specifies pixels to ignore.

    RETURNS:
    xc   : the center x position
    yc   : the center y position
    grid : grid of results (xr vs yr vs stddev)
    '''
    if mask is None:
        mask = np.zeros_like(im)


    dimx = im.shape[1]  ### x
    dimy = im.shape[0]  ### y

    x = (np.arange(dimx)+0.5)/dimx * 2.0 - 1.0
    y = ((np.arange(dimy)+0.5)/dimy * 2.0 - 1.0) *(dimy/dimx)

    xx, yy = np.meshgrid(x,y)

    xc = 2. * xr / dimx
    yc = 2. * yr / dimy


    XX = np.repeat(xx[np.newaxis, :,:], len(xc), axis=0)
    YY = np.repeat(yy[np.newaxis, :,:], len(xc), axis=0)
    XX = (XX.T- xc).T
    YY = (YY.T- yc).T

    XX2 = XX*XX
    YY2 = YY*YY
    print(len(im))
    grid = np.zeros((len(yr), len(xr)))
    print('Out of ' + str(len(xc)) + ' rows, this many have finished: ')
    for i in np.arange(len(xc)):
        print(str(i) + ', ', end='', flush=True)
        for j in np.arange(len(yc)):
            x2 = XX2[i,:,:]
            y2 = YY2[j,:,:]
            r2 = x2+y2
            rsq = np.sqrt(r2)
            r = 0.5*rsq*dimx
            for k in np.arange(rmax+1):
                vals = im[(r >=k) & (r < k+1) & (mask == 0)]

#                if (i == len(yc)/2) &(j == len(yc)/2):
#                    print(i, k)
#                    plt.figure()
#                    ind = (r >=k) & (r < k+1) & (mask == 0)
#                    im2 = np.copy(im)
#
#                    im2 = im2 * ind
#                    xreal = i + (dimx/2.) - (len(xr)/2.)
#                    yreal = j + (dimx/2.) - (len(yr)/2.)
#                    plt.plot(xreal, yreal, marker='*', color='yellow')
#                    plt.imshow(im2, cmap='magma')
#                    plt.title('r = ' + str(k))
##                    plt.axis('off')
#                    plt.savefig(f'/Users/sbetti/Desktop/radii/R{k}.png')
#                    plt.show()
                if len(vals) > 0:
                    sd = np.nanstd(vals)
                    if not np.isfinite(sd): sd = 0
                    med = np.nanmedian(vals)
                    div = (sd/abs(med))**2.
                    grid[j,i] += div
#            if (i == len(yc)/2) &(j == len(xc)/2):
#                plt.figure()
#                plt.imshow(grid*-1, cmap='magma')
#                plt.savefig(f'/Users/sbetti/Desktop/radii/grid.png')
#                plt.show()

    print()
    pos = np.where(grid == np.min(grid))
    print(-1*grid)
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3)
    ax1.imshow(im, cmap='magma',  origin='lower')
    ax2.imshow(mask, cmap='magma', origin='lower')
    ax3.imshow(-1*grid, cmap='magma', origin='lower')
    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')
    plt.show()
#    fits.writeto('grid.fits', -1*grid, overwrite=True)

#    xcc, ycc = gcntrd(-1 * grid, pos[1][0], pos[0][0], 0.5 * len(xr))
#    xcc = xcc[0][0]
#    ycc = ycc[0][0]
#    print('gcntd center:', xcc, ycc)
#    xcc, ycc = fit_gaussian(-1*grid, pos[1][0], pos[0][0])
#    print('fit_gaussian center:', xcc, ycc)
#    xcc, ycc = centroid_2dg(-1*grid)
#    print('centroid_2dg center:', xcc, ycc)
    xcc, ycc = centroid_2dg(-1*grid)
    print('centroid_1dg center:', xcc, ycc)

    ##calculating centroid for small grid
    xc = xcc + (dimx/2.) - (len(xr)/2.)
    yc = ycc + (dimy/2.) - (len(yr)/2.)

    return xc, yc


def fit_gaussian(image, xcen, ycen):
    """
    fits a gaussian function to the grid and finds the center of that function
    INPUTS:
        image - 2D array
        xcen, ycen - approximate center
    OUTPUTS:
        xcc, ycc - center of fitted gaussian
    """
    #setup, sets the most negative value (grid is always inverted) as 0
    #so that the grid is positive but still inverted
    min_guess = np.min(image)
    image = image-min_guess

    max_guess = image[xcen][ycen] #amplitude=1, x_0=0, y_0=0, gamma=1, alpha=1
    g_init = models.Moffat2(amplitude=max_guess, x_0=xcen, y_0=ycen, gamma=5, alpha=5)
    fit_g = fitting.LevMarLSQFitter()
    print(fit_g.fit_info['message'])
    y,x = np.mgrid[:image.shape[0],:image.shape[1]]
    z = image
    g = fit_g(g_init, x, y, z)

    xcc = g.x_mean
    ycc = g.y_mean
    return (xcc,ycc)




def gcntrd(img, x, y, fwhm, *args, **kwargs):
    """
    NAME: gcntrd
    PURPOSE: Compute the stellar centroid by Gaussian fits to marginal X,Y, sums
    DESCRIPTION: GCNTRD uses the DAOPHOT "FIND" centroid algorithm by fitting Gaussians to the marginal X,Y distributions.
                User can specify bad pixels (either by using the MAXGOOD keyword or setting them to NaN) to be ignored in
                the fit. Pixel values are weighted toward the center to avoid contamination by neighboring stars.
    INPUTS:
        img:    two dimensional image array
        x,y:    scalar or vector integers giving approximate stellar center
        fwhm:   floating scalar full width half max to compute centroid, centroid computed withing box of half width
                equal to 1.5 sigma = 0.637 * FWHM
    OUTPUTS:
        xcen:   computed x centroid position, same number of points as x
        ycen:   computed y centroid position
                (values for xcen and ycen will not be computed if the computed centroid falls outside of the box, or if
                there are too many bad pixels, or if the best-fit Gaussian has a negative height. If the centroid cannot be
                computed, then a message is displayed (unless 'silent' is set) and xcen and ycen are set to -1.
    OPTIONAL ARGS:
        silent:         gcntrd will not print error message if centroid cannot be found
        keepcenter:
    OPTIONAL KWARGS:
        maxgood - int:  only pixels with values leess than maxgood are used to determine centroid
    HISTORY:
        Created: 2004-06, W. Landsman  following algorithm used by P. Stetson in DAOPHOT2.
            Modified:   2008-03, W. Landsman to allow shifts of more than 1 pixel from initial guess
                        2009-01, W. Landsman to perform Gaussian convolution first before finding max pixel to smooth noise
        Py Trans: 2016-08 by Wyatt Mullen, wmullen1@stanford.edu
    """
    print('Starting gcntrd')
    sz_image = img.shape
    if len(sz_image) != 2: raise TypeError('Invalid dimensions - image array must be 2 dimensional')
    xsize = sz_image[1]
    ysize = sz_image[0]

    #if isinstance(x, int) or isinstance(x, float):
    npts = 1
    #else:
        #npts = len(x)

    maxbox = 13 #why this value?
    radius = max(0.637 * fwhm, 2.001)
    radsq = radius**2
    sigsq = (fwhm / 2.35482)**2
    nhalf = min(int(radius), (maxbox - 1) // 2)
    nbox = 2 * nhalf + 1 # of pix in side of convolution box

    ix = np.array([round(x)])
    iy = np.array([round(y)])

    g = np.zeros((nbox, nbox))
    row2 = (np.arange(nbox, dtype=float) - nhalf)**2
    g[nhalf,:] = row2
    for i in range(1, nhalf+1):
        temp = row2 + i**2
        g[nhalf-i,:] = temp
        g[nhalf+i,:] = temp

    mask = g <= radsq
    good = np.where(mask)
    pixels = len(good[0])
    g = math.e**(-0.5*g/sigsq)

    """ In fitting Gaussians to the marginal sums, pixels will arbitrarily be
    assigned weights ranging from unity at the corners of the box to
    NHALF^2 at the center (e.g. if NBOX = 5 or 7, the weights will be
                                     1   2   3   4   3   2   1
          1   2   3   2   1          2   4   6   8   6   4   2
          2   4   6   4   2          3   6   9  12   9   6   3
          3   6   9   6   3          4   8  12  16  12   8   4
          2   4   6   4   2          3   6   9  12   9   6   3
          1   2   3   2   1          2   4   6   8   6   4   2
                                     1   2   3   4   3   2   1
    respectively).  This is done to desensitize the derived parameters to
    possible neighboring, brighter stars. """

    x_wt = np.zeros((nbox, nbox))
    wt = nhalf - np.fabs(np.arange(nbox) - nhalf) + 1
    for i in range(0, nbox):
        x_wt[i,:] = wt
    y_wt = np.transpose(x_wt)
    pos = str(x) + ' ' + str(y)

    if 'keepcenter' not in args:
        c = g*mask
        sumc = np.sum(c)
        sumcsq = np.sum(c**2) - sumc**2 / pixels
        sumc = sumc / pixels
        c[good] = (c[good] - sumc) / sumcsq

    xcen, ycen = [], []
    for i in range(0,npts):
        if 'keepcenter' not in args:
            if (ix[i] < nhalf) or ((ix[i] + nhalf) > xsize - 1) or (iy[i] < nhalf) or ((iy[i] + nhalf) > ysize-1):
                if 'silent' not in args:
                    raise RuntimeError('Position ' + str(pos[i]) + ' is too near edge of image.')
            x1 = max(ix[i] - nbox, 0)
            x2 = min(ix[i] + nbox, xsize - 1)
            y1 = max(iy[i] - nbox, 0)
            y2 = min(iy[i] + nbox, ysize - 1)
            h = img[y1:y2 + 1, x1:x2 + 1]

            h = scipy.ndimage.convolve(h, c)
            h = h[nbox - nhalf: nbox + nhalf + 1, nbox - nhalf: nbox + nhalf + 1]
            d = img[iy[i] - nhalf: iy[i] + nhalf + 1, ix[i] - nhalf: ix[i] + nhalf + 1]

            if 'maxgood' in kwargs:
                ig = np.where(d < maxgood)
                mx = np.nanmax(d[ig])

            mx = np.nanmax(h) #max pix val in bigbox
            mx_pos = np.where(h == mx) #num pix w/max val
            idx = mx_pos[1] % nbox #x coord of max pix
            idy = mx_pos[0] % nbox #y coord of max pix
            if len(mx_pos[0]) > 1:
                idx = round(np.sum(idx) / len(idx))
                idy = round(np.sum(idy) / len(idy))
            else:
                idx = idx
                idy = idy

            xmax = ix[i] - nhalf + idx #x coord in original image array
            ymax = iy[i] - nhalf + idy #y coord in original image array
        else: #if keepcenter is specified
            xmax = ix[i]
            ymax = iy[i]

        ########################################################################################
        #check *new* center location for range
        #added by Hogg

        if (xmax < nhalf) or ((xmax + nhalf) > xsize-1) or (ymax < nhalf) or ((ymax + nhalf) > ysize-1):
            if 'silent' not in args:
                raise RuntimeError('Position ' + str(pos[i]) + ' is too near edge of image.')
        #########################################################################################

        d = img[int(ymax - nhalf): int(ymax + nhalf + 1), int(xmax - nhalf): int(xmax + nhalf + 1)]
        #extract subimage centered on max pixel, skipping debugging
        if 'maxgood' in kwargs:
            mask = (d < maxgood) #if we need values for this we should use np.amin(d, maxgood)
        else: # isinstance(img[0][0], float):
            mask = np.isfinite(d)
            #mask = np.zeros(d.shape)
        #else:
            #mask = np.ones(nbox, nbox)
        maskx = np.sum(mask, 1) > 0
        masky = np.sum(mask, 0) > 0

        #at least 3 points are needed in the partial sum to compute gaussian
        if np.sum(maskx) < 3 or np.sum(masky) < 3:
            if 'silent' not in args:
                raise RuntimeError('Position ' + str(pos[i]) + ' has insufficient good points')

        ywt = y_wt * mask
        xwt = x_wt * mask
        wt1 = wt * maskx
        wt2 = wt * masky
        """ Centroid computation:   The centroid computation was modified in Mar 2008 and now differs from DAOPHOT
        which multiplies the correction dx by 1/(1+abs(dx)). The DAOPHOT method is more robust (e.g. two different
        sources will not merge) especially in a package where the centroid will be subsequently be redetermined
        using PSF fitting.   However, it is less accurate, and introduces biases in the centroid histogram.
        The change here is the same made in the IRAF DAOFIND routine
        (see http://iraf.net/article.php?story=7211&query=daofind)"""

        #computation for x centroid
        sd = np.nansum(d * ywt, 0)
        sg = np.nansum(g * ywt, 0)
        sumg = np.nansum(wt1 * sg)
        sumgsq = np.nansum(wt1 * sg * sg)
        sumgd = np.nansum(wt1 * sg * sd)
        sumd = np.nansum(wt1 * sd)
        p = np.nansum(wt1)
        xvec = nhalf - np.arange(nbox)
        dgdx = sg * xvec
        sdgdxs = np.nansum(wt1 * dgdx ** 2)
        sdgdx = np.nansum(wt1 * dgdx)
        sddgdx = np.nansum(wt1 * sd * dgdx)
        sgdgdx = np.nansum(wt1 * sg * dgdx)

        #height of the best-fitting marginal Gaussian. If this is not positive then centroid will not make sense
        hx = (sumgd - sumg * sumd / p) / (sumgsq -sumg**2 / p)
        if hx <= 0:
            if 'silent' not in args:
                raise RuntimeError('Position ' + str(pos[i]) + ' cannot be fit by a Gaussian')

        skylvl = (sumd - hx * sumg) / p
        dx = (sgdgdx - (sddgdx - sdgdx * (hx * sumg + skylvl * p))) / (hx * sdgdxs / sigsq)
        if math.fabs(dx) >= nhalf:
            if 'silent' not in args:
                raise RuntimeError('Position ' + str(pos[i]) + ' is too far from initial guess')

        xcen.append(xmax + dx) #x centroid in original array

        # computation for y centroid
        sd = np.nansum(d * xwt, 1)
        sg = np.nansum(g * xwt, 1)
        sumg = np.nansum(wt2 * sg)
        sumgsq = np.nansum(wt2 * sg * sg)
        sumgd = np.nansum(wt2 * sg * sd)
        sumd = np.nansum(wt2 * sd)
        p = np.nansum(wt2)
        yvec = nhalf - np.arange(nbox)
        dgdy = sg * yvec
        sdgdys = np.nansum(wt1 * dgdy ** 2)
        sdgdy = np.nansum(wt1 * dgdy)
        sddgdy = np.nansum(wt1 * sd * dgdy)
        sgdgdy = np.nansum(wt1 * sg * dgdy)

        hy = (sumgd - sumg * sumd / p) / (sumgsq - sumg ** 2 / p)
        if hy <= 0:
            if 'silent' not in args:
                raise RuntimeError('Position ' + str(pos[i]) + ' cannot be fit by a Gaussian')

        skylvl = (sumd - hy * sumg) / p
        dy = (sgdgdy - (sddgdy - sdgdy * (hy * sumg+ skylvl * p))) / (hy * sdgdys / sigsq)
        if math.fabs(dy) >= nhalf:
            if 'silent' not in args:
                raise RuntimeError('Position ' + str(pos[i]) + ' is too far from initial guess')

        ycen.append(ymax + dy) # y centroid in original array

    print('Finishing gcntrd')
    return (xcen, ycen)







###################################
# make masks for reference images:

def makeMask(xs, ys, radius, cenx, ceny):
    shape = (xs, ys)
    fin_mask = np.zeros(shape)
    print(radius)
    for i in np.arange(len(cenx)):
        if radius[i] >5:
            radius[i] = 6
        center = PixCoord(cenx[i], ceny[i])
        circle = CirclePixelRegion(center, radius[i]-1)
        mask = circle.to_mask()
        newmask = mask.to_image(shape)
        fin_mask += newmask
    fin_mask[fin_mask >1] =1
    return fin_mask
