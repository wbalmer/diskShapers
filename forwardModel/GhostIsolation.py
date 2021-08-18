import glob
from astropy.io import fits
from astropy.modeling import models, fitting
import numpy as np
import matplotlib.pyplot as plt
from image_registration.fft_tools.shift import shift2d


def readInFilelist(path):
    imlist = glob.glob(path)
    print(path)
    firstim = fits.getdata(imlist[0])
    original = np.zeros((len(imlist), firstim.shape[0], firstim.shape[1]))
    for i in range(0, len(imlist)):
        original[i] = fits.getdata(imlist[i])
    print("read fits file into data")
    medianImage = np.nanmedian(original, axis=0)
    data = medianImage
    #fits.writeto("test_median.fits",data,overwrite=True)
    print("created median")
    return original, data


def findGhostSingleIm(gX,gY, data):
    """
    takes gX and gY, guesses for the location of the ghost image
    returns the actual coordinates of the approximate ghost center
    and the pixel value at that point
    """
    #xGhost = 383
    #yGhost = 217
    maxVal = 0
    tempX = 0
    tempY = 0

    #finds center based on brightest pixel around location of ghost
    for y in range(gY-15, gY+16):
        for x in range(gX-15,gX+16):
            if data[y,x] > maxVal:
                tempX = x
                tempY = y
                maxVal = data[y,x]
    print("Calculated ghost center is " + str(maxVal) + " at x=" + str(tempX) + " , y=" + str(tempY))
    print("returning")
    return (tempX, tempY, maxVal)


def makeSquare(xc,yc,data,filename="test_square.fits"):
    foundX, foundY, maxVal = findGhostSingleIm(xc,yc,data)
    maskedArray = np.zeros((31,31))
    for y in range(-15,16):
        for x in range(-15,16):
            maskedArray[y+15][x+15] = data[foundY+y][foundX+x]
    fits.writeto(filename, maskedArray, overwrite=True)
    return maskedArray, foundX, foundY


def moffat(xcen,ycen,amp,wid,power, fixed=False):
    if fixed:
        return models.Moffat2D(amplitude=amp, x_0=xcen, y_0=ycen, gamma=wid, alpha=power, fixed={"gamma":True, "alpha":True})
    else:
        return models.Moffat2D(amplitude=amp, x_0=xcen, y_0=ycen, gamma=wid, alpha=power)


def tie_xy(model):
    y_stddev = model.x_stddev
    return y_stddev


def gauss(xcen,ycen,amp,wid):
    # tied = {'y_stddev':tie_xy}
    return models.Gaussian2D(amplitude=amp, x_mean=xcen, y_mean=ycen,
                             x_stddev=wid, y_stddev=wid)#, tied=tied)


def makeMoffatGhost(array,xcen,ycen,amp,wid,power,fwhm):
    global best_vals2
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    y, x = np.mgrid[:31,:31]
    z = array
    #xcen,ycen,mv = findGhostSingleIm(gx,gy)
    mofmodel = moffat(xcen, ycen, amp, wid, power)
    gaumodel = gauss(xcen, xcen, amp, wid)
    print(mofmodel.x_0)
    lfit = fitting.LevMarLSQFitter()
    mofFit = lfit(mofmodel, x, y, z)
    gauFit = lfit(gaumodel, x, y, z)

    fwhm = round(mofFit.fwhm/1.07, 3)
    print('CUT MOF FWHM IS '+str(fwhm))
    print('CUT GAU FWHM IS '+str(gauFit.x_fwhm),str(gauFit.y_fwhm))
    core = (fwhm)/(2*np.sqrt(2**(1/mofFit.alpha)-1))
    smof = moffat(ycen, xcen, mofFit.amplitude, core, mofFit.alpha, fixed=True)
    smof = lfit(smof, x, y, z)
    print('FIXED MOFFAT FWHM IS '+str(smof.fwhm))
    fig = plt.figure(figsize=(9, 6))
    fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.94, wspace=0.5)
    ax1 = plt.subplot(2, 4, 1)
    im1 = ax1.imshow(z, origin='lower', interpolation='nearest', cmap='magma')
    ax1.set_title("VisAO Ghost")
    divider = make_axes_locatable(ax1)
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    cb1 = fig.colorbar(im1, cax=cax1)

    ax2 = plt.subplot(2, 4, 2)
    im2 = ax2.imshow(smof(x, y), origin='lower', interpolation='nearest', cmap='magma')
    ax2.set(yticklabels=[])
    ax2.set_title("Moffat")
    divider = make_axes_locatable(ax2)
    cax2 = divider.append_axes("right", size="5%", pad=0.05)
    cb2 = fig.colorbar(im2, cax=cax2)

    ax3 = plt.subplot(2, 4, 3)
    im3 = ax3.imshow(gauFit(x, y), origin='lower', interpolation='nearest', cmap='magma')
    ax3.set(yticklabels=[])
    ax3.set_title('Gaussian Fit')
    divider = make_axes_locatable(ax3)
    cax3 = divider.append_axes("right", size="5%", pad=0.05)
    cb3 = fig.colorbar(im3, cax=cax3)

    from gaussian import Gaussian
    gaumodelFWHM = Gaussian(31, fwhm).g*gauFit(x, y).max()

    ax3point5 = plt.subplot(2, 4, 4)
    im3point5 = ax3point5.imshow(gaumodelFWHM, origin='lower', interpolation='nearest', cmap='magma')
    ax3point5.set(yticklabels=[])
    ax3point5.set_title('Gaussian FWHM'+str(fwhm))
    divider = make_axes_locatable(ax3point5)
    cax3point5 = divider.append_axes("right", size="5%", pad=0.05)
    cax3point5 = fig.colorbar(im3point5, cax=cax3point5)
    ax3point5.set_label("Counts")

    ax4 = plt.subplot(2, 4, 6)
    ax4.set_title("Moffat Residual")
    im4 = ax4.imshow(z - smof(x, y), origin='lower', interpolation='nearest', cmap='magma')
    divider = make_axes_locatable(ax4)
    cax4 = divider.append_axes("right", size="5%", pad=0.05)
    cb4 = fig.colorbar(im4, cax=cax4)
    # ax4.set(yticklabels=[])


    ax5 = plt.subplot(2, 4, 7)
    ax5.set_title("Gaussian Fit Residual")
    im5 = ax5.imshow(z - gauFit(x, y), origin='lower', interpolation='nearest', cmap='magma')
    divider = make_axes_locatable(ax5)
    cax5 = divider.append_axes("right", size="5%", pad=0.05)
    cb5 = fig.colorbar(im5, cax=cax5)
    ax5.set(yticklabels=[])


    ax6 = plt.subplot(2, 4, 8)
    ax6.set_title('Gaussian FWHM'+str(fwhm)+" Residual")
    im6 = ax6.imshow(z - gaumodelFWHM, origin='lower', interpolation='nearest', cmap='magma')
    divider = make_axes_locatable(ax6)
    cax6 = divider.append_axes("right", size="5%", pad=0.05)
    cb6 = fig.colorbar(im6, cax=cax6)
    ax6.set(yticklabels=[])
    cb6.set_label("Counts")

    # fig.savefig('moffat_model.png')

    print("x center is " + str(mofFit.x_0))
    print("y center is " + str(mofFit.y_0))
    return (mofFit, mofFit.x_0, mofFit.y_0, gauFit, gaumodelFWHM, fwhm, smof)


def normalizeSquare(array):
    maxVal = 0
    for y in range(31):
        for x in range(31):
            if array[y][x] > maxVal:
                maxVal = array[y][x]
    for y in range(31):
        for x in range(31):
            array[y][x] = array[y][x] / maxVal


def shift(x,y,original):
    size = original.shape[0]
    shiftX, shiftY = x-15, y-15
    for z in range(size):
        shifted_image = shift2d(original[z],-shiftX,-shiftY)
        original[z] = shifted_image
        if(z%100 == 0):
            print("shift number " + str(z))


def ghostIsolation(filename, gx, gy, amp, wid, power, fwhm=None):

    original, data = readInFilelist(filename)
    maskedArray, foundX, foundY = makeSquare(gx,gy,data)

    # convert to maskedArray's coordinates
    foundXSmall = foundX+15-gx
    foundYSmall = foundY+15-gy

    model, xCenter, yCenter, modelgau, modelgauFWHM, newfwhm, smof = makeMoffatGhost(maskedArray,foundXSmall,foundYSmall,amp,wid,power, fwhm)

    y, x = np.mgrid[:31, :31]
    model = model(x, y)
    modelgau = modelgau(x, y)
    smof2 = smof(x, y)
    print(xCenter, yCenter)

    shift(xCenter,yCenter, original)

    medianImage = np.nanmedian(original, axis=0)
    data = medianImage
    print("created median after shifting")
    maskedArray2, a, b = makeSquare(foundX,foundY,data,"ghost.fits")
    #normalizeSquare(maskedArray2)
    #fits.writeto("test_final.fits", maskedArray2, overwrite=True)
    return maskedArray2, model, modelgau, modelgauFWHM, newfwhm, smof2
    '''
    with open("moffat.txt","w") as text_file:
        text_file.write(str(amp) + " " + str(ampErr) + " " + str(wid) + " " + str(widErr) + " " + str(area) + " " + str(areaErr))
        text_file.close()
    '''
