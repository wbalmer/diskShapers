#import os
#import sys
#import math
import glob
from astropy.io import fits
from astropy.modeling import models, fitting
import numpy as np
import matplotlib.pyplot as plt
#import glob
from scipy.optimize import curve_fit
from scipy.integrate import quad
#import pandas as pd
from numpy import sqrt
import image_registration as ir


def readInFilelist(path):
    imlist = glob.glob(path)
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


def moffat(xcen,ycen,amp,wid,power):
    return models.Moffat2D(amplitude=amp, x_0=xcen, y_0=ycen, gamma=wid, alpha=power)


def gauss(xcen,ycen,amp,wid):
    return models.Gaussian2D(amplitude=amp, x_mean=xcen, y_mean=ycen, x_stddev=wid, y_stddev=wid)


def makeMoffatGhost(array,xcen,ycen,amp,wid,power):
    global best_vals2
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    y, x = np.mgrid[:31,:31]
    z = array
    #xcen,ycen,mv = findGhostSingleIm(gx,gy)
    mofmodel = moffat(xcen, ycen, amp, wid, power)
    gaumodel = gauss(xcen, ycen, amp, wid)
    print(mofmodel.x_0)
    lfit = fitting.LevMarLSQFitter()
    mofFit = lfit(mofmodel, x, y, z)
    gauFit = lfit(gaumodel, x, y, z)

    fig = plt.figure(figsize=(12, 4))
    ax1 = plt.subplot(1, 5, 1)
    im1 = ax1.imshow(z, origin='lower', interpolation='nearest', cmap='magma')
    plt.title("Data")
    ax2 = plt.subplot(1, 5, 2)
    im2 = ax2.imshow(mofFit(x, y), origin='lower', interpolation='nearest', cmap='magma')
    ax2.set(yticklabels=[])
    ax2.set_title("Moffat")
    ax3 = plt.subplot(1, 5, 3)
    im3 = ax3.imshow(gauFit(x, y), origin='lower', interpolation='nearest', cmap='magma')
    ax3.set(yticklabels=[])
    plt.title('Gaussian')

    ax4 = plt.subplot(1, 5, 4)
    ax4.set_title("Moffat Residual")
    im4 = ax4.imshow(z - mofFit(x, y), origin='lower', interpolation='nearest', cmap='magma')
    divider = make_axes_locatable(ax4)
    cax4 = divider.append_axes("right", size="5%", pad=0.05)
    cb4 = fig.colorbar(im4, cax=cax4)
    ax4.set(yticklabels=[])
    cb4.set_label("Counts")

    ax5 = plt.subplot(1, 5, 5)
    ax5.set_title("Gaussian Residual")
    im5 = ax5.imshow(z - gauFit(x, y), origin='lower', interpolation='nearest', cmap='magma')
    divider = make_axes_locatable(ax5)
    cax5 = divider.append_axes("right", size="5%", pad=0.05)
    cb5 = fig.colorbar(im5, cax=cax5)
    ax5.set(yticklabels=[])
    cb5.set_label("Counts")

    plt.savefig('moffat_model.png')

    print("x center is " + str(mofFit.x_0))
    print("y center is " + str(mofFit.y_0))
    return mofFit, mofFit.x_0, mofFit.y_0


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
        shifted_image = ir.fft_tools.shift2d(original[z],-shiftX,-shiftY)
        original[z] = shifted_image
        if(z%100 == 0):
            print("shift number " + str(z))


def ghostIsolation(filename, gx, gy, amp, wid, power):

    original, data = readInFilelist(filename)
    maskedArray, foundX, foundY = makeSquare(gx,gy,data)

    #convert to maskedArray's coordinates
    foundXSmall = foundX+15-gx
    foundYSmall = foundY+15-gy

    model, xCenter, yCenter = makeMoffatGhost(maskedArray,foundXSmall,foundYSmall,amp,wid,power)
    y, x = np.mgrid[:31, :31]
    model = model(x, y)
    print(xCenter, yCenter)

    shift(xCenter,yCenter, original)

    medianImage = np.nanmedian(original, axis=0)
    data = medianImage
    print("created median after shifting")
    maskedArray2, a, b = makeSquare(foundX,foundY,data,"ghost.fits")
    #normalizeSquare(maskedArray2)
    #fits.writeto("test_final.fits", maskedArray2, overwrite=True)
    return maskedArray2, model
    '''
    with open("moffat.txt","w") as text_file:
        text_file.write(str(amp) + " " + str(ampErr) + " " + str(wid) + " " + str(widErr) + " " + str(area) + " " + str(areaErr))
        text_file.close()
    '''
