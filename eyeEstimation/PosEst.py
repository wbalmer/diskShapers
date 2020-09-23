import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shift_methods import centroid
from astropy.io import fits

files = glob.glob('quick_reduction/*')

dates = []
days = []
for file in files:
    quick, day = file.split('\\')
    day = day.split('_')[0]
    date = [i for i in files if day in i]
    if date not in dates:
        dates.append(date)
        days.append(day)

MagAO_pixscale = 0.0078513

# testing


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)


def astrometry(filepath, img, xguess, yguess):
    '''
    snags cen x and y for given fits im
    '''
    read = fits.getdata(filepath)
    plt.figure()
    plt.imshow(read[img])
    plt.xlim(210, 230)
    plt.ylim(210, 230)
    plt.savefig(filepath.replace('.fits', 'est.png'))
    xcen, ycen = centroid(read[img], xguess, yguess, 3)
    return(xcen, ycen)


def distfromcen(filepath, img, x, y):
    '''
    calcs distance from center of image
    '''
    read = fits.getdata(filepath)
    image = read[img]
    xlen, ylen = image.shape
    dx = x-(xlen/2)
    dy = y-(ylen/2)
    rho, phi = cart2pol(dx, dy)
    return(dx, dy, rho, phi)


def getEst(filepath, img, xguess, yguess):
    x, y = astrometry(filepath, img, xguess, yguess)
    rho, phi = distfromcen(filepath, img, x, y)[2:]
    PA = 360-(90-np.rad2deg(phi))
    sep = (MagAO_pixscale*rho)*1000
    return(sep, PA)


def astSuite(dates, KLmodes, xguess, yguess):
    array = pd.DataFrame(columns=['date', 'sep', 'PA'])
    array['date'] = days
    for i in range(len(dates)):
        date = dates[i]
        KLmode = KLmodes[i]
        sep, PA = getEst(date[1], KLmode, xguess, yguess)
        array['sep'][i] = sep
        array['PA'][i] = PA
    return(array)

# rho, phi = cart2pol(4,4)
# PA = 360-(90-np.rad2deg(phi))
# sep = (MagAO_pixscale*rho)*1000
# np.rad2deg(phi)
# sep

astSuite(dates, [8,7,7,5,5], 220, 220)
