# imports
import os
import glob
import pyklip
import numpy as np
import pandas as pd
#import gaplanets as gp
from astropy.io import fits
import pyklip.fakes as fakes
import matplotlib.pyplot as plt
import pyklip.instruments.MagAO as MagAO
from astropy.visualization import ZScaleInterval

filelist = glob.glob('datum\\*.fits')
head = fits.getheader(filelist[0])
dataset = MagAO.MagAOData(filelist, highpass=False)
contrast = -10**(-0.5)
sep = 11
theta = 122
fwhm = head["0PCTFWHM"]

ghost = fits.getdata('ghost.fits')

inpflux = np.zeros((dataset.input.shape[0], ghost.shape[0], ghost.shape[1]))

for i in range(dataset.input.shape[0]):
    inpflux[i] = contrast*ghost

# print(fakes.inject_planet.__doc__)
fakes.inject_planet(dataset.input, dataset.centers, inpflux, dataset.wcs,
                    sep, theta, fwhm=fwhm)

outputdir = 'output'
pfx = 'injTest1'
numann = 6
movm = 3
KLlist = [1, 5, 50]


pyklip.parallelized.klip_dataset(dataset, outputdir=outputdir, fileprefix=pfx,
                                 algo='klip', annuli=numann, subsections=1,
                                 movement=movm, numbasis=KLlist,
                                 calibrate_flux=False, mode="ADI",
                                 highpass=True, save_aligned=False,
                                 time_collapse='median')

result = outputdir+'\\'+pfx+'-KLmodes-all.fits'
resultfits = fits.getdata(result)

def showit(image, save='n', lims='n', cmap='magma', savename='showit_img.png'):
    plt.figure(figsize=(10,10))
    plt.imshow(image, cmap=cmap, origin='lower') #plots image
    plt.colorbar() #plots colorbar
    if lims != 'n':
        plt.xlim(lims[0][0], lims[0][1])
        plt.ylim(lims[1][0], lims[1][1])
    if save == 'y':
        plt.savefig(savename, dpi=150)
    return

showit(resultfits[2], lims=[[200,250],[200,250]], save='y', savename='negplanetreduct.png')
