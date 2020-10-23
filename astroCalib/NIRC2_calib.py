# NIRC 2 Trapezium calibration script
# this script produces results similar
# to table 3 in DeRosa 2020
# this data is already reduced,
# so we only really need to align it
# and then calculate the positions of the
# various binaries

import os
import glob
import numpy as np
from astropy.io import fits
from astroCalibReduction import crosscube
from photutils import DAOStarFinder


datadir = 'C:\\Users\\willi\\Dropbox (Amherst College)\\Research\\Follette-Lab\\Thesis-Data\William\\trapezium_data\\nirc2-trapezium'
# check data directory exists and has NIRC2 fits files in it
if os.path.exists(datadir) is False:
    raise FileNotFoundError("The data directory you entered does not exist!")
else:
    files = glob.glob(datadir+'\\*\\*.fit*')
    if files is []:
        raise FileNotFoundError("Empty data directory!")
    else:
        print('len of filelist is '+str(len(files))+' and first file is: ')
        print(files[0])
# sort data
names = []
bands = []
days = []
uniques = []

wavedict = {}
objdict = {}
datedict = {}

for im in files:
    hdr = fits.getheader(im)
    name = hdr['TARGNAME']
    passband = hdr['FILTER']
    date = hdr['DATE-OBS']
    day = date.split('T')[0]
    unique = day+'&'+passband+'&'+name
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

NIRC2_datasets = []
for unique in uniques:
    day, band, name = unique.split('&')
    thelist = []
    for file in files:
        if datedict[file] == day:
            if wavedict[file] == band:
                if objdict[file] == name:
                    thelist.append(file)
    NIRC2_datasets.append(thelist)

test = NIRC2_datasets[3]
im1data = fits.getdata(test[0])
im1head = fits.getheader(test[0])
testcube = np.zeros((len(test), im1data.shape[0], im1data.shape[1]))
for i in range(len(test)):
    im = fits.getdata(test[i])
    testcube[i] = im

shiftnodsubmed, shiftnodsubcube = crosscube(testcube,
                                            cenx=int(im1data.shape[1]/2),
                                            ceny=int(im1data.shape[0]/2),
                                            box=int(im1data.shape[1]/2)-10,
                                            returnmed='y',
                                            returncube='y')

fits.writeto('NIRCshiftmed.fits', shiftnodsubmed, im1head, overwrite=True)
fits.writeto('NIRCshiftcube.fits', shiftnodsubcube, im1head, overwrite=True)

# starfind
thresh = 20
fwhm = 3
sf = DAOStarFinder(thresh, fwhm)
table = sf.find_stars(shiftnodsubmed)
print(table)

import matplotlib.pyplot as plt
from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from photutils import CircularAperture
positions = np.transpose((table['xcentroid'], table['ycentroid']))
apertures = CircularAperture(positions, r=4.)
norm = ImageNormalize(stretch=LogStretch())
plt.imshow(shiftnodsubmed, cmap='inferno', origin='lower', norm=norm,
        interpolation='nearest')
apertures.plot(color='yellow', lw=1.5, alpha=0.5)
plt.savefig('test.png',dpi=100)
