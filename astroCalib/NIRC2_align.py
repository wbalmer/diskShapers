import os
import glob
import numpy as np
from astropy.io import fits
from photutils import DAOStarFinder
from photutils import CircularAperture
import matplotlib.pyplot as plt
from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize

from calibFuncs import sortData, crosscube


# input and output paths
datadir = 'C:\\Users\\willi\\Dropbox (Amherst College)\\Research\\Follette-Lab\\Thesis-Data\William\\trapezium_data\\nirc2-trapezium'
outdir = datadir+'\\stacked'

# check directories exists
if os.path.exists(datadir) is False:
    raise FileNotFoundError("The data directory you entered does not exist!")
if os.path.exists(outdir) is False:
    os.makedirs(outdir)

# sort the data in datadir
NIRC2_datasets = sortData(datadir, instrument='NIRC2', filesufx='*.fit*')

# cross correlate and shift, save resultant fits
for dataset in NIRC2_datasets:
    im1data = fits.getdata(dataset[0])
    im1head = fits.getheader(dataset[0])
    name = im1head['OBJECT']+'_'+im1head['DATE-OBS']+'_'
    cubesize = 0
    for image in dataset:
        im = fits.getdata(image)
        # count each frame so that if individual files are cubes no error
        if len(im.shape) > 2:
            cubesize += im.shape[0]
        else:
            cubesize += 1
    datasetcube = np.zeros((cubesize, im1data.shape[0], im1data.shape[1]))
    i = 0
    for image in range(len(dataset)):
        frame = fits.getdata(dataset[image])
        if len(frame.shape) > 2:
            j = i + frame.shape[0]
            datasetcube[i:j] = frame
        else:
            datasetcube[i] = frame
        i += 1

    shiftnodsubmed, shiftnodsubcube = crosscube(datasetcube,
                                                cenx=int(im1data.shape[1]/2),
                                                ceny=int(im1data.shape[0]/2),
                                                box=int(im1data.shape[1]/2)-10,
                                                returnmed='y',
                                                returncube='y')

    fits.writeto(outdir+'\\'+name+'NIRCshiftmed.fits', shiftnodsubmed,
                 im1head, overwrite=True)

    # run starfinder algorithm, make pictures
    thresh = 100
    fwhm = 3
    num = 10
    sf = DAOStarFinder(thresh, fwhm, brightest=num)
    table = sf.find_stars(shiftnodsubmed)
    print(table)
    # create apertures
    positions = np.transpose((table['xcentroid'], table['ycentroid']))
    apertures = CircularAperture(positions, r=3.)

    # Plot image with found stars circled
    norm = ImageNormalize(stretch=LogStretch())
    plt.figure()
    plt.imshow(shiftnodsubmed, cmap='inferno', origin='lower', norm=norm,
               interpolation='nearest')
    apertures.plot(color='red', lw=1.5, alpha=0.7)
    try:
        os.makedirs('.\\NIRC2starfind')
    except FileExistsError:
        pass
    plt.savefig('.\\NIRC2starfind\\'+name+'srcs.png', dpi=150)
