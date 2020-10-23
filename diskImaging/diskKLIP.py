# imports
import os
import glob
import pyklip
import numpy as np
# import gaplanets as gp
from astropy.io import fits
import matplotlib.pyplot as plt
import pyklip.instruments.MagAO as MagAO
import warnings
warnings.filterwarnings('ignore')


class diskDuster():
    '''
    A class to run KLIP to recover disk signa
    Initialized: William B. 10/21/2020
    '''

    def __init__(self, filepath, prefix, outputdir,
                 highpass=False, klipparams=[1, 10, [1, 5, 50]]):
        '''
        Initializes the class, injecting hole as specified and prepared for
        KLIP reduction via "run_KLIP" function
        Initialized: William B. 10/21/2020
        '''
        self.filelist = glob.glob(filepath+'\\*.fits')
        self.head = fits.getheader(self.filelist[0])
        self.highpass = highpass
        self.dataset = MagAO.MagAOData(self.filelist, highpass=self.highpass)

        self.outputdir = outputdir
        if os.path.exists(outputdir) is False:
            os.mkdir(outputdir)

        self.pfx = prefix
        self.numann, self.movm, self.KLlist = klipparams

    def run_KLIP(self):
        '''
        Runs a parallelized klip reduction using input parameters on the
        negative planet injection dataset.
        Initialized: William B. 10/21/2020
        '''
        pyklip.parallelized.klip_dataset(self.dataset,
                                         outputdir=self.outputdir,
                                         fileprefix=self.pfx,
                                         algo='klip', annuli=self.numann,
                                         subsections=1, movement=self.movm,
                                         numbasis=self.KLlist,
                                         calibrate_flux=False,
                                         mode="ADI", highpass=self.highpass,
                                         save_aligned=False,
                                         time_collapse='median')

        self.resultdir = self.outputdir+'\\'+self.pfx+'-KLmodes-all.fits'
        print('KLIP result is saved to: '+self.resultdir)
        return

    def get_result(self):
        '''
        Opens KLIP result fits file, takes image data for one KLmode,
        closes fits file.
        '''
        with fits.open(self.resultdir) as resulthdul:
            self.resultdatacube = resulthdul[1].data
        print('result dir is '+str(self.resultdir))
        return

    def showit(self, image, save='n', lims='n', cmap='magma',
               name='showit.png', vmin=None, vmax=None):
        '''
        Displays image data cleanly with attractive colorbar and limits
        Initialized: William B. 10/21/2020
        '''
        plt.figure(figsize=(7, 7))  # plots image
        plt.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
        plt.colorbar()  # plots colorbar
        if lims != 'n':
            plt.xlim(lims[0][0], lims[0][1])
            plt.ylim(lims[1][0], lims[1][1])
        if save == 'y':
            plt.savefig(name, dpi=150)
        return

    def show_result(self, KLi, save='n', name='diskimg.png',
                    vmin=None, vmax=None):
        '''
        Runs showit function on result of KLIP reduction
        Initialized: William B. 10/21/2020
        '''
        self.showit(self.resultdatacube[KLi], save='n', name=name,
                    vmin=vmin, vmax=vmax)
        return
