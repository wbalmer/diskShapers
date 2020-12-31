#+
# NAME: visao_removecosmics
#
# PURPOSE:
#  allows user to reject images with cosmic rays and creates new image and rotoff cubes without the rejected images. also writes out an array
#  specifying which indices in original sequence contain cosmics.
#
# INPUTS:
# fname: string file name of image cube that will be used to identify cosmic rays without the .fits suffix
# namestr: final file will be named with suffix _no+namestr+cosmics.fits. can be blank, but should generally be 'ha' or 'cont'
#
# OUTPUTS:
#
# OUTPUT KEYWORDS:
#    none
#
# EXAMPLE:
#
#
# HISTORY:
#  Written 2015 by Kate Follette, kbf@stanford.edu
#  07-29-2016 KBF. Revied to read a more generic set of image cubes.
#     Added all keyword and genericized to find and appy to any circsym cube

import sys
from astropy.io import fits
from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize



# specify namestring
try:
    # read in SDI image
    imcube, imhead = fits.getdata(sys.argv[1], header=True)
except FileNotFoundError:
    fname = input('File not found, please input path to SDI fits datacube')
    imcube, imhead = fits.getdata(sys.argv[1], header=True)


# from github gist https://gist.github.com/smidm/745b4006a54cfe7343f4a50855547cc3

import matplotlib.pylab as plt
import numpy as np

idx = 0
norm = ImageNormalize(stretch=LogStretch())
fig = plt.figure()
image = plt.imshow(imcube[idx], origin='lower', norm=norm)
closed = False


def handle_close(evt):
    global closed
    closed = True


def waitforbuttonpress():
    while plt.waitforbuttonpress(0.2) is None:
        if closed:
            return False
    return True


fig.canvas.mpl_connect('close_event', handle_close)
while True:
    image.set_data(imcube[idx])
    plt.draw()
    if not waitforbuttonpress():
        idx += 1
        break


def rmVisAOCosmics(fname, namestr='noCosmics.fits'):

    # specify namestring
    try:
        # read in SDI image
        imcube, imhead = fits.getdata(fname, header=True)
    except FileNotFoundError:
        fname = input('File not found, please input path to SDI fits datacube')
        imcube, imhead = fits.getdata(fname, header=True)

    # for testing
    imcube = imcube[0:3]

    nocosmic_index = []
    idx = 0

    def press(event):
        if event.key == 'n':
            nocosmic_index.append(idx)
        if event.key == 'y':
            ax.set_data(im)
            plt.draw()

    norm = ImageNormalize(stretch=LogStretch())
    fig, ax = plt.subplots()
    ax.imshow(imcube[0], origin='lower', norm=norm)

    for i in range(len(imcube)):
        im = imcube[i]
        fig.canvas.mpl_connect('key_press_event', press)

    # write no cosmics, cosmics fits cubes
    nocosmics_cube = np.zeros((imcube.shape[0], imcube.shape[1], imcube.shape[2]))
    for i in nocosmic_index:
        nocosmics_cube = imcube[i]

    fits.writeto('test.fits', nocosmics_cube, overwrite=True)

    # cull rotoff cube

    # write new rotoff cube

    # print cosmic stats

    return


# old IDL CODE

# pro visao_removecosmics, fname, namestr=namestr, stp=stp
#
#   if not keyword_set(namestr) then begin
#     print, "pleaase specify a name string. For example Line or Cont"
#     stop
#   endif
#
#   ##read in image (should be SDI)
#   im=readfits(string(fname)+'.fits', imhead)
#   zdim=(size(im))[3]
#   print, zdim
#
#   ds9_imselect, im, index=idx
#
#   writefits, string(fname)+'_nocosmics.fits', im, imhead
#   writefits, namestr+'cosmics.fits', idx
#
#   ## cull rotoff cube as well
#   rotoffs=readfits('rotoff_preproc.fits', rothead)
#   rotoffs_noc=dblarr(n_elements(idx))
#
#   for i=0, n_elements(idx)-1 do begin
#     rotoffs_noc[i]=rotoffs[idx[i]]
#   endfor
#
#   writefits, 'rotoff_no'+namestr+'cosmics.fits', rotoffs_noc, rothead
#
#   print, 'rejected', zdim-n_elements(idx), 'cosmics'
#
#   if keyword_set(stp) then  stop
#
# end
