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
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


class rmVisAOCosmics:
    def __init__(self, fname, namestr='noCosmics.fits'):
        self.fname = fname
        self.namestr = namestr
        self.cosmic_index = []
        self.nocosmic_index = []
        # specify namestring
        try:
            # read in SDI image
            data, imhead = fits.getdata(self.fname, header=True)
        except FileNotFoundError:
            fname = input('File not found, please input path to SDI fits datacube')
            data, imhead = fits.getdata(self.fname, header=True)

        idx = 0

        # setup axis
        fig, ax = plt.subplots()
        fig.subplots_adjust(bottom=0.15)
        norm = ImageNormalize(stretch=LogStretch())

        # display initial image
        im_h = ax.imshow(data[idx], origin='lower', norm=norm)

        # setup slider axis and Slider
        ax_depth = plt.axes([0.23, 0.02, 0.56, 0.04])
        slider_depth = Slider(ax_depth, 'depth', 0, data.shape[0]-1, valinit=idx)

        def update_depth(val):
            idx = int(round(slider_depth.val))
            im_h.set_data(data[idx])

        slider_depth.on_changed(update_depth)

        plt.show()


    # user inputs whether image has cosmic or not
    # plt.waitforbuttonpress(timeout=- 1) true if button, false if mouse

    # record index of image in either no cosmics or cosmics list

    # write no cosmics, cosmics fits cubes

    # cull rotoff cube

    # write new rotoff cube

    # print cosmic stats


if __name__ == '__main__':
    # run the thing as a script
    rmVisAOCosmics(sys.argv[1])
    print('done')

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
