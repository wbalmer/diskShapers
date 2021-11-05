# imports
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# plotting
import seaborn as sns
plt.rcParams['font.family'] = 'monospace'   # Fonts
plt.rcParams['font.monospace'] = 'DejaVu Sans Mono'
sns.set_context("talk")

# read in visao image (path can be to whichever image you'd like)
impath = 'forwardModel/data/18May15/Cont/sliced_100.fits'
data = fits.getdata(impath)

# zoom in on center of psf
z = data[180:270, 180:270]
# create mesh grid for 3d surface visual
x, y = np.meshgrid(range(z.shape[0]), range(z.shape[1]))

# show height map in 3d
fig = plt.figure(figsize=(9, 4))
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(x, y, z)
# show height map in 2d
ax2 = fig.add_subplot(122)
im = ax2.imshow(z)
ax2.set(xticklabels=[], yticklabels=[])
plt.suptitle('Saturation of central star in VisAO images')
plt.colorbar(im)
plt.savefig('saturation_ex.png', dpi=300)

platescale=7.95
platescale_err=0.10

def pix2arc(x):
    return (x-225) * platescale


def arc2pix(x):
    return (x / platescale)+225

def negpix2arc(x):
    return -(x-225) * platescale


def negarc2pix(x):
    return -(x / platescale)+225

fig = plt.figure(figsize=(6,6))
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.visualization import ImageNormalize
from astropy.visualization.stretch import AsinhStretch
norm = ImageNormalize(stretch=AsinhStretch(), vmax=10000)
ax = fig.add_subplot()
im = ax.imshow(data, origin='lower', norm=norm, cmap='magma_r')
ax.annotate('', fontsize=5, xy=(382.5, 225), xycoords='data', xytext=(0, 50), textcoords='offset points', arrowprops=dict(arrowstyle="-|>", linewidth = 1., color = 'black'))
# ax.set_xlabel('X [pixels]')
ax.set_ylabel('Y [pixels]')

ax.tick_params(axis='x',
                which='both',
                top=False,
                bottom=False,
                labelbottom=False
                )

secax = ax.secondary_xaxis('bottom', functions=(negpix2arc, negarc2pix))
secax.set_xlabel(r'$\Delta$RA [mas]')
# secay.set_ylabel(r'$\Delta$Dec [mas]')

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cb1 = fig.colorbar(im, cax=cax)#, ticks=[-500, 0, 500])
plt.tight_layout()
fig.savefig('ghost_ex.pdf', dpi=300, bbox_inches="tight")