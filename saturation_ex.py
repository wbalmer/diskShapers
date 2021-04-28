# imports
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# read in visao image (path can be to whichever image you'd like)
impath = 'forwardModel/data/18May15/Cont/sliced_1.fits'
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
