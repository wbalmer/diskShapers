# this is a really simple script to generate a 'mountain range'
# for one of the figures in the thesis

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

fwhms = [0.3, 0.9, 0.5, 0.4]
pos = [1, 5, 7, 3]
norms = [0.2, 0.7, 0.5, 0.3]
x_range = np.arange(-2, 9, 0.001)

gs = np.zeros(len(x_range))
for i in range(len(fwhms)):
    g = norm.pdf(x_range, pos[i], fwhms[i])
    g = g/np.max(g)
    g = g*norms[i]
    gs = gs + g

gs = gs/np.max(gs)

plt.plot(x_range, gs)
plt.axis('off')
plt.savefig('hiker_parable_mountain.png', dpi=300)
