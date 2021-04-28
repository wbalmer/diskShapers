import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling.functional_models import Moffat1D, Gaussian1D
import seaborn as sb
from scipy.special import gamma as G

def norm_gauss(sigma):
    return 1/np.sqrt(2 * np.pi * sigma**2)

def norm_moffat(width, power):
    return G(power) / (width * np.sqrt(np.pi) * G(power - 1/2))

# 2 = 2*np.sqrt(2np.ln(2*sigma))
# 1 = np.ln(2*sigma)
# np.e**1 = 2*sigma
sigma = np.e/2

x = np.arange(0,8,0.1)
moffat = Moffat1D(amplitude=1, x_0=0, gamma=2, alpha=2.5)
gauss = Gaussian1D(amplitude=sigma)
sb.set_context('paper', font_scale = 1.5)
sb.lineplot(x=x,y=moffat(x)*norm_moffat(2,2.5), label='Moffat, FWHM=2.2', linewidth=3, color='#870734')
sb.lineplot(x=x,y=gauss(x)*norm_gauss(sigma), label='Gaussian, FWHM=2.2', linewidth=3, color='#ef473a')
plt.xlabel('r')
plt.ylabel('Probability')
plt.savefig('gauss_vs_moffat.png', dpi=200)