"""
# Original author: Ian Crossfield (Python 2.7)

Handy Routines for Doppler Imaging and Maximum-Entropy.

To start up with data to fit of size N pixels, and M pixels per
observation, do:

import dime
dime.setup(N, M)
"""

######################################################

# 18-02-2020
# Emma Bubb - change to run on Python 3

######################################################

# EB update the imported toolkits
import numpy as np

class MaxEntropy:
    """A class for running Maximum Entropy inversions of Doppler images."""

    def __init__(self, alpha, nk, nobs):
        self.data = None
        self.weights = None
        self.Rmatrix = None
        self.alpha = alpha
        self.nk = nk
        self.nobs = nobs
        self.j = np.arange(self.nk * self.nobs)
        self.k = (np.floor(1.0*self.j / self.nk) * self.nk).astype(int)
        self.l = (np.ceil((self.j + 1.0) / self.nk) * self.nk - 1).astype(int)
        self.jfrac = (self.j % self.nk) / (self.nk - 1.0)

    def set_data(self, data, weights, Rmatrix):
        self.data = data
        self.weights = weights
        self.Rmatrix = Rmatrix

    # The SciPy way:
    def nentropy(self, x):
        """ Compute Normalized Entropy, Sum(y * ln(y)), where y_i = x_i/sum(x)"""
        # 2013-08-07 21:18 IJMC: Created
        norm_x = x / np.sum(x)
        entropy = -np.sum(norm_x * np.log(norm_x))
        return entropy

    def dnentropy_dx(self, x):
        xsum = 1.0*np.sum(x)
        norm_x = x / xsum
        nx = len(x)
        vec2 = np.log(norm_x) + 1.
        vec1s = (-np.tile(x, (nx,1)) + xsum*np.diag(np.ones(nx)))
        grads = -np.dot(vec1s, vec2)/xsum/xsum
        return grads

    def gnorm(self, unnorm_model):
        """ Compute the normalizing function"""
        return unnorm_model[self.k] + (unnorm_model[self.l] - unnorm_model[self.k]) * self.jfrac

    def normalize_model(self, unnorm_model):
        return unnorm_model / self.gnorm(unnorm_model)

    def dchisq_norm_dx(self, unnorm_model):
        normalizer = self.gnorm(unnorm_model)
        dif = (self.weights * (self.data - unnorm_model / normalizer))
        Rk = self.Rmatrix[:, self.k]
        dfdx = (normalizer * self.Rmatrix - unnorm_model * (Rk + self.jfrac * (self.Rmatrix[:, self.l] - Rk))) / normalizer / normalizer
        grads = -2.0 * (dif * dfdx).sum(1)
        return grads

    def getgrad_norm_sp(self, x):
        model = np.dot(x.ravel(), self.Rmatrix)
        ds = self.dnentropy_dx(x)
        dchi = self.dchisq_norm_dx(model) 
        return 0.5 * dchi - self.alpha * ds

    def entropy_map_norm_sp(self, map_pixels, retvals=False):
        if (map_pixels<=0).any():
            map_pixels[map_pixels<=0] = 1e-6 #EB: if any pixel values are negative, set to 1e-6 (vv small basically zero)
        entropy = self.nentropy(map_pixels) #EB: call function 'nentropy' to calculate the mormalised entropy
        model = np.dot(map_pixels.ravel(), self.Rmatrix)
        norm_model = self.normalize_model(model) #call function 'normalize_model' to normalise the model (basically model/normalising function)
        chisq = (self.weights * (norm_model - self.data)**2).sum() # EB: changed method of finding chi squared from calling function to calculating directly
        metric = 0.5 * chisq - self.alpha * entropy

        if retvals:
            return metric, chisq, entropy
        else:
            return metric

    def calc_Rmatrix(self):
        pass
