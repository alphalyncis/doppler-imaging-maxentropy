"""
# Original author: Ian Crossfield (Python 2.7)

Handy Routines for Doppler Imaging and Maximum-Entropy.
"""

######################################################
# 06-2024 Xueqing Chen: change to class structure
# 02-2020 Emma Bubb: change to run on Python 3
######################################################

import numpy as np
import matplotlib.pyplot as plt

import scipy.constants as const
import scipy.optimize as opt
from scipy import interpolate
from scipy.signal import savgol_filter
import starry

import ELL_map_class as ELL_map
import modelfitting as mf

class DopplerImaging():
    """A class for Doppler Imaging routines.
    Attributes
    ----------
    obskerns_norm : 3darray, shape=(nobs, nchip, nk)
        The observed line profiles (kerns).

    intrinsic_profiles : 2darray, shape=(nchip, nk)
        The model line profiles (modekerns).

    dbeta: float
        d_lam/lam_ref of the wavelength range that the line profile sits on.

    nk: int
        Size of line profile kernel.

    nobs: int
        Number of observations.

    phases: 1darray, shape=(nobs)
        Phases corresponding to the obs timesteps. In radian (0~2*pi).

    inc: float
        Inclination of star in degrees (common definition, 90 is equator-on)
    """

    def __init__(self, obskerns_norm, intrinsic_profiles, kwargs_IC14,
                nk, nobs, dbeta):
        # settings
        self.alpha = kwargs_IC14['alpha']
        self.nk = nk
        self.nobs = nobs
        self.dbeta = dbeta
        self.dv = -self.dbeta * np.arange(np.floor(-self.nk/2.+.5), np.floor(self.nk/2.+.5)) * const.c / 1e3 # km/s
        self.phases = kwargs_IC14['phases']
        self.inc = kwargs_IC14['inc']
        self.inc_ = (90 - self.inc) * np.pi / 180 # IC14 defined 0 <-> equator-on, pi/2 <-> face-on
        self.vsini = kwargs_IC14['vsini']
        self.lld = kwargs_IC14['LLD']
        self.iseqarea = kwargs_IC14['eqarea']
        self.nlat = kwargs_IC14['nlat']
        self.nlon = kwargs_IC14['nlon']
        self.dime = MaxEntropy(self.alpha, self.nk, self.nobs)

        self.obskerns_norm = obskerns_norm
        self.intrinsic_profiles = intrinsic_profiles
        self.nchip = self.obskerns_norm.shape[1]

        ### Set up the intrinsic profile function
        mean_profile = np.median(self.intrinsic_profiles, axis=0) # can safely take means over chips now
        modIP = 1. - np.concatenate((np.zeros(300), mean_profile, np.zeros(300)))
        modDV = - np.arange(np.floor(-modIP.size/2.+.5), np.floor(modIP.size/2.+.5)) * self.dbeta * const.c / 1e3
        self.modelfunc = interpolate.UnivariateSpline(modDV[::-1], modIP[::-1], k=1., s=0.) # function that returns the intrinsic profile

        ### Set up observed data and weights
        self.observed_1d = np.median(self.obskerns_norm, axis=1).ravel() # mean over chips and ravel to 1d

        # calc error for each obs as the weights
        smoothed = savgol_filter(self.obskerns_norm, 31, 3)
        resid = self.obskerns_norm - smoothed
        err_pix = np.array([np.abs(resid[:,:,pix] - np.median(resid, axis=2)) for pix in range(self.nk)]) # error of each pixel in LP by MAD, shape=(nk, nobs, nchips)
        err_LP = 1.4826 * np.median(err_pix, axis=0) # error of each LP, shape=(nobs, nchips)
        err_each_obs = err_LP.mean(axis=1) # error of each obs, shape=(nobs)
        err_observed_1d = np.tile(err_each_obs[:, np.newaxis], (1,self.nk)).ravel() # look like a step function over different times

        # mask out non-surface velocity space with weight=0
        width = int(self.vsini/1e3/np.abs(np.diff(self.dv).mean())) + 15 # vsini edge plus uncert=3
        central_indices = np.arange(self.nobs) * self.nk + int(self.nk/2)
        mask = np.zeros_like(self.observed_1d, dtype=bool)
        for central_idx in central_indices:
            mask[central_idx - width:central_idx + width + 1] = True
        self.weights = (mask==True).astype(float) / err_observed_1d **2

        ### Set up Map object
        if self.iseqarea:
            self.dmap = ELL_map.Map(nlat=self.nlat, nlon=self.nlon, type='eqarea', inc=self.inc_, verbose=True)
        else:
            self.dmap = ELL_map.Map(nlat=self.nlat, nlon=self.nlon, inc=self.inc_) #ELL_map.map returns a class object

        self.ncell = self.dmap.ncell
        self.uncovered = list(range(self.ncell))
        self.flatguess = 100 * np.ones(self.ncell)
        self.bounds = [(1e-6, 300)] * self.ncell

        ### Set up the R matrix
        self.Rmatrix = np.zeros((self.ncell, self.nobs*self.dv.size), dtype=np.float32)
        self.compute_Rmatrix()
        self.flatmodel = self.dime.normalize_model(np.dot(self.flatguess, self.Rmatrix))

        ### Set up attributes for results
        self.bestparams = None
        self.bestparamgrid = None
        self.metric = None
        self.chisq = None
        self.entropy = None
        self.fitres = None
    
    def compute_Rmatrix(self):
        """
        Compute the R matrix for the inversion.
        """
        for kk, rot in enumerate(self.phases):
            speccube = np.zeros((self.ncell, self.dv.size), dtype=np.float32) 
            if self.iseqarea:
                this_map = ELL_map.Map(nlat=self.nlat, nlon=self.nlon, type='eqarea', inc=self.inc_, deltaphi=-rot)
            else:
                this_map = ELL_map.Map(nlat=self.nlat, nlon=self.nlon, inc=self.inc_, deltaphi=-rot)
            this_doppler = 1. + self.vsini*this_map.visible_rvcorners.mean(1)/const.c/np.cos(self.inc_) # mean rv of each cell in m/s
            good = (this_map.projected_area>0) * np.isfinite(this_doppler)    
            for ii in good.nonzero()[0]:
                if ii in self.uncovered:
                    self.uncovered.remove(ii) # remove cells that are visible at this rot
                speccube[ii,:] = self.modelfunc(self.dv + (this_doppler[ii]-1)*const.c/1000.)
            limbdarkening = (1. - self.lld) + self.lld * this_map.mu
            Rblock = speccube * ((limbdarkening*this_map.projected_area).reshape(this_map.ncell, 1)*np.pi/this_map.projected_area.sum())
            self.Rmatrix[:, self.dv.size*kk:self.dv.size*(kk+1)] = Rblock

    def solve(self, create_obs_from_diff=True, solver='scipy', maxiter=1e4, ftol=1e-5):
        '''Solve the Doppler imaging problem using the Maximum Entropy method.'''
        
        # create diff+flat profile
        uniform_profiles = np.zeros((self.nchip, self.nk))
        for c in range(self.nchip):
            uniform_profiles[c] = self.obskerns_norm[:,c].mean(axis=0) # time-avged LP for each chip
        mean_dev = np.median(np.array(
            [self.obskerns_norm[:,c]-uniform_profiles[c] for c in range(self.nchip)]
        ), axis=0) # mean over chips
        flatmodel_2d = np.reshape(self.flatmodel, (self.nobs, self.nk))
        new_observation_2d = mean_dev + flatmodel_2d
        new_observation_1d = new_observation_2d.ravel()

        if create_obs_from_diff:
            self.observed_1d = new_observation_1d

        # Scale the observations to match the model's linear trend and offset:
        linfunc = lambda x, c0, c1: x * c0 + c1
        coeff, cov = opt.curve_fit(linfunc, xdata=self.observed_1d, ydata=self.flatmodel, 
                                    sigma=self.weights, absolute_sigma=True, p0=[1, 0])
        self.observed_1d = linfunc(self.observed_1d, *coeff)

        ### Solve!
        self.dime.set_data(self.observed_1d, self.weights, self.Rmatrix)
        
        if solver == 'ic14':
            ftol = 0.01
            bfit = mf.gfit(self.dime.entropy_map_norm_sp, self.flatguess, fprime=self.dime.getgrad_norm_sp, 
                        args=(), ftol=ftol, disp=1, maxiter=maxiter, bounds=self.bounds)
            self.bestparams = bfit[0]
            self.fitres = bfit

        elif solver == 'scipy':
            res = opt.minimize(self.dime.entropy_map_norm_sp, self.flatguess, 
                              method='L-BFGS-B', jac=self.dime.getgrad_norm_sp, options={'ftol':ftol})
            self.bestparams = res.x
            self.fitres = res

        self.metric, self.chisq, self.entropy = self.dime.entropy_map_norm_sp(self.bestparams, retvals=True)
        print(f"metric: {self.metric:.2f}, chisq: {self.chisq:.2f}, entropy: {self.entropy:.2f}")

        self.bestparams[self.uncovered] = np.nan # set completely uncovered cells to nan
        bestparams2d = self.reshape_map_to_grid() # update self.bestparamgrid
        self.bestparamgrid = np.roll(np.flip(bestparams2d, axis=1), 
                                       int(0.5*bestparams2d.shape[1]), axis=1)

    def reshape_map_to_grid(self, plot_unstretched_map=False):
        '''Reshape a 1D map to a 2D grid.
        Returns 2d map grid.'''
        if self.iseqarea:
            # reshape into list
            start = 0
            bestparamlist = []
            for m in range(self.dmap.nlat):
                bestparamlist.append(self.bestparams[start:start+self.dmap.nlon[m]])
                start = start + self.dmap.nlon[m]
            # interp into rectangular array
            max_length = max([len(x) for x in bestparamlist])
            stretched_arrays = []
            for array in bestparamlist:
                x_old = np.arange(len(array))
                x_new = np.linspace(0, len(array) - 1, max_length)
                y_new = np.interp(x_new, x_old, array)
                stretched_arrays.append(y_new)

            bestparams2d = np.vstack(stretched_arrays)

            if plot_unstretched_map:
                # pad into rectangular array
                padded_arrays = []
                for array in bestparamlist:
                    left_pad = int((max_length - len(array)) / 2)
                    right_pad = max_length - len(array) - left_pad
                    padded_array = np.pad(array, (left_pad, right_pad), 'constant')
                    padded_arrays.append(padded_array)
                    array_2d = np.vstack(padded_arrays)
                    plt.imshow(array_2d, cmap='plasma')

        else:
            bestparams2d = np.reshape(self.bestparams, (-1, self.nlon))

        return bestparams2d

    def plot_IC14_map(self, colorbar=False, clevel=5, sigma=1, vmax=None, vmin=None, cmap=plt.cm.plasma, annotate=False):
        '''Plot doppler map from an array.'''
        cmap = plt.cm.plasma.copy()
        cmap.set_bad('gray', 1)
        fig = plt.figure(figsize=(5,3))
        ax = fig.add_subplot(111, projection='mollweide')
        lon = np.linspace(-np.pi, np.pi, self.bestparamgrid.shape[1])
        lat = np.linspace(-np.pi/2., np.pi/2., self.bestparamgrid.shape[0])
        Lon, Lat = np.meshgrid(lon,lat)
        if vmax is None:
            im = ax.pcolormesh(Lon, Lat, self.bestparamgrid, cmap=cmap, shading='gouraud')
        else:
            im = ax.pcolormesh(Lon, Lat, self.bestparamgrid, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
        #contour = ax.contour(Lon, Lat, gaussian_filter(bestparamgrid, sigma), clevel, colors='white', linewidths=0.5)
        if colorbar:
            fig.colorbar(im, fraction=0.065, pad=0.2, orientation="horizontal", label="%")
        yticks = np.linspace(-np.pi/2, np.pi/2, 7)[1:-1]
        xticks = np.linspace(-np.pi, np.pi, 13)[1:-1]
        ax.set_yticks(yticks, labels=[f'{deg:.0f}˚' for deg in yticks*180/np.pi], fontsize=7, alpha=0.5)
        ax.set_xticks(xticks, labels=[f'{deg:.0f}˚' for deg in xticks*180/np.pi], fontsize=7, alpha=0.5)
        ax.grid('major', color='k', linewidth=0.25)
        for item in ax.spines.values():
            item.set_linewidth(1.2)

        if annotate:
            map_type = "eqarea" if self.iseqarea else "latlon"
            plt.text(-3.5, -1, f"""
                chip=averaged{kwargs_fig['goodchips']} 
                solver=IC14new {map_type} 
                noise={kwargs_fig['noisetype']} 
                err_level={flux_err} 
                contrast={kwargs_fig['contrast']} 
                limbdark={self.lld}""",
            fontsize=8)

    def plot_starry_map(self, ydeg=7, colorbar=False):
        fig, ax = plt.subplots(figsize=(7,3))
        showmap = starry.Map(ydeg=ydeg)
        showmap.load(self.bestparamgrid_r)
        showmap.show(ax=ax, projection="moll", colorbar=colorbar)

    def plot_fit_results(self):
        obs_2d = np.reshape(self.observed_1d, (self.nobs, self.nk))
        model_observation = self.dime.normalize_model(np.dot(self.bestparams, self.Rmatrix))
        bestmodel_2d = np.reshape(model_observation, (self.nobs, self.nk))
        flatmodel_2d = np.reshape(self.flatmodel, (self.nobs, self.nk))

        plt.figure(figsize=(5, 7))
        for i in range(self.nobs):
            plt.plot(self.dv, obs_2d[i] - 0.02*i, color='k', linewidth=1)
            #plt.plot(obs[i] - 0.02*i, '.', color='k', markersize=2)
            plt.plot(self.dv, bestmodel_2d[i] - 0.02*i, color='r', linewidth=1)
            plt.plot(self.dv, flatmodel_2d[i] - 0.02*i, '--', color='gray', linewidth=1)
        plt.legend(labels=['obs', 'best-fit map', 'flat map'])

class MaxEntropy():
    """A class for Maximum Entropy computations for Doppler imaging."""

    def __init__(self, alpha, nk, nobs):
        self.alpha = alpha
        self.nk = nk
        self.nobs = nobs
        self.data = None
        self.weights = None
        self.Rmatrix = None
        self.j = np.arange(self.nk * self.nobs)
        self.k = (np.floor(1.*self.j / self.nk) * self.nk).astype(int)
        self.l = (np.ceil((self.j + 1.) / self.nk) * self.nk - 1.).astype(int)
        self.jfrac = (self.j % self.nk) / (self.nk - 1.)

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
        xsum = 1. * np.sum(x)
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