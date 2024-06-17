######################################################
# Doppler Imaging and Maximum-Entropy

# Original author: Ian Crossfield (Python 2.7) 
# 02-2020 Emma Bubb: change to run on Python 3
# 06-2024 Xueqing Chen: change to class structure
######################################################

import numpy as np
import matplotlib.pyplot as plt
import paths

import scipy.constants as const
import scipy.optimize as opt
from scipy import interpolate
from scipy.signal import savgol_filter

from astropy.io import fits
import starry

import lsd_utils as lsd
import ELL_map_class as ELL_map
import modelfitting as mf
from config_run import npixs

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

    def __init__(self, instru, obskerns_norm, intrinsic_profiles, kwargs_IC14,
                nk, nobs, dbeta):
        ### General parameters ###
        self.nobs = nobs
        self.goodchips = kwargs_fig['goodchips']
        self.nchip = len(self.goodchips)
        self.npix0 = npixs[instru]
        self.pad = 100
        self.npix = self.npix0 - 2 * self.pad
        self.nk = nk
        print(f"nobs: {self.nobs}, nchip: {self.nchip}, npix: {self.npix}")

        ### Wavelegnth parameters ###
        self.wav0_nm = np.zeros((self.nchip, self.npix0))
        self.wav_nm = np.zeros((self.nchip, self.npix))
        self.wav_angs = np.array(self.wav_nm) * 10 # from nm to angstroms
        self.dbeta = np.diff(self.wav_angs).mean()/self.wav_angs.mean()
        self.dv = np.arange(np.floor(-self.nk/2.+0.5), np.floor(self.nk/2.+0.5)) \
            * - self.dbeta * const.c * 1e-3  # velocity grid in km/s

        ### Spectrum and LSD parameters ###
        self.observed = np.empty((self.nobs, self.nchip, self.npix), dtype=float)
        self.template = np.empty_like(self.observed)
        self.residual = np.empty_like(self.observed)
        self.error    = np.empty_like(self.observed)
        self.mean_spectrum = np.empty((self.nchip, self.npix0))
        #self.flux_err = np.empty((), dtype=float)
        self.err_LSD_profiles = np.empty((), dtype=float)

        self.deltaspecs = np.zeros((self.nobs, self.nchip, self.npix), dtype=float)
        self.kerns = np.zeros((self.nobs, self.nchip, self.nk), dtype=float)
        self.modkerns = np.zeros((self.nobs, self.nchip, self.nk), dtype=float)
        #self.make_lsd_profile(line_file, cont_file)

        self.obskerns_norm = np.zeros_like(self.kerns)
        self.intrinsic_profiles = np.zeros((self.nchip, self.nk), dtype=float) # is avg modkerns over time

        ### Max entropy inversion parameters ###
        self.alpha = kwargs_IC14['alpha']
        self.phases = kwargs_IC14['phases']
        self.inc = kwargs_IC14['inc']
        self.inc_ = (90 - self.inc) * np.pi / 180 # IC14 defined 0 <-> equator-on, pi/2 <-> face-on
        self.vsini = kwargs_IC14['vsini']
        self.lld = kwargs_IC14['LLD']
        self.iseqarea = kwargs_IC14['eqarea']
        self.nlat = kwargs_IC14['nlat']
        self.nlon = kwargs_IC14['nlon']
        self.dime = MaxEntropy(self.alpha, self.nk, self.nobs)

        ### Set up the intrinsic profile function
        mean_profile = np.median(self.intrinsic_profiles, axis=0) # can safely take means over chips now
        modIP = 1. - np.concatenate((np.zeros(300), mean_profile, np.zeros(300)))
        modDV = - np.arange(np.floor(-modIP.size/2.+.5), np.floor(modIP.size/2.+.5)) * self.dbeta * const.c * 1e-3
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

    def make_lsd_profile(self, line_file, cont_file, modname='t1500g1000f8', plot_deltaspec=False):
        '''Calculate the LSD profile for a given spectrum.'''
        # Read daospec linelist
        lineloc, lineew, _ = dao_getlines(line_file)
        pspec_cont = fits.getdata(cont_file)
        hdr_pspec_cont = fits.getheader(cont_file)
        wspec = hdr_pspec_cont['crval1'] + np.arange(pspec_cont.size)*hdr_pspec_cont['cdelt1']
        factor = 1e11 if "t1" in modname else 1e5 # don't know why different model needs scaling with a factor
        pspec_cont = pspec_cont/factor
        spline = interpolate.UnivariateSpline(wspec, pspec_cont, s=0.0, k=1.0) #interpolate over the continuum measurement
        
        # Calculate the LSD profile
        for i, jj in enumerate(self.goodchips): 
            print("chip", jj)
            for kk in range(self.nobs):
                shift = 1. + self.rv  # best rv shift for Callie is 9e-5
                deltaspec = lsd.make_deltaspec(
                    lineloc*shift, lineew, self.wav_angs[i], verbose=False, cont=spline(self.wav_angs[i]))
                _, self.kerns[kk,i] ,_ ,_ = lsd.dsa(deltaspec, self.observed[kk,i], self.nk)
                _, self.modkerns[kk,i],_ ,_ = lsd.dsa(deltaspec, self.template[kk,i,self.pad:-self.pad], self.nk) 
                self.deltaspecs[kk,i] = deltaspec
        self.err_LSD_profiles = np.median(self.kerns.mean(1).std(0))

        if plot_deltaspec:
            self.plot_lsd_specta()

        # Shift kerns to center
        self.modkerns, self.kerns = self.shift_kerns_to_center(shiftkerns=False)
    
        # Normalize kerns
        self.obskerns_norm = self.cont_normalize_kerns()

        self.intrinsic_profiles = np.array([self.modkerns[:,i].mean(0) for i in range(self.nchip)])

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

    def shift_kerns_to_center(self, sim=False, shiftkerns=True, verbose=False):
        '''shift modkerns to center at dv=0 and shift kerns for same amount.'''
        cen_modkerns = np.zeros_like(self.modkerns)
        cen_kerns = np.copy(self.kerns)
        for i,jj in enumerate(self.goodchips):
            for k in range(self.nobs):
                # find systematic rv offset from modkerns
                systematic_rv_offset = (self.modkerns[k,i]==self.modkerns[k,i].max()).nonzero()[0][0] - (self.dv==0).nonzero()[0][0]
                # shift modkerns to center at dv=0
                cen_modkerns[k,i] = np.interp(np.arange(self.nk), np.arange(self.nk) - systematic_rv_offset, self.modkerns[k,i])
                if verbose:
                    if (k == 0) and verbose:
                        print("modkerns shifted to center.")
                # shift kerns with same amount, if not sim or crires
                if shiftkerns and (not sim) and (self.instru != 'CRIRES'): 
                    cen_kerns[k,i] = np.interp(np.arange(self.nk), np.arange(self.nk) - systematic_rv_offset, self.kerns[k,i])
                    if (k == 0) and verbose:
                        print("kerns shifted to same amount.")

                if verbose:
                    print("chip:", jj , "obs:", k, "offset:", systematic_rv_offset)
        return cen_modkerns, cen_kerns

    def cont_normalize_kerns(self):
        '''Continuum-normalize kerns by fitting a line at the flat edges of kern.'''
        obskerns = 1. - self.kerns
        obskerns_norm = np.zeros_like(obskerns)
        continuumfit = np.zeros((self.nobs, self.nchip, 2))
        edgelen = 15 if self.instru != "CRIRES" else 7
        for i in range(self.nchip):
            for n in range(self.nobs):
                inds = np.concatenate((np.arange(0, edgelen), np.arange(self.nk - edgelen, self.nk)))
                continuumfit[n,i] = np.polyfit(inds, obskerns[n,i,inds], 1)
                obskerns_norm[n,i] = obskerns[n,i] / np.polyval(continuumfit[n,i], np.arange(self.nk))
        return obskerns_norm

    def plot_lsd_specta(self, t=0):
        plt.figure(figsize=(15, 2*self.nchip))
        for i, jj in enumerate(self.goodchips):
            plt.subplot(self.nchip, 1, i+1)
            plt.plot(self.wav_angs[i], self.deltaspecs[t,i], linewidth=0.5, color='C0', label="deltaspec")
            plt.plot(self.wav_angs[i], self.template[t,i], linewidth=0.6, color='C1', label="template")
            plt.plot(self.wav_angs[i], self.observed[t,i], linewidth=0.6, color='k', label="observed")
            plt.text(x=self.wav_angs[i].min()-10, y=0, s=f"order={jj}")
            if i==0:
                plt.title(f"model vs. lines at t={t}")
        plt.legend(loc=4, fontsize=9)
        plt.savefig(paths.output / "LSD_deltaspecs.png", transparent=True)


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
        

######################################################
##### utils ##########################################
######################################################
def dao_getlines(f_linelist):
    """
    Read the line locations and equivalent widths from a DAOSPEC output file.

    Example:
      f_linelist = 'model_spec.clines'
      (lineloc, lineew, linespec) = getlines(f_linelist)
    """
    #2009-02-22 10:15 IJC: Initiated

    # Get the line locations and EWs:
    with open(f_linelist, 'r') as f:
        raw = f.readlines()

    dat = np.zeros([len(raw), 2], dtype=float)                                                 
    for i, line in enumerate(raw):                                         
        dat[i,:]= list(map(float, line.split()[0:2]))

    lineloc = dat[:,0]
    lineew = dat[:,1]/1e3
    linespec = [line.split()[-1] for line in raw]
    return (lineloc, lineew, linespec)

