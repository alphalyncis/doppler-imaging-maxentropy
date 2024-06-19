######################################################
# Doppler Imaging and Maximum-Entropy

# Original author: Ian Crossfield (Python 2.7) 
# 02-2020 Emma Bubb: change to run on Python 3
# 06-2024 Xueqing Chen: change to class structure
######################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import paths
import pickle

import scipy.constants as const
import scipy.optimize as opt
from scipy import interpolate
from scipy.signal import savgol_filter
from scipy.ndimage.filters import gaussian_filter

from astropy.io import fits
import starry

import lsd_utils as lsd
import ELL_map_class as ELL_map
import modelfitting as mf
from config_run import npixs #TODO: remove this dependency

class DopplerImaging():
    """A class for Doppler Imaging reconstruction.

    Attributes
    ----------
    instru : str
        Instrument name, either "IGRINS" or "CRIRES".

    goodchips : list
        Indices of good orders to use, starting from 0.

    npix : int
        Number of pixels in the spectrum.

    nchip : int
        Number of orders used in Doppler imaging.

    nobs: int
        Number of observed epoches.

    nk: int
        Number of pixels of the LSD line profile kernel.

    inc : float
        Inclination of target in degrees (90 is equator-on).

    vsini : float
        Projected rotational velocity of target in km/s.

    rv : float
        Radial velocity of target in km/s.

    lld : float
        Limb-darkening coefficient.

    wav_nm : 2darray, shape=(nchip, npix)
        Wavelengths of the spectrum in nm.

    wav_angs : 2darray, shape=(nchip, npix)
        Wavelengths of the spectrum in angstroms.

    dbeta : float
        d_lam/lam_ref of the wavelength range that the line profile sits on.

    dv : 1darray, shape=(nk)
        Velocity grid in km/s.

    observed : 3darray, shape=(nobs, nchip, npix)
        The observed spectra cube in each epoch and order.

    template : 3darray, shape=(nobs, nchip, npix)
        The template spectra cube in each epoch and order.

    error : 3darray, shape=(nobs, nchip, npix)
        The error of observed spectra in each epoch and order.

    timestamps : 1darray, shape=(nobs)
        Timestamps of the observations in hours.

    flux_err : float
        Average error level of the observed spectrum.

    kerns : 3darray, shape=(nobs, nchip, nk)
        The observed line profiles extracted by LSD.

    modkerns : 3darray, shape=(nobs, nchip, nk)
        The unbroaded model line profiles extracted by LSD.

    err_LSD_profiles : float
        Error level of the observed LSD profiles.

    phases: 1darray, shape=(nobs)
        Phases corresponding to the observed timesteps. 
        In radian (0 ~ 2*pi).

    alpha : float
        Regularization parameter for the maximum entropy inversion.

    Rmatrix : 2darray, shape=(ncell, nobs*nk)
        The Doppler Imaging R matrix. See Vogt et al. 1987 for details.

    dime : MaxEntropy object
        The Maximum Entropy object, handy for computing inversion.

    dmap : ELL_map.Map object
        The map object for the Doppler imaging grid.

    iseqarea : bool
        Whether to use equal-area grid for the map. Default is True.
    
    nlat : int
        Number of latitudes in the map grid. 
        If equal area, is the input nlat to compute the grid,
        Updated to the actual nlat of longest row after the grid is computed.

    nlon : int
        Number of longitudes in the map grid.
        If equal area, is the input nlon to compute the grid.
        Updated to the actual nlon after the grid is computed.

    ncell : int
        Number of cells in the map grid. 
        If equal area, gets updated after the grid is computed.

    uncovered : list
        List of uncovered cells in the map grid.

    flatguess : 1darray, shape=(ncell)
        Initial guess of the flat map. Default is 100.

    flatmodel : 1darray, shape=(ncell)
        The flat map model computed by flatguess * Rmatrix.

    bounds : list 
        List of lower and upper bounds for map cell values.

    bestparams : 1darray, shape=(ncell)
        The best-fit map cell values in 1d.

    bestparamgrid : 2darray, shape=(nlat, nlon)
        The best-fit map cell values reshaped to 2d grid.
        If equal area, projected to a rectangular grid.

    fitres : Any
        Stores the optimization result of the best-fit map.

    """

    def __init__(self, wav_nm, observed, template, error, timestamps,
                 goodchips, instru='IGRINS', kwargs_IC14={}):
        #TODO: add types to the attributes

        ### General attributes ###
        self.instru = instru
        self.nobs = observed.shape[0]
        self.npix = observed.shape[2]
        self.goodchips = goodchips
        self.nchip = len(goodchips)
        print(f"nobs: {self.nobs}, nchip: {self.nchip}, npix: {self.npix}")

        ### Physical attributes ###
        self.inc = kwargs_IC14['inc']
        self.vsini = kwargs_IC14['vsini']
        self.rv = kwargs_IC14['rv']
        self.lld = kwargs_IC14['LLD']

        ### Spectrum attributes ###
        self.wav_nm = wav_nm
        self.wav_angs = np.array(self.wav_nm) * 10 # from nm to angstroms
        self.observed = np.empty((self.nobs, self.nchip, self.npix), dtype=float)
        self.template = np.empty_like(self.observed)
        self.error    = np.empty_like(self.observed)

        self.observed = observed
        self.template = template
        self.error = error
        self.timestamps = timestamps
        self.flux_err = eval(f'{np.median(self.error):.3f}') if self.instru == "IGRINS" else 0.02

        ### LSD attributes ###
        self.nk = kwargs_IC14['nk']
        self.dbeta = np.diff(self.wav_angs).mean()/self.wav_angs.mean()
        self.dv = np.arange(np.floor(-self.nk/2.+0.5), np.floor(self.nk/2.+0.5)) \
            * - self.dbeta * const.c * 1e-3  # velocity grid in km/s

        self.deltaspecs = np.zeros((self.nobs, self.nchip, self.npix), dtype=float) 
        self.kerns      = np.zeros((self.nobs, self.nchip, self.nk), dtype=float) # observed profiles
        self.modkerns   = np.zeros((self.nobs, self.nchip, self.nk), dtype=float) # unbroaded model profiles
        self.err_LSD_profiles = np.empty((), dtype=float)
        self.obskerns_norm = np.zeros_like(self.kerns)
        self.uniform_profiles   = np.zeros((self.nchip, self.nk), dtype=float) # is avged obskerns over time
        self.intrinsic_profiles = np.zeros((self.nchip, self.nk), dtype=float) # is avged modkerns over time

        ### Maximum entropy inversion parameters ###
        self.alpha = kwargs_IC14['alpha']
        self.phases = kwargs_IC14['phases'] #TODO: simplify phase/timestamp?
        self.inc_ = (90 - self.inc) * np.pi / 180 # IC14 defined 0 <-> equator-on, pi/2 <-> face-on
        self.iseqarea = kwargs_IC14['eqarea']
        self.nlat = kwargs_IC14['nlat']
        self.nlon = kwargs_IC14['nlon']
        self.dime = MaxEntropy(self.alpha, self.nk, self.nobs)

        ### Set up Map object
        if self.iseqarea:
            self.dmap = ELL_map.Map(nlat=self.nlat, nlon=self.nlon, type='eqarea', inc=self.inc_, verbose=True)
        else:
            self.dmap = ELL_map.Map(nlat=self.nlat, nlon=self.nlon, inc=self.inc_) #ELL_map.map returns a class object

        self.ncell = self.dmap.ncell
        self.uncovered = list(range(self.ncell))
        self.flatguess = 100 * np.ones(self.ncell)
        self.flatmodel = None
        self.bounds = [(1e-6, 300)] * self.ncell

        ### Set up the R matrix
        self.Rmatrix = np.zeros((self.ncell, self.nobs*self.dv.size), dtype=np.float32)
        
        ### Set up attributes for results
        self.bestparams = None
        self.bestparamgrid = None
        self.metric = None
        self.chisq = None
        self.entropy = None
        self.fitres = None

    def make_lsd_profile(self, line_file, cont_file, modname='t1500g1000f8', savedir=None,
                         plot_deltaspec=False, plot_lsd_profiles=True, plot_deviation_map=True):
        '''Calculate the LSD profile for a given spectrum.

        Parameters
        ----------
            line_file : str
                Path to the linelist file.

            cont_file : str
                Path to the continuum file.

            modname : str
                Model name for the continuum file. Default is 't1500g1000f8'.

            savedir : str
                Full path to save the figures. Default is None.

            plot_deltaspec : bool
                If True, plot the LSD profile and the template. Default is False.

            plot_lsd_profiles : bool
                If True, plot the LSD profiles. Default is True.

            plot_deviation_map : bool
                If True, plot the deviation map. Default is True.
        
        '''
        # Read daospec linelist
        lineloc, lineew, _ = lsd.dao_getlines(line_file)
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
                _, self.modkerns[kk,i],_ ,_ = lsd.dsa(deltaspec, self.template[kk,i], self.nk) 
                self.deltaspecs[kk,i] = deltaspec
        self.err_LSD_profiles = np.median(self.kerns.mean(1).std(0))

        if plot_deltaspec:
            self.plot_lsd_spectra()
            if savedir is not None:
                plt.savefig(f'{savedir}/deltaspec.png', bbox_inches="tight", dpi=150, transparent=True)

        # Shift kerns to center
        self.modkerns, self.kerns = self.shift_kerns_to_center(shiftkerns=False)
    
        # Normalize kerns
        self.obskerns_norm = self.cont_normalize_kerns()

        self.uniform_profiles = np.array([self.obskerns_norm[:,c].mean(axis=0) for c in range(self.nchip)])
        self.intrinsic_profiles = np.array([self.modkerns[:,c].mean(axis=0) for c in range(self.nchip)])

        if plot_lsd_profiles:
            self.plot_lp_timeseries(self.obskerns_norm, self.timestamps)
            if savedir is not None:
                plt.savefig(f'{savedir}/LSD_profiles.png', bbox_inches="tight", dpi=150, transparent=True)

        if plot_deviation_map:
            self.plot_deviation_map(self.obskerns_norm, self.timestamps)
            if savedir is not None:
                plt.savefig(f'{savedir}/deviation_map.png', bbox_inches="tight", dpi=150, transparent=True)

    def compute_Rmatrix(self):
        '''Compute the R matrix for the inversion. See Vogt et al. 1987 for details.
        Requires intrinsic_profiles attribute to be set first.
        Updates Rmatrix attribute.

        Parameters
        ----------
            None

        Return
        ------
            None
        '''
        ### Set up the intrinsic profile function
        mean_profile = np.median(self.intrinsic_profiles, axis=0) # can safely take means over chips now
        modIP = 1. - np.concatenate((np.zeros(300), mean_profile, np.zeros(300)))
        modDV = - np.arange(np.floor(-modIP.size/2.+.5), np.floor(modIP.size/2.+.5)) * self.dbeta * const.c * 1e-3
        modelfunc = interpolate.UnivariateSpline(modDV[::-1], modIP[::-1], k=1., s=0.) # function that returns the intrinsic profile

        for kk, rot in enumerate(self.phases):
            speccube = np.zeros((self.ncell, self.dv.size), dtype=np.float32) 
            if self.iseqarea:
                this_map = ELL_map.Map(nlat=self.nlat, nlon=self.nlon, type='eqarea', inc=self.inc_, deltaphi=-rot)
            else:
                this_map = ELL_map.Map(nlat=self.nlat, nlon=self.nlon, inc=self.inc_, deltaphi=-rot)
            this_doppler = 1. + self.vsini*this_map.visible_rvcorners.mean(1)/const.c/np.cos(self.inc_) # mean rv of each cell in m/s
            good = (this_map.projected_area>0) * np.isfinite(this_doppler)    
            for ii in good.nonzero()[0]:
                if ii in self.uncovered:       # to collect uncovered cells,
                    self.uncovered.remove(ii)  # remove cells that are visible at this rot
                speccube[ii,:] = modelfunc(self.dv + (this_doppler[ii]-1)*const.c/1000.)
            limbdarkening = (1. - self.lld) + self.lld * this_map.mu
            Rblock = speccube * ((limbdarkening*this_map.projected_area).reshape(this_map.ncell, 1)*np.pi/this_map.projected_area.sum())
            self.Rmatrix[:, self.dv.size*kk:self.dv.size*(kk+1)] = Rblock

    def solve(self, create_obs_from_diff=True, solver='scipy', maxiter=1e4, ftol=1e-5):
        '''Solve the Doppler imaging inversion using the Maximum Entropy method.
        Computes the R matrix, sets up the observed data and weights, and solves 
        for the best parameters by minizing (metric = 0.5 * chisq - alpha * entropy).
        Updates the bestparams and bestparamgrid attributes when called.

        Parameters
        ----------
            create_obs_from_diff: bool
                if True, create new observation from the difference between observed 
                and uniform profile.

            solver: str
                minimizer used for fitting, 'ic14' or 'scipy'.

        Return
        ------
            None
        '''
        ### Compute the R matrix
        self.compute_Rmatrix()

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
        
        ### Create diff+flat profile
        self.flatmodel = self.dime.normalize_model(np.dot(self.flatguess, self.Rmatrix))
        if create_obs_from_diff:
            mean_dev = np.median(np.array(
                [self.obskerns_norm[:,c] - self.uniform_profiles[c] for c in range(self.nchip)]
            ), axis=0) # mean deviation over chips
            flatmodel_2d = np.reshape(self.flatmodel, (self.nobs, self.nk))
            new_observation_2d = mean_dev + flatmodel_2d
            new_observation_1d = new_observation_2d.ravel()
            self.observed_1d = new_observation_1d

        ### Scale the observations to match the model's linear trend and offset
        linfunc = lambda x, c0, c1: x * c0 + c1
        coeff, cov = opt.curve_fit(linfunc, xdata=self.observed_1d, ydata=self.flatmodel, 
                                    sigma=self.weights, absolute_sigma=True, p0=[1, 0])
        self.observed_1d = linfunc(self.observed_1d, *coeff)

        ### Solve!
        self.dime.set_data(self.observed_1d, self.weights, self.Rmatrix)
        
        if solver == 'ic14':
            bfit = mf.gfit(self.dime.entropy_map_norm_sp, self.flatguess, fprime=self.dime.getgrad_norm_sp, 
                        args=(), ftol=0.01, disp=1, maxiter=maxiter, bounds=self.bounds)
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
        '''Reshape a 1D map vector to a 2D grid.
        Requires bestparams attribute, to be called after solve().

        Return
        ------
            bestparams2d: 2darray, shape=(nlat, nlon)
                The best-fit map cell values reshaped to 2d grid.
        '''
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

    def plot_lsd_spectra(self, t=0, savedir=None):
        '''Plot the LSD profile and the template.

        Parameters
        ----------
            t: int
                Index of the observation epoch to plot, from 0 to nobs-1.
            savedir: str
                Full path (include filename) to save the figure.
        '''
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
        if savedir is not None:
            plt.savefig(savedir, bbox_inches="tight", dpi=150, transparent=True)

    def plot_lp_timeseries_all(self, line_profiles, intrinsic_profiles=None, savedir=None, gap=0.03):
        '''Plot time series of line profiles of each order.

        Parameters
        ---------- 
            line_profiles: 3darray, shape=(nobs, nchip, nk)
            intrinsic_profiles: 2darray, shape=(nchip, nk)
            savedir: str
                Full path (include filename) to save the figure.
            gap: float
                Gap between each line profile.
        '''
        colors = [cm.gnuplot_r(x) for x in np.linspace(0, 1, self.nobs+4)]
        plt.figure(figsize=(self.nchip*3, 4))
        for i, jj in enumerate(self.goodchips):
            plt.subplot(1, self.nchip, i+1)
            for n in range(self.nobs):
                plt.plot(self.dv, line_profiles[n,i] - gap*n, color=colors[n])
            if intrinsic_profiles is not None:
                plt.plot(self.dv, 1-intrinsic_profiles[i], color='k')
            plt.title(f"chip={jj}")
            plt.xlabel("dv")
        plt.tight_layout()
        if savedir is not None:
            plt.savefig(savedir, bbox_inches="tight", dpi=150, transparent=True)

    def plot_lp_timeseries(self, line_profiles, timestamps, savedir=None, gap=0.025):
        '''Plot time series of order-averaged line profiles.

        Parameters
        ----------
            line_profiles: 3darray, shape=(nobs, nchip, nk)
            timestamps: 1darray, shape=(nobs)
            savedir: str
                Full path (include filename) to save the figure.
            gap: float
                Gap between each line profile.
        '''
        colors = [cm.gnuplot_r(x) for x in np.linspace(0, 1, self.nobs+4)]
        fig, ax = plt.subplots(figsize=(4, 5))
        cut = int((self.nk - 70) / 2 + 1.)
        for n in range(self.nobs):
            ax.plot(self.dv[cut:-cut], line_profiles.mean(axis=0).mean(axis=0)[cut:-cut] - gap*n, "--", color="gray", alpha=0.5)
            ax.plot(self.dv[cut:-cut], line_profiles[n].mean(axis=0)[cut:-cut] - gap*n, color=colors[n+1])
        ax.set_xlabel("velocity (km/s)")
        ax.set_xticks([-50, -25, 0, 25, 50])
        ax.set_ylabel("Line intensity")
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ybound())
        ax2.set_yticks([1 - gap*n for n in range(self.nobs)], labels=[f"{t:.1f}h" for t in timestamps], fontsize=9)
        #plt.axvline(x=vsini/1e3, color="k", linestyle="dashed", linewidth=1)
        #plt.axvline(x=-vsini/1e3, color="k", linestyle="dashed", linewidth=1)
        #plt.legend(loc=4, bbox_to_anchor=(1,1))
        if savedir is not None:
            plt.savefig(savedir, bbox_inches="tight", dpi=200, transparent=True)

    def plot_deviation_map_all(self, line_profiles, timestamps, savedir=None, lim=0.003):
        '''Plot deviation map for each order. 
        A darker deviation pattern means a surface feature fainter than the background.
        
        Parameters
        ----------
            line_profiles: 3darray, shape=(nobs, nchip, nk)

            timestamps: 1darray, shape=(nobs)

            savedir: str
                Full path (include filename) to save the figure.
        '''
        uniform_profiles = np.zeros((self.nchip, self.nk))
        ratio = 1.3 if self.nobs < 10 else 0.7

        # plot deviation map for each chip
        plt.figure(figsize=(self.nchip*4, 3))
        for i, jj in enumerate(self.goodchips):
            uniform_profiles[i] = line_profiles[:,i].mean(axis=0) # averaged LP over times
            #TODO: change uniform_profiles to median
            plt.subplot(1, self.nchip, i+1)
            plt.imshow(line_profiles[:,i] - uniform_profiles[i], 
                extent=(self.dv.max(), self.dv.min(), timestamps[-1], 0),
                aspect=int(ratio * 29),
                cmap='YlOrBr') # positive diff means dark spot
            cut = self.nk - 70 if self.nk > 70 else 0
            plt.xlim(self.dv.min() + cut, self.dv.max() - cut),
            plt.xlabel("velocity (km/s)")
            plt.ylabel("Elapsed time (h)")
            plt.title(f"chip={jj}")
        plt.tight_layout()
        if savedir is not None:
            plt.savefig(savedir, bbox_inches="tight", dpi=150, transparent=True)
    
    def plot_deviation_map(self, line_profiles, timestamps, savedir=None, meanby='median', 
                           colorbar=True, lim=0.003):
        '''Plot order-averaged deviation map.
        A darker deviation pattern means a surface feature fainter than the background.

        Parameters
        ----------
            line_profiles: 3darray, shape=(nobs, nchip, nk)

            timestamps: 1darray, shape=(nobs)

            savedir: str
                Full path (include filename) to save the figure.
            meanby: str
                Method to calculate the mean deviation: 
                'median': take the median of the deviation over all orders
                'median_each': take the median LP for each order, then take the deviation from uniform LP
                'mean': take the mean of the deviation over all orders
        '''
        ratio = 1.3 if self.nobs < 10 else 0.7

        if meanby == "median":
            mean_dev = np.median(np.array([line_profiles[:,i] - self.uniform_profiles[i] for i in range(self.nchip)]), axis=0) # mean over chips
        elif meanby == "median_each":
            mean_dev = np.median(line_profiles, axis=1) - np.median(self.uniform_profiles,axis=0)
        elif meanby == "mean":
            mean_dev = np.mean(np.array([line_profiles[:,i] - self.uniform_profiles[i] for i in range(self.nchip)]), axis=0) # mean over chips
        plt.figure(figsize=(10,6))
        plt.imshow(mean_dev, 
            extent=(self.dv.max(), self.dv.min(), timestamps[-1], 0),
            aspect=int(ratio * 29),
            cmap='YlOrBr',
            vmin=-lim, vmax=lim) # positive diff means dark spot
        cut = self.nk - 70 if self.nk > 70 else 0
        plt.xlim(self.dv.min() + cut, self.dv.max() - cut),
        plt.xlabel("velocity (km/s)", fontsize=8)
        plt.xticks([-50, -25, 0, 25, 50], fontsize=8)
        plt.ylabel("Elapsed time (h)", fontsize=8)
        plt.yticks([0, 1, 2, 3, 4, 5], fontsize=8)
        plt.vlines(x=self.vsini/1e3, ymin=0, ymax=timestamps[-1], colors="k", linestyles="dashed", linewidth=1)
        plt.vlines(x=-self.vsini/1e3, ymin=0, ymax=timestamps[-1], colors="k", linestyles="dashed", linewidth=1)
        if colorbar:
            cb = plt.colorbar(fraction=0.06, pad=0.28, aspect=15, orientation="horizontal", label="%")
            cb_ticks = cb.ax.get_xticks()
            cb.ax.set_xticklabels([f"{t*100:.1f}" for t in cb_ticks])
            cb.ax.tick_params(labelsize=8)
        plt.tight_layout()
        if savedir is not None:
            plt.savefig(savedir, bbox_inches="tight", dpi=150, transparent=True)

    def plot_mollweide_map(self, clevel=5, sigma=1, colorbar=False, vmax=None, vmin=None, 
                           colormap=plt.cm.plasma, contour=False, savedir=None, annotate=False):
        '''Plot Doppler map from a 2d array.

        Parameters
        ----------
            clevel: int
                Number of contour levels.
            sigma: float
                Smoothing factor for contour.
            colorbar: bool
                Whether to plot colorbar.
            vmax: float
                Max value for colorbar, relative to 100.
            vmin: float
                Min value for colorbar, relative to 100.
            colormap: plt.cm object
                Colormap for the plot.
            contour: bool
                Whether to plot contour.
            savedir: str
                Full path (include filename) to save the figure.
            annotate: bool
                Whether to annotate the plot.
        '''
        cmap = colormap.copy()
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
        if contour:
            contour = ax.contour(Lon, Lat, gaussian_filter(self.bestparamgrid, sigma), clevel, colors='white', linewidths=0.5)
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
                chip=averaged{self.goodchips} 
                solver=IC14new {map_type}
                err_level={flux_err}
                limbdark={self.lld}""",
            fontsize=8)
        
        if savedir is not None:
            plt.savefig(savedir, bbox_inches="tight", dpi=150, transparent=True)

    def plot_starry_map(self, ydeg=7, colorbar=False, savedir=None):
        '''Plot the best-fit map using starry.

        Parameters
        ----------
            ydeg: int
                Degree of spherical harmonics used in starry map.

            colorbar: bool
                Whether to plot colorbar.
            
            savedir: str
                Full path (include filename) to save the figure.
        '''
        fig, ax = plt.subplots(figsize=(7,3))
        showmap = starry.Map(ydeg=ydeg)
        showmap.load(self.bestparamgrid_r)
        showmap.show(ax=ax, projection="moll", colorbar=colorbar)

    def plot_fit_results(self, savedir=None):
        '''Plot the observed and best-fit LP series.

        Parameters
        ----------
            savedir: str
                Full path (include filename) to save the figure.
        '''
        obs_2d = np.reshape(self.observed_1d, (self.nobs, self.nk))
        model_observation = self.dime.normalize_model(np.dot(self.bestparams, self.Rmatrix))
        bestmodel_2d = np.reshape(model_observation, (self.nobs, self.nk))
        flatmodel_2d = np.reshape(self.flatmodel, (self.nobs, self.nk))

        plt.figure(figsize=(7, 7))
        for i in range(self.nobs):
            plt.plot(self.dv, obs_2d[i] - 0.02*i, color='k', linewidth=1)
            plt.plot(self.dv, bestmodel_2d[i] - 0.02*i, color='r', linewidth=1)
            plt.plot(self.dv, flatmodel_2d[i] - 0.02*i, '--', color='gray', linewidth=1)
        plt.legend(labels=['observed', 'best-fit', 'flat'], loc=4)
        plt.xlabel("velocity (km/s)")

        if savedir is not None:
            plt.savefig(savedir, bbox_inches="tight", dpi=150, transparent=True)


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
        
############################################################################################################
##### Utils #####
############################################################################################################

def load_data(model_datafile, goodchips, instru='IGRINS', pad=100):
    '''Load model and data from a pickle file.'''

    with open(model_datafile, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    lams = np.median(data["chiplams"], axis=0) # in um
    nobs = data['fobs0'].shape[0]
    npix = data['fobs0'].shape[2]
    nchip = len(goodchips)
    print(f"Data loaded from file {model_datafile}.")
    print(f"nobs: {nobs}, nchip: {nchip}, npix: {npix}")

    observed = np.empty((nobs, nchip, npix))
    template = np.empty((nobs, nchip, npix))
    error = np.empty((nobs, nchip, npix))

    if instru == "IGRINS":
        for k in range(nobs):
            for i, jj in enumerate(goodchips):
                observed[k, i] = np.interp(
                    lams[jj], 
                    data["chiplams"][k][jj],
                    data["fobs0"][k][jj]
                )
                template[k, i] = np.interp(
                    lams[jj],
                    data["chiplams"][k][jj],
                    data["chipmodnobroad"][k][jj]
                )
                error[k, i] = np.interp(
                    lams[jj],
                    data["chiplams"][k][jj],
                    remove_spike(data["eobs0"][k][jj])
                )

    elif instru == "CRIRES":
        for k in range(nobs):
            for i, jj in enumerate(goodchips):
                observed[k, i] = np.interp(
                    lams[jj],
                    data["chiplams"][k][jj],
                    data["obs1"][k][jj] / data["chipcors"][k][jj],
                )
                template[k, i] = np.interp(
                    lams[jj],
                    data["chiplams"][k][jj],
                    data["chipmodnobroad"][k][jj] / data["chipcors"][k][jj],
                )

    # Trim the edges of spectrum
    wav_nm = lams[goodchips, pad:-pad] * 1000 # um to nm
    observed = observed[:, :, pad:-pad]
    template = template[:, :, pad:-pad]
    error = error[:, :, pad:-pad]

    print("observed:", observed.shape)
    print("template:", template.shape)
    print("wav:", wav_nm.shape)

    return wav_nm, template, observed, error

def remove_spike(data, kern_size=10, lim_denom=5):
    data_pad = np.concatenate([np.ones(kern_size)*np.median(data[:kern_size]), data, np.ones(kern_size)*np.median(data[-kern_size:-1])])
    data_filt = np.copy(data)
    for i, val in enumerate(data):
        i_pad = i + kern_size
        seg = data_pad[i_pad-kern_size:i_pad+kern_size]
        seg = seg[np.abs(seg-np.median(seg))<20]
        lim = np.median(seg)/lim_denom
        if val > np.median(seg) + lim or val < np.median(seg) - lim:
            data_filt[i] = np.median(seg[int(kern_size/5):-int(kern_size/5)])
    return data_filt

def make_toy_spectrum(wavmin, wavmax, npix, 
                      amps=[0.9, 0.9, 0.9, 0.9, 0.9],
                      centers=[2330, 2335, 2338, 2345, 2347], 
                      widths=[0.03, 0.03, 0.03, 0.03, 0.03]):
    '''Make a toy spectrum for testing.

    Parameters
    ----------
        wavmin: float
            Minimum wavelength in nm.
        wavmax: float
            Maximum wavelength in nm.
        npix: int
            Number of pixels in spectrum.
        amps: list
            Amplitudes of the Gaussian lines.
        centers: list
            Centers of the Gaussian lines.
        widths: list
            Widths of the Gaussian lines.

    Returns
    -------
        wav: 1darray
            Wavelength array in nm.
        toy_spec: 1darray
            Toy spectrum with Gaussian lines.
    '''
    wav = np.linspace(wavmin, wavmax, npix)
    spec = np.ones_like(wav)
    for i in range(len(amps)):
        spec -= amps[i] * np.exp(-0.5 * (wav - centers[i])**2 / widths[i]**2)
    err = 0.002

    return wav, spec, err