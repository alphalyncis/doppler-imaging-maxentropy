import os
from astropy.io import fits
from astropy.table import Table
import numpy as np
from scipy import signal
import scipy.constants as const
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

# 20260608 - fixed eso sky units, fixed slit width and nrows, update itime, LLD, resolution

homedir = os.path.expanduser('~')
teldir = f'{homedir}/uoedrive/data/telluric'
outdir = 'metric_output'

### Physical constants
rjup = 7.1492e7       # Jupiter equatorial radius, m
pc   = 3.08568025e16  # parsec, m
c_si = 299792458      # speed of light, m/s
h    = 6.626068e-34   # Planck's constant, J·s
BT_SETTL_TEFF_GRID = list(range(700, 3100, 100)) # BT-Settl CIFIST grid

def gaussian(p, x):
    """ Compute a gaussian distribution at the points x.
        p is a three- or four-component array, list, or tuple:

        y =  [p3 +] p0/(p1*sqrt(2pi)) * exp(-(x-p2)**2 / (2*p1**2))

        p[0] -- Area of the gaussian
        p[1] -- one-sigma dispersion
        p[2] -- central offset (mean location)
        p[3] -- optional constant, vertical offset

        NOTE: FWHM = 2*sqrt(2*ln(2)) * p1  ~ 2.3548*p1

        SEE ALSO:  egaussian"""
    #2008-09-11 15:11 IJC: Created for LINEPROFILE
    # 2011-11-10 12:00 IJMC: Don't copy x or p inputs; fiddled with
    # background addition and ordering of operations
    # 2012-12-23 12:12 IJMC: Made it a tad faster.
    
    if not isinstance(x, np.ndarray):
        x = np.array(x, dtype=float, copy=False)

    y = p[0]/(p[1]*np.sqrt(2*np.pi)) * np.exp(-0.5 * ((x-p[2])/p[1])**2)

    if len(p)>3:
        y += p[3]
    
    return y

def putvecsinarray(vecs):
    """Take a tuple, list, or array of 1D arrays and always return their 
       Vstacked combination.  (Just a helper function, folks!)
    """
    # 2017-04-19 12:28 IJMC: Created
    
    if isinstance(vecs,tuple) or isinstance(vecs,list):
        vecs = np.vstack(vecs).transpose()
    elif isinstance(vecs, np.ndarray) and vecs.ndim < 2:
        vecs = vecs.reshape(len(vecs),1)
    else:
        vecs = np.array(vecs, copy=False)
    return vecs

def polyfitr(x, y, N, s, fev=100, w=None, diag=False, clip='both', \
                 verbose=False, plotfit=False, plotall=False, eps=1e-13, catchLinAlgError=False, xerr=None, yerr=None, retodr=False, checkvals=True):
    """Matplotlib's polyfit with weights and sigma-clipping rejection.

    :DESCRIPTION:
      Do a best fit polynomial of order N of y to x.  Points whose fit
      residuals exeed s standard deviations are rejected and the fit is
      recalculated.  Return value is a vector of polynomial
      coefficients [pk ... p1 p0].

    :OPTIONS:
        w: a set of weights for the data; uses CARSMath's weighted
             polynomial fitting routine instead of numpy's standard
             polyfit.  
             NOTE: if using errors in both x and y ("orthogonal
             distance regression") then don't set w --- instead set
             xerr and yerr (see below).

        fev:  number of function evaluations to call before stopping

        'diag'nostic flag:  Return the tuple (p, chisq, n_iter)

        clip: 'both' -- remove outliers +/- 's' sigma from fit
              'above' -- remove outliers 's' sigma above fit
              'below' -- remove outliers 's' sigma below fit
              'None'/None -- no  outlier removal

        xerr/yerr : one-sigma uncertainties in x and y. If these are
                    set, you are committing to an "orthogonal distance regression"

        retodr : bool
          If True, return the tuple (parameters, scipy_ODR_object)

        catchLinAlgError : bool
          If True, don't bomb on LinAlgError; instead, return [0, 0, ... 0].

    :REQUIREMENTS:
       :doc:`CARSMath`

    :NOTES:
       Iterates so long as n_newrejections>0 AND n_iter<fev.

    """
    # 2008-10-01 13:01 IJC: Created & completed
    # 2009-10-01 10:23 IJC: 1 year later! Moved "import" statements within func.
    # 2009-10-22 14:01 IJC: Added 'clip' options for continuum fitting
    # 2009-12-08 15:35 IJC: Automatically clip all non-finite points
    # 2010-10-29 09:09 IJC: Moved pylab imports inside this function
    # 2012-08-20 16:47 IJMC: Major change: now only reject one point per iteration!
    # 2012-08-27 10:44 IJMC: Verbose < 0 now resets to 0
    # 2013-05-21 23:15 IJMC: Added catchLinAlgError
    # 2017-05-12 11:37 IJMC: Added option for orthogonal distance regression
    
    #from CARSMath import polyfitw
    #from numpy import polyfit, polyval
    from numpy.linalg import LinAlgError
    #from pylab import plot, legend, title, figure

    import scipy.odr as odr
    
    if verbose < 0:
        verbose = 0

    xx = np.array(x, copy=False)
    yy = np.array(y, copy=False)
    noweights = (w is None) and (xerr is None) and (yerr is None)
    if noweights:
        ww = np.ones(xx.shape, float)
    else:
        ww = np.array(w, copy=False)

    fitxy = False
    if xerr is None and yerr is not None:
        w = 1./np.array(yerr)**2
    elif xerr is not None and yerr is not None:
        fitxy = True
        xerr = putvecsinarray(xerr).ravel()
        yerr = putvecsinarray(yerr).ravel()
        
    ii = 0
    nrej = 1

    if checkvals:
        goodind = np.isfinite(xx)*np.isfinite(yy)
        if noweights:
            pass
        elif fitxy:
            goodind *= (np.isfinite(xerr) * np.isfinite(yerr))
        else:
            goodind *= np.isfinite(ww)

        xx2 = xx[goodind]
        yy2 = yy[goodind]
        if fitxy:
            xerr2 = xerr[goodind]
            yerr2 = yerr[goodind]
        else:
            ww2 = ww[goodind]

            
    if fitxy:
        poly_model = odr.Model(np.polyval)
        guess = polyfitr(xx2,yy2, N, s=s, fev=fev, w=1./yerr2**2)
    
    while (ii<fev and (nrej!=0)):
        if noweights:
            p = np.polyfit(xx2,yy2,N)
            residual = yy2 - np.polyval(p,xx2)
            stdResidual = np.std(residual)
            clipmetric = s * stdResidual
        else:
            if fitxy:
                data = odr.RealData(xx2, yy2, sx=xerr2, sy=yerr2)
                odrobj = odr.ODR(data, poly_model, beta0=guess, maxit=10000)
                odrout = odrobj.run()
                p = odrout.beta
                residual = np.sqrt((odrout.delta / xerr2)**2 + (odrout.eps / yerr2)**2)
                guess = p.copy()

            else:
                if catchLinAlgError:
                    try:
                        #p = np.polyfitw(xx2,yy2, ww2, N)
                        p = np.polyfit(xx2, yy2, N, w=np.sqrt(ww2))
                    except LinAlgError:
                        p = np.zeros(N+1, dtype=float)
                else:
                    #p = polyfitw(xx2,yy2, ww2, N)
                    p = np.polyfit(xx2, yy2, N, w=np.sqrt(ww2))

                #p = p[::-1]  # polyfitw uses reverse coefficient ordering
                
                residual = (yy2 - np.polyval(p,xx2)) * np.sqrt(ww2)
                
            clipmetric = s

        if clip=='both':
            worstOffender = abs(residual).max()
            #pdb.set_trace()
            if worstOffender <= clipmetric or worstOffender < eps:
                ind = np.ones(residual.shape, dtype=bool)
            else:
                ind = abs(residual) < worstOffender
        elif clip=='above':
            worstOffender = residual.max()
            if worstOffender <= clipmetric:
                ind = np.ones(residual.shape, dtype=bool)
            else:
                ind = residual < worstOffender
        elif clip=='below':
            worstOffender = residual.min()
            if worstOffender >= -clipmetric:
                ind = np.ones(residual.shape, dtype=bool)
            else:
                ind = residual > worstOffender
        else:
            ind = np.ones(residual.shape, dtype=bool)
    
        xx2 = xx2[ind]
        yy2 = yy2[ind]
        if fitxy:
            xerr2 = xerr2[ind]
            yerr2 = yerr2[ind]
        elif (not noweights):
            ww2 = ww2[ind]
        ii = ii + 1
        nrej = len(residual) - len(xx2)
        if plotall:
            plt.figure()
            plt.plot(x,y, '.', xx2,yy2, 'x', x, np.polyval(p, x), '--')
            plt.legend(['data', 'fit data', 'fit'])
            plt.title('Iter. #' + str(ii) + ' -- Close all windows to continue....')

        if verbose:
            print( str(len(x)-len(xx2)) + ' points rejected on iteration #' + str(ii))

    if (plotfit or plotall):
        plt.figure()
        plt.plot(x,y, '.', xx2,yy2, 'x', x, np.polyval(p, x), '--')
        plt.legend(['data', 'fit data', 'fit'])
        plt.title('Close window to continue....')

    if diag:
        chisq = ( (residual)**2 / yy2 ).sum()
        p = (p, chisq, ii)

    if retodr:
        ret = p, odrout
    else:
        ret = p
    return ret

def rotationalProfile(delta_epsilon, delta_lam):
    """Compute the rotational profile of a star, assuming solid-body
    rotation and linear limb darkening.

    This uses Eq. 18.14 of Gray's Photospheres, 2005, 3rd Edition.

    :INPUTS:

      delta_epsilon : 2-sequence

        [0] : delta_Lambda_L = lambda * V * sin(i)/c; the rotational
              displacement at the stellar limb.

        [1] : epsilon, the linear limb darkening coefficient, used in
              the relation I(theta) = I0 + epsilon * (cos(theta) - 1).

        [2] : OPTIONAL! The central location of the profile (otherwise
              assumed to be located at delta_lam=0).

      delta_lam : scalar or sequence
        Wavelength minus offset: Lambda minus lambda_0.  Grid upon
        which computations will be done.

    :EXAMPLE:
      ::

        import pylab as py
        import spec

        dlam = py.np.linspace(-2, 2, 200) # Create wavelength grid
        profile = spec.rotationalProfile([1, 0.6], dlam)

        py.figure()
        py.plot(dlam, profile)
    """
    # 2013-05-26 10:37 IJMC: Created.

    delta_lambda_L, epsilon = delta_epsilon[0:2]
    if len(delta_epsilon)>2:  # optional lambda_offset
        lamdel2 = 1. - ((delta_lam - delta_epsilon[2])/delta_lambda_L)**2
    else:
        lamdel2 = 1. - (delta_lam/delta_lambda_L)**2
    
    if not hasattr(delta_lam, '__iter__'):
        delta_lam = np.array([delta_lam])

    ret = (4*(1.-epsilon) * np.sqrt(lamdel2) + np.pi*epsilon*lamdel2) / \
        (2*np.pi * delta_lambda_L * (1. - epsilon/3.))    

    ret[lamdel2<0] = 0.

    return ret

def modelspec_tel_template2(params, lam_template, template, lam_atmo, atmo, NPW, NPC, npix, retlam=False):
    """    
     :INPUTS:
       params:      

          [0]: vsini for rotational broadening profile, in units of km/s.
               (See :func:`spec.rotationalProfile`)

          [1]: linear limb darkening coefficient for rotational
               broadening profile. (See :func:`spec.rotationalProfile`)

          [2]: rv of target divided by speed of light, (rv/c)

          [3]: fwhm for (Gaussian) instrumental profile convolution,
               in units of km/s.

          [4]: effective scaling of telluric transmission specturm in
              'atmo'.  This will be scaled as: 1.0 - (scale*(1.0 -
              atmo)). After convolution, any negative values will be
              set to zero. Thus line-broadening is explicitly *not*
              treated correctly.

          [5:5+NPW]: the wavelength solution coefficients. These will
                     be passed to numpy.polyval with the vector
                     "arange(npix)/npix"

          [5+NPW:5+NPW+NPC]: the flux normalization
                     coefficients. These will also be passed to
                     numpy.polyval with the vector "arange(npix)/npix"

       lam_template: wavelength scale of input template. Ideally this
                       is more finely sampled than lam.

       template: template SED: the unbroadened, assumed-known spectrum
                   for this object. Ideally, this will have rather
                   broader coverage, and higher spectral resolution,
                   than the desired model.

       lam_atmo: wavelength scale of input telluric transmission
                       spectrum. Ideally this is more finely sampled
                       than the desired output wavelength scale.

       atmo: template of telluric transmission. Ideally, this will
                   have higher spectral resolution than the spectrum
                   you wish to model.

       NPW : int
         number of polynomial coefficients for wavelength solution.

       NPC : int
         number of polynomial coefficients for continuum correction.

       npix : int
         Number of pixels in modeled spectrum.

       retlam : bool
         If True, return the tuple (model, spectrum).

       :EXAMPLE:
         ::

            XXX update this for vsini/LLD case!!!

            NPW = 3
            npix = wobs.size
            pix = np.arange(npix, dtype=float)/npix
            wcoef = np.polyfit(pix, wobs, NPW-1)
            ccoef = [1./np.median(template)]
            NPC = len(ccoef)
            guess = np.concatenate(([17, 1e-4, 9, 1], wcoef, ccoef))

            mygmod, mygw = fit_atmo.modelspec_tel_template(guess, lam_template, template, lam_atmo, atmo, NPW, NPC, npix, retlam=True)

       Things like flux conservation and line-broadening are not
       well-treated in this function!
       """
    # 2013-05-08 07:38 IJMC: Created
    # 2013-08-06 10:28 IJMC: Updated to use vsini and LLD
    import pdb

    vsini = params[0]
    lld = params[1]
    rv = params[2]
    instrument_profile_fwhm_kms = params[3] # km/s
    if params[4]<0:
        atmo_scale = 0
    else:
        atmo_scale = params[4]

    if vsini<0:
        vsini = 0

    wavelength_coefs = params[5:5+NPW]
    continuum_coefs = params[5+NPW:5+NPW+NPC]
    
    lam_template = np.array(lam_template, copy=False)
    template = np.array(template, copy=False)
    lam_atmo = np.array(lam_atmo, copy=False)
    atmo = np.array(atmo, copy=False)

    if lam_template.shape!=template.shape or lam_template.ndim!=1:
        return -1
    if lam_atmo.shape!=atmo.shape:
        return -1
        
    pix = np.arange(npix, dtype=float)/npix
    lam = np.polyval(wavelength_coefs, pix)

    # Create model-convolution Kernel and convolve template:
    pixsize_ms = np.diff(lam_template).mean()/lam_template.mean() * const.c
    xkern = np.arange(-int(1200.*vsini/pixsize_ms), int(1200.*vsini/pixsize_ms)+1)
    if xkern.size>=template.size:
        xkern = np.arange(-template.size/2, template.size/2)

    if xkern.size<=1:
        rotational_profile = np.array([1])
    else:
        dv = xkern * pixsize_ms
        rotational_profile = rotationalProfile([vsini*1000., lld, 0], dv)
        rotational_profile /= rotational_profile.sum()
        #kern = gaussian([1., fwhm/2.3548, 0, 0], xkern)

    #pdb.set_trace()
    new_template = np.convolve(template, rotational_profile, 'same')

    # Scale the telluric spectrum.
    new_atmo = 1.0 - atmo_scale*(1.0 - atmo)
    new_atmo[new_atmo<0] = 0.

    # Shift template to specified RV & interpolate to wavelength grid
    #pdb.set_trace()
    new_template = np.interp(lam, lam_template*(1.+rv), new_template, left=0., right=0.)
    output = new_template * np.interp(lam, lam_atmo, new_atmo, left=0., right=0.)

    # Convolve by appropriate Insturment Profile:
    new_pixsize_ms = np.diff(lam).mean()/lam.mean() * const.c
    ip_xkern = np.arange(-int(1200. * instrument_profile_fwhm_kms/new_pixsize_ms), int(1200.*instrument_profile_fwhm_kms/new_pixsize_ms)+1)
    if ip_xkern.size>=output.size:
        ip_xkern = np.arange(-output.size/2, output.size/2)
    if ip_xkern.size<=1:
        instrument_profile = np.array([1])
    else:
        instrument_profile = gaussian([1., (1000./2.3548) * (instrument_profile_fwhm_kms/new_pixsize_ms), 0, 0], ip_xkern)
        instrument_profile /= instrument_profile.sum()

    output = np.convolve(output, instrument_profile, 'same')

    # Multiply by appropriate normalization polynomial
    output *= np.polyval(continuum_coefs, pix)
    output[output < 0] = 0.

    if retlam:
        ret = (output, lam)
    else: 
        ret = output
        
    return ret

def spt_num_to_teff(spt_num, teff_atmo=None):
    """
    Return Teff using Teff_atmo if available; otherwise compute from SpT.
        L dwarfs (0--9):   2220 - 100 * spt_num
        T dwarfs (10--19): 1300 - 75 * (spt_num - 10)
        Y dwarfs (>=20):  500  - 50 * (spt_num - 20)
    """
    if pd.notna(teff_atmo):
        return teff_atmo

    if spt_num < 10:
        return 2220 - 100 * spt_num
    elif spt_num < 20:
        return 1300 - 75 * (spt_num - 10)
    else:
        return 500 - 50 * (spt_num - 20)

def nearest_btsettl_teff(teff): 
    """Return the nearest available BT-Settl grid Teff.""" 
    return min(BT_SETTL_TEFF_GRID, key=lambda t: abs(t - teff))

def btsettl_modelfn(teff, homedir):
    """Construct BT-Settl CIFIST filename for nearest grid Teff."""
    grid_teff = nearest_btsettl_teff(teff)
    return f"{homedir}/uoedrive/data/bt-settl-cifist/lte{grid_teff/100:05.1f}-5.0-0.0a+0.0.BT-Settl.spec.7.dat.txt"

def compute_model_quantities(model):
    '''
    Helper: model-only part of the spectral loop (does not depend on distance).
    Returns per-bin quantities that are purely a function of the BT-Settl model:
        ew                    -- equivalent-width spectrum (A)
        median_atmo           -- median telluric transmission per bin
        telobs_per_photon     -- median(telluric_obs / geometric_scalefactor) per bin,
                                i.e. the photon-flux template before distance scaling
        lo_background_per_bin -- sky background per bin (distance-independent)
    '''
    ew                    = np.zeros(nbins)
    median_atmo           = np.zeros(nbins)
    telobs_per_photon     = np.zeros(nbins)
    lo_background_per_bin = np.zeros(nbins)

    for jj in range(nbins):
        lolim = lam_centers[jj] * (1. - dloglam / 2.)
        hilim = lam_centers[jj] * (1. + dloglam / 2.)

        tind = (model[0] > lolim) & (model[0] < hilim)
        lam_template = model[0, tind]
        template     = model[1, tind]

        aind = (atm0[0] > lolim) & (atm0[0] < hilim)
        true_atmo     = atm0[1, aind]
        true_lam_atmo = atm0[0, aind]
        lam_atmo = lam_template.copy()
        atmo     = np.ones(lam_atmo.size)

        lam_poly = [(hilim - lolim), lolim]
        params   = [vsini, LLD, RV, 3e5 / resolution, telluric_scaling] + lam_poly + [1]
        observation, lam = modelspec_tel_template2(
            params, lam_template, template, lam_atmo, atmo, NPW, NPC, npix, retlam=True)
        telluric_observation, _ = modelspec_tel_template2(
            params, lam_template, template, true_lam_atmo, true_atmo, NPW, NPC, npix, retlam=True)

        goodind = ((observation > 0).nonzero()[0][0],
                   -(observation[::-1] > 0).nonzero()[0][0] - 1)
        lam                  = lam[goodind[0]:goodind[-1]]
        observation          = observation[goodind[0]:goodind[-1]]
        telluric_observation = telluric_observation[goodind[0]:goodind[-1]]
        dlam = np.concatenate([[0], np.diff(lam)])

        # EW — depends only on observation (model spectrum shape), not distance
        cfit = polyfitr(lam, observation, 2, 0.5, clip='below', fev=lam.size * 0.8)
        pspec_cont = np.polyval(cfit, lam)
        pspec_cont[pspec_cont < np.sort(observation)[int(lam.size * 0.01)]] = \
            observation[pspec_cont < np.sort(observation)[int(lam.size * 0.01)]]
        pspec_cont[pspec_cont < observation] = observation[pspec_cont < observation]
        pspec_cont = signal.medfilt(pspec_cont, 101)
        dlam = np.concatenate([[0], np.diff(lam)])
        ew[jj] = (1e4 * dlam * (1. - observation / pspec_cont)).sum()

        # Store telluric flux template normalised out of geometric_scalefactor
        photon_energy = h * c_si / (lam * 1e-6) * 1e7  # erg/photon
        telobs_per_photon[jj] = np.median(
            telluric_observation / photon_energy * itime * area * total_efficiency * dlam)

        # Sky background - does not depend on target distance
        lo_background_flam    = np.interp(lam, radatm0[0], radatm0[1])
        lo_background         = lo_background_flam * slitwidth * pix_arcsec * dlam * 1000 / 1e4
        lo_background_per_bin[jj] = np.median(lo_background * itime * area)

        median_atmo[jj] = np.median(true_atmo)

    return ew, median_atmo, telobs_per_photon, lo_background_per_bin

def compute_distance_quantities(distance, ew, telobs_per_photon, lo_background_per_bin, flux_correction=None):
    '''Helper: distance-dependent quantities, given cached model outputs'''
    geometric_scalefactor = ((radius * rjup) / (distance * pc)) ** 2
    integration = telobs_per_photon * geometric_scalefactor
    if flux_correction is not None:
        integration = integration * flux_correction
    sky_flux = lo_background_per_bin * nrows
    integration = integration * nrows
    snr      = integration / np.sqrt(integration + sky_flux + readnoise**2*nrows*ndit) # snr per extracted spectral pixel by summing nrows
    metric   = snr * ew / 25000
    return metric, snr, integration, sky_flux

def compute_synthetic_mags(model_lam_um, model_flam_cgs, distance, phot_bands):
    """
    Compute synthetic Vega magnitudes by integrating the BT-Settl model SED
    (already distance-scaled) over each photometric band.

    Parameters
    ----------
    model_lam_um   : 1-D array, wavelength grid in µm
    model_flam_cgs : 1-D array, F_lambda in erg/s/cm²/µm  (surface flux)
    distance       : float, distance in pc
    phot_bands     : dict as defined above

    Returns
    -------
    mags : dict  {band_name: Vega_magnitude}
    """
    # Scale surface flux to observed flux at Earth
    geometric_scalefactor = ((radius * rjup) / (distance * pc)) ** 2
    # model[1] is erg/s/cm²/µm from the stellar surface, not per cm² at Earth
    # The BT-Settl files give H_lambda (surface flux density), so we need:
    obs_flam = model_flam_cgs * geometric_scalefactor   # erg/s/cm²/µm at Earth

    mags = {}
    for bn, bp in phot_bands.items():
        ind = (model_lam_um >= bp['lam_lo']) & (model_lam_um <= bp['lam_hi'])
        if ind.sum() < 2:
            mags[bn] = np.nan
            continue
        lam  = model_lam_um[ind]
        flam = obs_flam[ind]
        dlam = np.concatenate([[lam[1] - lam[0]], np.diff(lam)])

        # Convert F_lambda -> F_nu:  F_nu = F_lambda * lambda^2 / c
        # lambda in cm, c in cm/s; result in erg/s/cm²/Hz
        lam_cm = lam * 1e-4            # µm -> cm
        c_cgs  = const.c * 1e2         # m/s -> cm/s
        fnu    = flam * (lam_cm**2 / c_cgs) * 1e4   # extra 1e4: µm->cm in dlam

        # Band-averaged F_nu (simple trapezoidal; good enough for broad bands)
        fnu_mean = np.trapz(fnu, lam) / (bp['lam_hi'] - bp['lam_lo'])

        # Vega magnitude
        mags[bn] = -2.5 * np.log10(fnu_mean / (bp['F0_Jy'] * 1e-23))
    return mags


### Band definitions for metric integration
band_inds  = [0.95, 1.8], [1.8, 2.4], [3.3, 3.6], [4.6, 4.95]
band_names = ['ANDES YJH', 'ANDES K', 'METIS L',  'METIS M']

### CSV columns supplying the peak-to-peak variability amplitude (%) for each band.
band_amp_cols = {'ANDES YJH': 'A_J_fill', 'ANDES K': 'A_K_fill', 'METIS L': 'A_3.6_fill', 'METIS M': 'A_4.5_fill'}

### Photometric bands for magnitude cross-check
phot_bands = {
    'J' : {'lam_lo': 1.14, 'lam_hi': 1.34, 'F0_Jy': 1594.0},
    'H' : {'lam_lo': 1.51, 'lam_hi': 1.78, 'F0_Jy': 1024.0},
    'K' : {'lam_lo': 2.02, 'lam_hi': 2.31, 'F0_Jy':  666.7},
    'L' : {'lam_lo': 2.75, 'lam_hi': 3.87, 'F0_Jy':  309.54},  # WISE W1
    'M' : {'lam_lo': 3.96, 'lam_hi': 5.34, 'F0_Jy':  171.79},  # WISE W2
}

### General (instrument / observation) inputs — unchanged
minlam = 0.785
maxlam = 5.4
dloglam = .02
npix = 2048
vsini = 20.           # km/s
radius = 1.           # Jupiter radii
LLD = 0.3             # linear limb darkening coefficient
RV = 0. / 3e5         # target RV / c
telluric_scaling = 1.
NPW = 2
NPC = 1

amp_1049b = 0.05      # reference amplitude (WISE 1049B) used to normalise metric
fs = 16               # fontsize

# Instrument / observation parameters
itime = 900          # sec
diameter = 3900.      # cm
area = np.pi * (diameter / 2.) ** 2  # cm^2
pix_arcsec = 0.01        # 0.1 arcsec per pixel ## METIS LMS: 0.01"/pix by eye on detector images
slitwidth = 0.04        # 0.2 arcsec ## METIS LMS: 0.021" slice width, but acutally using ~0.01"/pix * 4 pixels in scopesim
resolution = 100000    # spectral resolution
total_efficiency = 0.1
readnoise = 70.      # electrons
nrows = 4            # detector rows spanned by the spectrum (for read noise calculation)
ndit = 3             # number of exposures (for read noise calculation)

### Load input files
vartablefn = 'variablesheet_full_short.csv'
df = pd.read_csv(vartablefn)

telfn    = f'{teldir}/transdata_0,5-5_mic.fits'
atm0    = fits.getdata(telfn).T
#telradfn = 'mauna_kea_emission_h2o=16_airmass=1,5_photon-sec-arcsec2-nm-m2.fits'
#radatm0 = fits.getdata(telradfn)
telradfn = f'{teldir}/eso_armazones_sky_emission_h2o35_am15.dat'
radatm0 = np.loadtxt(telradfn).T

### Initialize wavelength bins
nbins = np.ceil(np.log10(maxlam / minlam) / np.log10(1. + dloglam)).astype(int)
lam_centers = minlam * (1. + dloglam) ** np.arange(nbins)
print(f"nbins: {nbins}\nlam_centers:\n{lam_centers}")

### Calc Teff and select BTSettl model
# sort by model filename so all targets sharing a BT-Settl grid Teff are
# processed consecutively — the expensive spectral loop then runs once per unique model
# rather than once per target.
valid = df.dropna(subset=['spt_num', 'dist_pc']).copy()
skipped = df.index.difference(valid.index)
if len(skipped):
    print(f"Skipping {len(skipped)} rows with missing spt_num or distance: "
          f"{df.loc[skipped, 'Target'].tolist()}")

valid['_modelfn'] = valid.apply(
    lambda row: btsettl_modelfn(spt_num_to_teff(row['spt_num'], row['Teff_atmo']), homedir), axis=1)
valid = valid.sort_values('_modelfn').reset_index()  # original index saved as 'index' column

for bn in band_names:
    df[f'metric_{bn}'] = np.nan
for bn in phot_bands:
    df[f'sim_{bn}mag'] = np.nan

last_modelfn   = None
cached_model_q = None  # (ew, median_atmo, telobs_per_photon, lo_background_per_bin)

### Main loop over targets
for _, row in valid.iterrows():
    orig_idx = row['index']
    target   = row['Target']
    distance = row['dist_pc']
    spt_num  = row['spt_num']
    modelfn  = row['_modelfn']
    Teff     = spt_num_to_teff(spt_num)

    # Load model and run expensive spectral loop only when the grid Teff changes
    if modelfn != last_modelfn:
        print(f"Loading model: {os.path.basename(modelfn)}  "
              f"(Teff={Teff:.0f} K -> grid {nearest_btsettl_teff(Teff)} K)")
        model = np.loadtxt(modelfn).T
        model[0] *= 1e-4   # Angstrom -> um
        model[1] *= 1e4    # erg/s/cm2/A -> erg/s/cm2/um
        cached_model_q = compute_model_quantities(model)
        last_modelfn   = modelfn

    ew, median_atmo, telobs_per_photon, lo_background_per_bin = cached_model_q

    print(f"  {target}  dist={distance:.1f} pc  Teff={Teff:.0f} K")

    amp = {bn: row[band_amp_cols[bn]] / 100.0 for bn in band_names}

    # Synthetic-magnitude cross-check 
    syn_mags = compute_synthetic_mags(model[0], model[1], distance, phot_bands)
    for bn, mag in syn_mags.items():
        df.at[orig_idx, f'sim_{bn}mag'] = mag
    print(f"  Synthetic mags (sim): "
          + "  ".join(f"{bn}={syn_mags[bn]:.2f}" for bn in phot_bands))
    
    # Build per-bin flux correction from synthetic vs observed mags
    flux_correction = np.ones(nbins)
    obs_mag_cols = {'J': 'Jmag', 'H': 'Hmag', 'K': 'Kmag', 'L': 'Lmag', 'M': 'Mmag'}
    for bn, bp in phot_bands.items():
        obs_mag = row[obs_mag_cols[bn]]
        sim_mag = syn_mags[bn]
        if np.isfinite(obs_mag) and np.isfinite(sim_mag):
            correction = 10**((sim_mag - obs_mag) / 2.5)
            ind = (lam_centers >= bp['lam_lo']) & (lam_centers <= bp['lam_hi'])
            flux_correction[ind] = correction
        # else: leave as 1.0 (no correction) for bins where mag data is missing


    metric, median_snr, median_flux, median_skyflux = \
        compute_distance_quantities(distance, ew, telobs_per_photon, lo_background_per_bin,
                                    flux_correction=flux_correction)

    # Compute and store integrated band metrics
    blines = ''
    for bi, bn in zip(band_inds, band_names):
        ind = (lam_centers > bi[0]) & (lam_centers <= bi[1])
        val = np.sqrt((metric[ind] ** 2).sum()) * (amp[bn] / amp_1049b)
        snr = np.nanmedian(median_snr[ind])
        df.at[orig_idx, f'metric_{bn}'] = val
        df.at[orig_idx, f'SNR_{bn}'] = snr
        blines += f'{bn:>4s} band: {val:.2f} integrated metric, SNR = {snr:.2f}\n'
    blines += f'Diameter: {diameter/100.:.1f} m\nDistance: {distance:.1f} pc\n'
    print(blines)



    # --- Diagnostic plots (per target) ---
    xt = [0.6, 0.8, 1, 2, 3, 4, 5]

    # Figure 1: publication-style summary panel
    names_pub = ('metric', 'ew', 'median_snr')
    ylabs_pub = ('Mapping\nSensitivity', '$\\Sigma$ EW $(\\AA)$', 'SNR')
    lims_pub  = ([0, 1.1], [0, 350], [0, 400])
    data_pub  = (metric, ew, median_snr)
    fig2, axs = plt.subplots(len(names_pub), 1, figsize=[4.5, 6])
    axs = axs[::-1]  # bottom panel = index 0, matching original ordering
    for ii, (name, data, ylab, ylim) in enumerate(zip(names_pub, data_pub, ylabs_pub, lims_pub)):
        ax = axs[ii]
        ax.semilogx(lam_centers, data, '-k', drawstyle='steps-mid', linewidth=2)
        ax.set_xticks(xt)
        ax.set_xticklabels(xt if ii == 0 else [])
        ax.set_xlim(min(xt), max(xt))
        #ax.set_ylim(ylim)
        ax.set_ylabel(ylab, fontsize=fs)
        ax.minorticks_on()
        ax.text(0.04, 0.08, f'({"abcde"[len(names_pub) - ii - 1]})',
                fontsize=fs, transform=ax.transAxes)
        if ii == 0:
            ax.set_xlabel('Wavelength ($\\mu$m)', fontsize=fs)
            #ax.text(0.97, 0.94, blines, fontsize=fs*0.4, ha='right', va='top', transform=ax.transAxes)
        for bi in band_inds:
            if min(bi) > 1:
                ax.axvspan(min(bi), max(bi), color='0.8', zorder=0)
    axs[-1].set_title(
        f'{os.path.basename(modelfn).split("-")[0]}, {distance:.1f} pc, {diameter/100.:.1f} m',
        fontsize=fs)
    plt.tight_layout()
    plt.savefig(f'{outdir}/di_metric_{target.replace(" ", "_")}.pdf', bbox_inches='tight')
    plt.close('all')

### Save results
#df.index = df.index + 1
df.to_csv(f'{outdir}/variability_full_with_di_metrics.csv', index=False, float_format='%.3f')
print("\nDone. Results written to variability_full_with_di_metrics.csv")
#print(df[['Target', 'spt_num', 'dist_pc'] + [f'metric_{bn}' for bn in band_names]].to_string())
print(df[['Target', 'spt_num', 'dist_pc']
         + [f'metric_{bn}' for bn in band_names]
         + [f'sim_{bn}mag'   for bn in phot_bands]].to_string())