import numpy as np
from astropy.table import Table
import modelfitting as mf
import sys
from scipy import signal
import pickle
import glob
from astropy.io import fits
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import os
homedir = os.path.expanduser('~')
c_km_s = 299792.458  # speed of light, km/s


def fit_nonstack(target, modelpath, band):
    """
    Fit each observation separately, each with wcoef guess that come from their own wl.
    """
    ###################################
    #  Open IGRINS data
    ###################################
    if target == "W1049B":
        datafile = f'../data/fitted/METIS_{target}_{band}_gcm.pickle'
        print(f"Fitting observed data {datafile}")
        with open(datafile, 'rb') as f:
            data = pickle.load(f)
            lams = data['chiplams']
            observed = data['observed']
            error = data['error']


    ##########################
    # Open model
    ##########################

    if "diamondback" in modelpath:
        model = np.loadtxt(modelpath, skiprows=4)
        modwl = model[:,0][::-1]
        modflux = model[:,1][::-1]


    ##########################
    ## Fitting model
    ##########################
    nobs, nchip, npix = observed.shape 

    NPW = 4  # number of terms for polynomial fit to the wavelengths
    x = np.arange(npix, dtype=float)/npix
    chipfits = []
    chipmods = np.zeros((nobs, nchip, npix), dtype=float)
    chiplams = np.zeros((nobs, nchip, npix), dtype=float)
    chipmodnobroad = np.zeros((nobs, nchip, npix), dtype=float)
    chisqarr = np.zeros((nobs, nchip), dtype=float)

    # set up tables for best fit vsini, limb darkening values, and rv
    orderval=[]
    obsval=[]
    vsini = []
    lld = []
    rv = []
    wcoefs = []
    ccoefs = []
    chisq = []
    individual_fits = []

    for c in (np.arange(nchip)):
        modelname = modelpath.split('/')[-1]
        print(f"Current fitting: model {modelname}, order {c}.")
        lolim = lams[:, c, :].min() - 0.003
        hilim = lams[:, c, :].max() + 0.003
        tind = np.where((modwl>lolim) & (modwl < hilim))[0]
        print(lolim, hilim)
        lam_template = modwl[tind]
        template = modflux[tind]
        template /= np.median(template)

        print(lam_template.shape, template.shape)
        
        # to be compatible with rotational profile convolution kernel
        if len(lam_template) < 400:
           new_lam = np.linspace(lam_template[0], lam_template[-1], 400)
           template = np.interp(new_lam, lam_template, template)
           lam_template = new_lam

        chipfits = []
        for t in np.arange(nobs):
            wcoef = np.polyfit(x, lams[t, c, :], NPW-1)
            # fit continuum coefficients / flux scaling
            ccoef = [-0.1, 1.2/np.median(template)]
            NPC = len(ccoef)
            ind90 = np.sort(observed[t, c])[int(0.9*npix)]  
            ccoef = np.polyfit(x[observed[t,c]>ind90], observed[t,c][observed[t,c]>ind90], NPC-1)

            guess = np.concatenate(([21, 0.3, 9e-5], ccoef))
            #print("---Intial input---\n", guess, lam_template[-1], template[-1], wcoef, NPC, npix)
            #print("---Intial eval---\n", mf.modelspec_template(guess, lam_template, template, wcoef, NPC, npix))
            bounds = [
                [0, 50],   # vsini
                [0, 0.5],     # lld
                [0, 100/c_km_s], # rv
                [-10, 10], 
                [-10, 10]]
            fitargs = (mf.modelspec_template, lam_template, template, wcoef, NPC, npix, observed[t,c], error[t,c], dict(uniformprior=bounds))
            fit = mf.fmin(mf.errfunc, guess, args=fitargs, full_output=True, disp=True, maxiter=10000, maxfun=100000)
            mymod, myw = mf.modelspec_template(fit[0], lam_template, template, wcoef, NPC, npix, retlam=True)

            print("fitted params:", fit[0])
            print("chisq:", fit[1])

            chipfits.append(fit)
            chipmods[t,c] = mymod
            chiplams[t,c] = myw

            # save best parameters
            orderval.append(c)
            obsval.append(t)
            vsini.append(fit[0][0])
            lld.append(fit[0][1])
            rv.append(fit[0][2])
            wcoefs.append(wcoef)
            ccoefs.append(fit[0][3:])
            chisq.append(fit[1])

            # make non-broadened model
            fitnobroad = fit[0].copy()
            fitnobroad[0:2] = 0.
            mymodnobroad = mf.modelspec_template(fitnobroad, *fitargs[1:-3])
            chipmodnobroad[t,c] = mymodnobroad

        individual_fits.append(chipfits)

    ##########################
    ## Save result
    ##########################

    # make table of best parameters
    results = Table()
    results['order'] = orderval
    results['obs'] = obsval
    results['chisq'] = [f"{i:.2f}" for i in chisq]
    results['vsini'] = [f"{i:.2f}" for i in vsini]
    results['lld'] = [f"{i:.2f}" for i in lld]
    results['rv'] = [f"{i:.2e}" for i in rv]
    #results['wcoef'] = [f"{wcoef[0]}, {wcoef[1]}, {wcoef[2]}, {wcoef[3]}" for wcoef in wcoefs]
    #results['ccoef'] = [f"{ccoef[0]}, {ccoef[1]}" for ccoef in ccoefs]


    if "diamondback" in modelpath:
        resultdir = f"{homedir}/uoedrive/result/fitted/diamondback"

    results.write(f'{resultdir}/METIS_{target}_{band}_fitting_results_{modelname[:12]}.txt', format='ascii', overwrite=True)
    #fits.writeto(f'{resultdir}/METIS_{target}_{band}_chipmods_{modelname[:12]}.fits', chipmods, overwrite=True)
    #fits.writeto(f'{resultdir}/METIS_{target}_{band}_chiplams_{modelname[:12]}.fits', chiplams, overwrite=True)
    #fits.writeto(f'{resultdir}/METIS_{target}_{band}_chipmodnobroad_{modelname[:12]}.fits', chipmodnobroad, overwrite=True)

    # save to pickle
    saveout = dict(chipmods=chipmods, chiplams=chiplams, observed=observed, error=error, modelfn=modelname, chipmodnobroad=chipmodnobroad, individual_fits=individual_fits)
    with open(f'{resultdir}/METIS_{target}_{band}_{modelname[:12]}.pickle', 'wb') as f:
        pickle.dump(saveout, f)

    return fit[0], lam_template, template, wcoef, NPC, npix

if __name__ == "__main__":
    target = "W1049B"
    band = "K"
    modellist = sorted(glob.glob(f"{homedir}/uoedrive/data/diamondback/spectra/t1*g1000*_m0.0_co1.0.spec"))

    for model in modellist:
        print(f"***Running fit to model {model}***")
        fitres, lam_template, template, wcoef, NPC, npix = fit_nonstack(target=target, modelpath=model, band=band)

    print(f"Fitting done for {target} {band}")