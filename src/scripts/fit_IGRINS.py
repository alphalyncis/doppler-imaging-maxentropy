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
c = 299792458  # speed of light, m/s

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

def fit_nonstack(target, modelpath, band):
    """
    Fit each observation separately, each with wcoef guess that come from their own wl.
    """
    ###################################
    #  Open IGRINS data
    ###################################
    if target == "W1049B":
        filelist = sorted(glob.glob(f'{homedir}/uoedrive/data/IGRINS/SDCK*_1f.spec.fits'))
        fluxes = []
        wls = []
        for filename in filelist:
            wlname = filename.split('_1f')[0]+'.wave.fits'

            flux = fits.getdata(filename)
            wl = fits.getdata(wlname)

            # trim first and last 100 columns
            flux = flux[:, 100:1948]
            wl = wl[:, 100:1948]
            
            fluxes.append(flux)
            wls.append(wl) 

    if target == "2M0036":
        filelist = sorted(glob.glob(f'{homedir}/uoedrive/data/IGRINS_{target}/SDC{band}*.spec_a0v.fits'))
        fluxes = []
        wls = []
        for filename in filelist:
            hdu = fits.open(filename)
            flux = hdu[0].data
            wl = hdu[1].data

            # trim first and last 100 columns
            flux = flux[:, 100:1948]
            wl = wl[:, 100:1948]

            fluxes.append(flux)
            wls.append(wl)

    ###############################
    ## Collapse data over the whole observation into one mean, smoothed spectrum
    ###############################

    dims = np.array(fluxes).shape
    fluxes = np.array(fluxes)
    wls = np.array(wls)

    # remove first order with all NaNs when making various arrays
    obs0 = fluxes[:, 1:, :]*dims[0]
    eobs0 = fluxes[:, 1:, :]*np.sqrt(dims[0])  # make noise spectrum (assuming just photon noise)
    fobs0 = np.array([[signal.medfilt(obs0[obs,jj], 3) for jj in range(dims[1]-1)] for obs in range(dims[0])])
    fobs0 /= np.array([np.nanmedian(fobs0[obs], axis=1) for obs in range(dims[0])]).reshape(dims[0], dims[1]-1, 1)
    eobs0 /= np.array([np.nanmedian(fobs0[obs], axis=1) for obs in range(dims[0])]).reshape(dims[0], dims[1]-1, 1)
    wfobs0 = 1./eobs0**2   # make noise spectrum
    wind = np.isinf(wfobs0)  # remove points with infinite values
    wfobs0[wind] = 0.

    # fix wavelength array to have same shape
    wls = wls[:, 1:24, :] 

    ##########################
    # Open model
    ##########################

    if "BTSettl" in modelpath:
        model = Table.read(modelpath, format='fits')
        modelname = modelpath.split("/")[-1]
        model['wl'] = model['Wavelength']
        model['flux'] = model['Flux']

    if "Callie" in modelpath:
        model = fits.getdata(modelpath)
        modelname = modelpath.split("/")[-1]

    ##########################
    ## Fitting model
    ##########################
    nobs = wls.shape[0]
    norders = 20  # skip last few non-fitting orders for now
    npix = dims[2] 

    NPW = 4  # number of terms for polynomial fit to the wavelengths
    pix = np.arange(npix, dtype=float)/npix
    chipfits = []
    chipmods = np.zeros((nobs, norders, npix), dtype=float)
    chiplams = np.zeros((nobs, norders, npix), dtype=float)
    chipmodnobroad = np.zeros((nobs, norders, npix), dtype=float)
    chipguesses = np.zeros((nobs, norders, npix), dtype=float)
    chisqarr = np.zeros((nobs, norders), dtype=float)

    # set up tables for best fit vsini, limb darkening values, and rv
    orderval=[]
    obsval=[]
    vsini = []
    lld = []
    rv = []
    wcoefs = []
    ccoefs = []
    chisq = []

    for jj in (np.arange(norders)):
        print(f"Current fitting: model {modelname}, order {jj}.")
        lolim = wls[:, jj, :].min() - 0.003
        hilim = wls[:, jj, :].max() + 0.003
        tind = (model['wl']>lolim) * (model['wl'] < hilim)
        lam_template = model['wl'][tind]
        template = model['flux'][tind]
        template /= np.median(template)
        
        # to be compatible with rotational profile convolution kernel
        if len(lam_template) < 400:
           new_lam = np.linspace(lam_template[0], lam_template[-1], 400)
           template = np.interp(new_lam, lam_template, template)
           lam_template = new_lam

        for obs in np.arange(nobs):
            wcoef = np.polyfit(pix, wls[obs, jj, :], NPW-1)
            # fit continuum coefficients / flux scaling
            ccoef = [-0.1, 1.2/np.median(template)]
            NPC = len(ccoef)
            ind90 = np.sort(fobs0[obs, jj])[int(0.9*npix)]  
            ccoef = np.polyfit(pix[fobs0[obs,jj]>ind90], fobs0[obs,jj][fobs0[obs,jj]>ind90], NPC-1)

            guess = np.concatenate(([21, 0.3, 9e-5], wcoef, ccoef))
            fitargs = (mf.modelspec_template, lam_template, template, NPW, NPC, npix, fobs0[obs,jj], wfobs0[obs,jj])
            fit = mf.fmin(mf.errfunc, guess, args=fitargs, full_output=True, disp=True, maxiter=10000, maxfun=100000)
            print("fitted params:", fit[0])
            print("chisq:", fit[1])
            mymod, myw = mf.modelspec_template(fit[0], *fitargs[1:-2], retlam=True)
            chipfits.append(fit)
            chipmods[obs,jj] = mymod
            chiplams[obs,jj] = myw

            # save best parameters
            orderval.append(jj)
            obsval.append(obs)
            vsini.append(fit[0][0])
            lld.append(fit[0][1])
            rv.append(fit[0][2])
            wcoefs.append(fit[0][3:7])
            ccoefs.append(fit[0][7:])
            chisq.append(fit[1])

            chisqarr[obs,jj] = fit[1]

    ##########################
    ## Save result
    ##########################

    # make table of best parameters
    results = Table()
    results['order'] = orderval
    results['obs'] = obsval
    results['chisq'] = chisq
    results['vsini'] = vsini
    results['lld'] = lld
    results['rv'] = rv
    results['wcoef'] = [f"{wcoef[0]}, {wcoef[1]}, {wcoef[2]}, {wcoef[3]}" for wcoef in wcoefs]
    results['ccoef'] = [f"{ccoef[0]}, {ccoef[1]}" for ccoef in ccoefs]


    if "BTSettl" in modelpath:
        resultdir = f"{homedir}/uoedrive/result/CIFIST"
    if "Callie" in modelpath:
        resultdir = f"{homedir}/uoedrive/result/Callie"
    results.write(f'{resultdir}/IGRINS_{target}_{band}_nonstack_fitting_results_{modelname[:12]}.txt', format='ascii', overwrite=True)
    fits.writeto(f'{resultdir}/IGRINS_{target}_{band}_nonstack_chipmods_{modelname[:12]}.fits', chipmods, overwrite=True)
    fits.writeto(f'{resultdir}/IGRINS_{target}_{band}_nonstack_chiplams_{modelname[:12]}.fits', chiplams, overwrite=True)

def fit_binned(target, modelpath, band, telluricpath):
    """
    Bin obs in 14/7 groups, fit each group.
    """
    ###################################
    #  Open IGRINS data
    ###################################
    tag = "binned"
    fluxes = []
    wls = []
    sns = []
    if "W1049B" in target:
        binsize = 4
        filelist = sorted(glob.glob(
            f'{homedir}/uoedrive/data/IGRINS_{target}/SDC{band}*_1f.spec.fits'
        ))
        for filename in filelist:
            wlname = filename.split('_1f')[0]+'.wave.fits'
            snname = filename.split('_1f')[0]+'_1f.sn.fits'

            flux = fits.getdata(filename)
            wl = fits.getdata(wlname)
            sn = fits.getdata(snname)

            # trim first and last 100 columns
            flux = flux[:, 100:1948]
            wl = wl[:, 100:1948]
            sn = sn[:, 100:1948]
            
            fluxes.append(flux)
            wls.append(wl) 
            sns.append(sn)

    elif "W1049A" in target:
        binsize = 4
        filelist = sorted(glob.glob(
            f'{homedir}/uoedrive/data/IGRINS_{target}/SDC{band}*_2f.spec.fits'
        ))
        for filename in filelist:
            wlname = filename.split('_2f')[0]+'.wave.fits'
            snname = filename.split('_2f')[0]+'_2f.sn.fits'

            flux = fits.getdata(filename)
            wl = fits.getdata(wlname)
            sn = fits.getdata(snname)

            # trim first and last 100 columns
            flux = flux[:, 100:1948]
            wl = wl[:, 100:1948]
            sn = sn[:, 100:1948]
            
            fluxes.append(flux)
            wls.append(wl)
            sns.append(sn)

    if "2M0036" in target:
        binsize = 2
        filelist = sorted(glob.glob(f'{homedir}/uoedrive/data/IGRINS_{target}/SDC{band}*.spec_a0v.fits'))
        for filename in filelist:
            hdu = fits.open(filename)
            flux = hdu[0].data
            wl = hdu[1].data
            sn = hdu[5].data

            # trim first and last 100 columns
            flux = flux[:, 100:1948]
            wl = wl[:, 100:1948]
            sn = sn[:, 100:1948]

            fluxes.append(flux)
            wls.append(wl)
            sns.append(sn)

    ###############################
    ## Collapse data over the whole observation into one mean, smoothed spectrum
    ###############################

    fluxes = np.array(fluxes)
    wls = np.array(wls)
    print(fluxes.shape)
    print(wls.shape)
    sns = np.array(sns)
    print(sns.shape)

    # collapse every 4/2 obs to 1
    fluxes_binned = np.median(fluxes.reshape(int(fluxes.shape[0]/binsize), binsize, fluxes.shape[1], fluxes.shape[2]), axis=1)
    wls_binned = np.median(wls.reshape(int(wls.shape[0]/binsize), binsize, wls.shape[1], wls.shape[2]), axis=1)
    dims = fluxes_binned.shape
    nobs = dims[0]
    obs0 = fluxes_binned[:, 1:, :] # remove first order with all NaNs
    sns_binned = np.median(sns.reshape(int(sns.shape[0]/binsize), binsize, sns.shape[1], sns.shape[2]), axis=1)
    eobs0 = fluxes_binned[:, 1:, :]/sns_binned[:, 1:, :]  # make noise spectrum (assuming just photon noise)
    cobs0 = np.array([[remove_spike(obs0[obs,jj]) for jj in range(dims[1]-1)] for obs in range(dims[0])])
    fobs0 = np.array([[signal.medfilt(cobs0[obs,jj], kernel_size=3) for jj in range(dims[1]-1)] for obs in range(dims[0])])
    fobs0 /= np.array([np.nanmedian(fobs0[obs], axis=1) for obs in range(dims[0])]).reshape(dims[0], dims[1]-1, 1)
    eobs0 /= np.array([np.nanmedian(fobs0[obs], axis=1) for obs in range(dims[0])]).reshape(dims[0], dims[1]-1, 1)
    wfobs0 = 1./eobs0**2   # make noise spectrum
    wind = np.isinf(wfobs0)  # remove points with infinite values
    wfobs0[wind] = 0.

    # fix wavelength array to have same shape
    wls = wls_binned[:, 1:24, :] 
    if (nobs != wls.shape[0]) or (dims[2] != wls.shape[2]):
        raise ValueError("Dimensions don't match!")
    print(f"nobs: {nobs}, norder:{wls.shape[1]}, npix: {dims[2]}")

    ##########################
    # Open model
    ##########################

    if "BTSettl" in modelpath:
        model = Table.read(modelpath, format='fits')
        modelname = modelpath.split("/")[-1]
        model['wl'] = model['Wavelength']
        model['flux'] = model['Flux']

    if "Callie" in modelpath:
        model = fits.getdata(modelpath)
        modelname = modelpath.split("/")[-1]

    atm0 = fits.getdata(telluricpath)

    ##########################
    ## Fitting model
    ##########################
    norders = 20  # skip last few non-fitting orders for now
    npix = dims[2] 

    NPW = 4  # number of terms for polynomial fit to the wavelengths
    pix = np.arange(npix, dtype=float)/npix
    chipmods = np.zeros((nobs, norders, npix), dtype=float)
    chiplams = np.zeros((nobs, norders, npix), dtype=float)
    chipmodnobroad = np.zeros((nobs, norders, npix), dtype=float)
    chipcors = np.zeros((nobs, norders, npix), dtype=float)

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
    for jj in (np.arange(norders)):
        lolim = wls[:, jj, :].min() - 0.003
        hilim = wls[:, jj, :].max() + 0.003
        tind = (model['wl']>lolim) * (model['wl'] < hilim)

        lam_template = model['wl'][tind]
        template = model['flux'][tind]
        template /= np.median(template)
        
        # to be compatible with rotational profile convolution kernel
        if len(lam_template) < 400:
           new_lam = np.linspace(lam_template[0], lam_template[-1], 400)
           template = np.interp(new_lam, lam_template, template)
           lam_template = new_lam

        chipfits = []
        for kk in np.arange(nobs):
            print(f"Current fitting: model {modelname}, obs{kk}, order {jj}.")
            wcoef = np.polyfit(pix, wls[kk, jj, :], NPW-1)
            # fit continuum coefficients / flux scaling
            ccoef = [-0.1, 1.2/np.median(template)]
            NPC = len(ccoef)
            ind90 = np.sort(fobs0[kk, jj])[int(0.9*npix)]  
            ccoef = np.polyfit(pix[fobs0[kk,jj]>ind90], fobs0[kk,jj][fobs0[kk,jj]>ind90], NPC-1)

            # start fit
            guess = np.concatenate(([21, 0.3, 9e-5], ccoef))
            bounds = [[0, 100], [0, 1], [0, 100/c], [-10, 10], [-10, 10]]
            tag = "bounds"
            fitargs = (mf.modelspec_template, lam_template, template, wcoef, NPC, npix, fobs0[kk,jj], wfobs0[kk,jj], dict(uniformprior=bounds))
            fit = mf.fmin(mf.errfunc, guess, args=fitargs, full_output=True, disp=True, maxiter=5000, maxfun=100000)
            mymod, myw = mf.modelspec_template(fit[0], lam_template, template, wcoef, NPC, npix, retlam=True)

            print("fitted params:", fit[0])
            print("chisq:", fit[1])

            chipfits.append(fit)
            chipmods[kk,jj] = mymod
            chiplams[kk,jj] = myw
            #chipcors[kk, jj] = mycor

            # save best parameters
            orderval.append(jj)
            obsval.append(kk)
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
            chipmodnobroad[kk, jj] = mymodnobroad

        individual_fits.append(chipfits)

    ##########################
    ## Save result
    ##########################

    # make table of best parameters
    results = Table()
    results['order'] = orderval
    results['obs'] = obsval
    results['chisq'] = chisq
    results['vsini'] = vsini
    results['lld'] = lld
    results['rv'] = rv
    results['wcoef'] = [f"{wcoef[0]}, {wcoef[1]}, {wcoef[2]}, {wcoef[3]}" for wcoef in wcoefs]
    results['ccoef'] = [f"{ccoef[0]}, {ccoef[1]}" for ccoef in ccoefs]


    if "BTSettl" in modelpath:
        resultdir = f"{homedir}/uoedrive/result/CIFIST"
    if "Callie" in modelpath:
        resultdir = f"{homedir}/uoedrive/result/Callie"
    results.write(f'{resultdir}/IGRINS_{target}_{band}_{tag}_fitting_results_{modelname[:12]}.txt', format='ascii', overwrite=True)
    fits.writeto(f'{resultdir}/IGRINS_{target}_{band}_{tag}_chipmods_{modelname[:12]}.fits', chipmods, overwrite=True)
    fits.writeto(f'{resultdir}/IGRINS_{target}_{band}_{tag}_chiplams_{modelname[:12]}.fits', chiplams, overwrite=True)
    fits.writeto(f'{resultdir}/IGRINS_{target}_{band}_{tag}_chipcors_{modelname[:12]}.fits', chipcors, overwrite=True)
    fits.writeto(f'{resultdir}/IGRINS_{target}_{band}_{tag}_chipmodnobroad_{modelname[:12]}.fits', chipmodnobroad, overwrite=True)
    
    # save to pickle
    saveout = dict(chipmods=chipmods, chiplams=chiplams, bs0=obs0, eobs0=eobs0, fobs0=fobs0, wfobs0=wfobs0, wobs=wls, modelfn=modelname, individual_fits=individual_fits, chipmodnobroad=chipmodnobroad)
    with open(f'{resultdir}/IGRINS_{target}_{band}_{modelname[:12]}.pickle', 'wb') as f:
        pickle.dump(saveout, f)

    return fit[0], lam_template, template, wcoef, NPC, npix

if __name__ == "__main__":
    telluric = f"{homedir}/uoedrive/data/telluric/transdata_0,5-14_mic_hires.fits"
    try:
        target = sys.argv[1] # "W1049B" or "2M0036_2020110x"
    except:
        raise NameError("Please provide a target ('W1049A', 'W1049B' or '2M0036').")
    try:
        band = sys.argv[2] # "K" or "H"
        try:
            band in ["K", "H"]
        except:
            raise NotImplementedError("Band must be 'K' or 'H' ")
    except:
        raise NameError("Please provide a band 'K' or 'H'.")
    try:
        modeldir = sys.argv[3] # e.g. BTSettlModels/CIFIST2015; CallieModels
        try:
            os.path.exists(f"{homedir}/uoedrive/data/{modeldir}")
        except:
            raise FileNotFoundError(f"Path {homedir}/uoedrive/data/{modeldir} dose not exist.")
    except:
        raise NameError("Please provide a model directory under 'home/uoedrive/data/'.")

    modellist = sorted(glob.glob(f"{homedir}/uoedrive/data/{modeldir}/*.fits"))
    if len(modellist) == 0:
        raise NameError("model dir is not valid.")
        
    for model in modellist:
        print(f"***Running fit to model {model}***")
        fitres, lam_template, template, wcoef, NPC, npix = fit_binned(target=target, modelpath=model, band=band, telluricpath=telluric)