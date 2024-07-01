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

def fit_binned_telluric(target, modelpath, telluricpath, band):
    """
    Fit averaged observations with the same guess for wcoef (fitted using wl of first/avg order) 
    for all observations, i.e., one fit per order.
    """
    ###################################
    ##  Open CRIRES data
    ###################################
    tag = "binned"

    if target == "W1049B":
        filename = f'{homedir}/uoedrive/data/CRIRES/fainterspectral-fits_6.pickle'
        with open(filename, 'rb') as f:
            ret = pickle.load(f, encoding="latin1")
    elif target == "W1049A":
        filename = f'{homedir}/uoedrive/data/CRIRES/brighterspectral-fits_6.pickle'
        with open(filename, 'rb') as f:
            ret = pickle.load(f, encoding="latin1")
    else:
        raise NotImplementedError("No other data.")

    ###############################
    ## Collapse data over the whole observation into one mean, smoothed spectrum
    ###############################
    nobs = 14
    obs1 = ret['obs1']
    wobs0 = ret['wobs0']

    ##########################
    ## Open model
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
    ## Open telluric file
    ##########################

    atm0 = fits.getdata(telluricpath)

    ##########################
    ## Fitting model
    ##########################
    NPW = 4
    npix = ret['wobs'].shape[1]
    pix = np.arange(npix, dtype=float)/npix
    chipmods = np.zeros((14, 4, 1024), dtype=float)
    chiplams = np.zeros((14, 4, 1024), dtype=float)
    chipmodnobroad = np.zeros((14, 4, 1024), dtype=float)
    chipcors = np.zeros((14, 4, 1024), dtype=float)
    chipfits = []

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
    for kk in range(nobs):
        chipfits = []
        for jj in range(4):
            print(f"Current fitting: model {modelname}, obs{kk}, order {jj}.")
            lolim = ret['wobs'][jj].min() - 0.003
            hilim = ret['wobs'][jj].max() + 0.003
            tind = (model['wl']>lolim) * (model['wl'] < hilim)
            aind = (atm0[:, 0]>lolim) * (atm0[:, 0] < hilim)

            lam_template = model['wl'][tind]
            template = model['flux'][tind]
            template /= np.median(template)

            lam_atmo = atm0[aind, 0]
            atmo = atm0[aind, 1]
            '''
            if kk==0 and jj==0:
                plt.figure(figsize=(10,4))
                plt.plot(lam_template, template)
                plt.plot(lam_atmo, atmo)
                plt.plot(ret['wobs'][jj], obs1[kk, jj])'''
            wcoef = np.polyfit(pix, ret['wobs'][jj], NPW-1)
            ccoef = [-0.1, 1.2/np.median(template)]
            NPC = len(ccoef)
            ind90 = np.sort(obs1[kk, jj])[int(0.9*npix)]  
            ccoef = np.polyfit(pix[obs1[kk, jj]>ind90], obs1[kk, jj][obs1[kk, jj]>ind90], NPC-1)

            # start fit
            guess = np.concatenate(([25, 0.3, 9e-5, 30, 1.3], ccoef))
            bounds = [[0, 100], [0, 1], [0, 100/c], [0, np.inf], [0, np.inf], [-10, 10], [-10, 10]]
            tag = "bounds"
            fitargs = (mf.modelspec_tel_template, lam_template, template, lam_atmo, atmo, wcoef, NPC, npix, obs1[kk, jj], wobs0[kk, jj], dict(uniformprior=bounds))  # with telluric correction
            fit = mf.fmin(mf.errfunc, guess, args=fitargs, full_output=True, disp=False, maxiter=5000, maxfun=100000)
            mymod, myw = mf.modelspec_tel_template(fit[0], lam_template, template, lam_atmo, atmo, wcoef, NPC, npix, retlam=True)
            mycor = mf.modelspec_tel_template(fit[0], lam_template, np.ones(template.size), lam_atmo, atmo, wcoef, NPC, npix, retlam=False)
            print("fitted params:", fit[0])
            print("chisq:", fit[1])

            chipfits.append(fit)
            chipmods[kk, jj] = mymod
            chiplams[kk, jj] = myw
            chipcors[kk, jj] = mycor

            # save best parameters
            orderval.append(jj)
            obsval.append(kk)
            vsini.append(fit[0][0])
            lld.append(fit[0][1])
            rv.append(fit[0][2])
            wcoefs.append(wcoef)
            ccoefs.append(fit[0][5:])
            chisq.append(fit[1])

            # make non-broadened model
            fitnobroad = fit[0].copy()
            fitnobroad[0:2] = 0.
            mymodnobroad = mf.modelspec_tel_template(fitnobroad, lam_template, template, lam_atmo, atmo, wcoef, NPC, npix, retlam=False, verbose=True)
            print("chipmodnobroad:", mymodnobroad)
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
    #results['wcoef'] = [f"{wcoef[0]}, {wcoef[1]}, {wcoef[2]}, {wcoef[3]}" for wcoef in wcoefs]
    #results['ccoef'] = [f"{ccoef[0]}, {ccoef[1]}" for ccoef in ccoefs]

    if "BTSettl" in modelpath:
        resultdir = f"{homedir}/uoedrive/result/CIFIST"
    if "Callie" in modelpath:
        resultdir = f"{homedir}/uoedrive/result/Callie"
    results.write(f'{resultdir}/CRIRES_{target}_{band}_{tag}_fitting_results_{modelname[:12]}.txt', format='ascii', overwrite=True)
    fits.writeto(f'{resultdir}/CRIRES_{target}_{band}_{tag}_chipmods_{modelname[:12]}.fits', chipmods, overwrite=True)
    fits.writeto(f'{resultdir}/CRIRES_{target}_{band}_{tag}_chiplams_{modelname[:12]}.fits', chiplams, overwrite=True)
    fits.writeto(f'{resultdir}/CRIRES_{target}_{band}_{tag}_chipmodnobroad_{modelname[:12]}.fits', chipmodnobroad, overwrite=True)
    fits.writeto(f'{resultdir}/CRIRES_{target}_{band}_{tag}_chipcors_{modelname[:12]}.fits', chipcors, overwrite=True)

    # save to pickle
    saveout = dict(chipmods=chipmods, chiplams=chiplams, chipcors=chipcors, obs1=obs1, wobs0=wobs0, wobs=ret['wobs'], modelfn=modelname, individual_fits=individual_fits, chipmodnobroad=chipmodnobroad)
    with open(f'{resultdir}/CRIRES_{target}_{band}_{modelname[:12]}.pickle', 'wb') as f:
        pickle.dump(saveout, f)

if __name__ == "__main__":
    #target = "W1049B"
    band = "K"
    telluric = f"{homedir}/uoedrive/data/telluric/transdata_0,5-14_mic_hires.fits"
    try:
        modeldir = sys.argv[1] # e.g. BTSettlModels/CIFIST2015; CallieModels
        try:
            os.path.exists(f"{homedir}/uoedrive/data/{modeldir}")
        except:
            raise FileNotFoundError(f"Path {homedir}/uoedrive/data/{modeldir} dose not exist.")
    except:
        raise NameError("Please provide a model directory under 'home/uoedrive/data/'.")
    
    try:
        target = sys.argv[2] # e.g. W1049B, W1049A
    except:
        raise NameError("Please provide a target name.")

    modellist = sorted(glob.glob(f"{homedir}/uoedrive/data/{modeldir}/*.fits"))
    for model in modellist:
        print(f"***Running fit to model {model}***")
        fit_binned_telluric(target=target, modelpath=model, telluricpath=telluric, band=band)