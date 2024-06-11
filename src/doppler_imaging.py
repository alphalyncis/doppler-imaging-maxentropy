# Functions for solving doppler maps using different pipelines.
# Xueqing Chen 24.04.2023
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import starry
import pickle
import os
from astropy.io import fits
import scipy.constants as const
from scipy import interpolate
from scipy.signal import savgol_filter
from scipy.ndimage.filters import gaussian_filter
import paths

import pymc3 as pm
import pymc3_ext as pmx
import theano.tensor as tt
from tqdm import tqdm
from matplotlib.colors import Normalize
import emcee

import modelfitting as an # for an.lsq and an.gfit
import ELL_map_class as ELL_map
from dime import MaxEntropy # Doppler Imaging & Maximum Entropy, needed for various funcs
import cartopy.crs as ccrs
from PIL import Image

#TODO: target should name as -> target+night, since can have several nights for one target 
#TODO: run bands separately or together
#TODO: test more parameters for starry solver
#TODO: test sampling rate

################################################################################
####################   Methods for workflow    #################################
################################################################################

def load_data(model_datafile, instru, nobs, goodchips, use_toy_spec=False):
    global npix, npix0, flux_err
    # Load model and data
    with open(model_datafile, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    lams = data["chiplams"][0] # in um
    nchip = len(goodchips)
    npix = lams.shape[1]
    print(f"nobs: {nobs}, nchip: {nchip}, npix: {npix}")

    observed = np.empty((nobs, nchip, npix))
    template = np.empty((nobs, nchip, npix))
    residual = np.empty((nobs, nchip, npix))
    error = np.empty((nobs, nchip, npix))

    if instru == "IGRINS":
        for k in range(nobs):
            for i, jj in enumerate(goodchips):
                observed[k, i] = np.interp(
                    lams[jj], 
                    data["chiplams"][k][jj],
                    data["fobs0"][k][jj] #/ data["chipcors"][k][c+firstchip],
                )
                template[k, i] = np.interp(
                    lams[jj],
                    data["chiplams"][k][jj],
                    data["chipmodnobroad"][k][jj] #/ data["chipcors"][k][c+firstchip]
                )
                residual[k, i] = np.interp(
                    lams[jj], 
                    data["chiplams"][k][jj],
                    data["fobs0"][k][jj] - data["chipmods"][k][jj]
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
                residual[k, i] = np.interp(
                    lams[jj], 
                    data["chiplams"][k][jj],
                    data["obs1"][k][jj] - data["chipmods"][k][jj]
                )

    pad = 100
    npix0 = len(lams[0])
    npix = len(lams[0][pad:-pad])
    wav_nm = np.zeros((nchip, npix))
    wav0_nm = np.zeros((nchip, npix0))
    mean_spectrum = np.zeros((nchip, npix0))
    observed = observed[:, :, pad:-pad]
    flux_err = eval(f'{error.mean():.3f}') if instru =="IGRINS" else 0.02

    for i, jj in enumerate(goodchips):
        wav_nm[i] = lams[jj][pad:-pad] * 1000 # um to nm
        wav0_nm[i] = lams[jj] * 1000 # um to nm
        if use_toy_spec:
            toy_spec = (
                1.0
                - 0.99 * np.exp(-0.5 * (wav0_nm[i] - 2330) ** 2 / 0.03 ** 2)
                - 0.99 * np.exp(-0.5 * (wav0_nm[i] - 2335) ** 2 / 0.03 ** 2)
                - 0.99 * np.exp(-0.5 * (wav0_nm[i] - 2338) ** 2 / 0.03 ** 2)
                - 0.99 * np.exp(-0.5 * (wav0_nm[i] - 2345) ** 2 / 0.03 ** 2)
                - 0.99 * np.exp(-0.5 * (wav0_nm[i] - 2347) ** 2 / 0.03 ** 2)
            )
            for k in range(nobs):
                template[k, i] = toy_spec
            mean_spectrum[i] = toy_spec
            flux_err = 0.002
        mean_spectrum[i] = np.mean(template[:, i], axis=0)

    print("mean_spectrum:", mean_spectrum.shape)
    print("template:", template.shape)
    print("observed:", observed.shape)
    print(f"wav: {wav_nm.shape}, wav0: {wav0_nm.shape}")

    return mean_spectrum, template, observed, residual, error, wav_nm, wav0_nm

def spectra_from_sim(modelmap, contrast, roll, smoothing, mean_spectrum, wav_nm, wav0_nm,
                     error, residual, noisetype, kwargs_sim, savedir, n_lat=181, n_lon=361, 
                     r_deg=33, lat_deg=30, lon_deg=0, r2_deg=20, lat2_deg=45, lon2_deg=0,
                     plot_ts=False, plot_IC14=True, colorbar=True):
    nobs = kwargs_sim['nt']
    cmap = "plasma"
    # create fakemap
    if modelmap == "1spot":
        spot_brightness = contrast
        print(f"Running spot brightness {spot_brightness*100}% of surrounding")
        fakemap = np.ones((n_lat, n_lon))
        x, y = np.meshgrid(np.linspace(-180, 180, n_lon), np.linspace(-90, 90, n_lat))
        fakemap[np.sqrt((y-lat_deg)**2 + (x-lon_deg)**2) <= r_deg] = spot_brightness # default spot at lon=0

    if modelmap == "2spot":
        spot_brightness = contrast
        print(f"Running spot brightness {spot_brightness*100}% of surrounding")
        fakemap = np.ones((n_lat, n_lon))
        x, y = np.meshgrid(np.linspace(-180, 180, n_lon), np.linspace(-90, 90, n_lat), )
        fakemap[np.sqrt((y-lat_deg)**2 + (x-lon_deg)**2) <= r_deg] = spot_brightness
        fakemap[np.sqrt((y-lat2_deg)**2 + (x-lon2_deg)**2) <= r2_deg] = spot_brightness

    elif modelmap == "1band":
        band_width = r_deg
        band_lat = lat_deg
        amp = 1 - contrast
        print(f"Running wave max amplitude diff {amp*100}%")
        phase = lon_deg
        fakemap = np.ones((n_lat, n_lon))
        x, y = np.meshgrid(np.linspace(-180, 180, n_lon), np.linspace(-90, 90, n_lat))
        band_ind = np.s_[(90-band_lat)-band_width:(90-band_lat)+band_width]
        fakemap[band_ind] += amp * np.sin((x[band_ind]-lon_deg-90) * np.pi/180)
        #fakemap[band_ind] -= amp

    elif modelmap == "1uniband":
        print(f"Running band with brightness {contrast*100}% of surrounding")
        phase = 0.7 #0-1?
        fakemap = np.ones((n_lat, n_lon))
        x, y = np.meshgrid(np.linspace(-180, 180, n_lon), np.linspace(-90, 90, n_lat), )
        band_ind = np.s_[(90-lat_deg)-r_deg:(90-lat_deg)+r_deg]
        fakemap[band_ind] = contrast
        
    elif modelmap == "2band":
        band_width = 10
        band_lat = 45
        band2_lat = 0
        amp = 1 - contrast
        print(f"Running 2 waves max amplitude diff {amp*100}%")
        phase = 0.55 
        phase2 = 0.75
        fakemap = np.ones((n_lat, n_lon))
        x, y = np.meshgrid(np.linspace(-180, 180, n_lon), np.linspace(-90, 90, n_lat), )
        band_ind = np.s_[(90-band_lat)-band_width:(90-band_lat)+band_width]
        fakemap[band_ind] += amp * np.sin((x[band_ind]/360 - phase) * 2*np.pi)
        band2_ind = np.s_[(90-band2_lat)-band_width:(90-band2_lat)+band_width]
        fakemap[band2_ind] += amp * np.sin((x[band2_ind]/360 - phase2) * 2*np.pi)

    elif modelmap == "blank":
        fakemap = np.ones((n_lat, n_lon))

    elif modelmap == "gcm":
        fakemap = np.loadtxt(paths.data / 'modelmaps/gcm.txt')
        fakemap /= np.median(fakemap)
        diff = 1 - fakemap
        print(f"Running GCM peak amplitude diff {diff*100}%")
        ampold = diff.max()
        amp = 1 - contrast
        diffnew = diff * amp / ampold
        fakemap = 1 - diffnew
        n_lat, n_lon = fakemap.shape
        fakemap = np.roll(fakemap[::-1, :], shift=int(lon_deg/360 * n_lon), axis=1)

    elif modelmap == "SPOT":
        fakemap = str(paths.data / 'modelmaps/SPOT.png')

    elif modelmap == "testspots":
        spot_brightness = contrast
        lat_1, lon_1 = 60, -90
        lat_2, lon_2 = 30, 0
        lat_3, lon_3 = 0, 90
        r_deg = 20
        print(f"Running spot brightness {spot_brightness*100}% of surrounding")
        fakemap = np.ones((n_lat, n_lon))
        x, y = np.meshgrid(np.linspace(-180, 180, n_lon), np.linspace(-90, 90, n_lat))
        fakemap[np.sqrt((y-lat_1)**2 + (x-lon_1)**2) <= r_deg] = spot_brightness # default spot at lon=0
        fakemap[np.sqrt((y-lat_2)**2 + (x-lon_2)**2) <= r_deg] = spot_brightness
        fakemap[np.sqrt((y-lat_3)**2 + (x-lon_3)**2) <= r_deg] = spot_brightness

    #fakemap = np.roll(fakemap[::-1, :], shift=int(roll*n_lon), axis=1)

    # Compute simulated flux
    allchips_flux = []
    for i in range(wav_nm.shape[0]):
        sim_map = starry.DopplerMap(lazy=False, wav=wav_nm[i], wav0=wav0_nm[i], **kwargs_sim)
        sim_map.load(maps=[fakemap], smoothing=smoothing)
        sim_map[1] = kwargs_sim["u1"]

        flux_err_add = 0.02
        print("flux_err:", flux_err)
        noise = {
            "none": np.zeros((nobs, npix)),
            "random": np.random.normal(np.zeros((nobs, npix)), flux_err),
            #"obserr": error[:, i, pad:-pad],
            #"residual": residual[:, i, pad:-pad],
            #"res+random": residual[:, i, pad:-pad] + np.random.normal(np.zeros((nobs, npix)), flux_err_add)
        }

        sim_map.spectrum = mean_spectrum[i]
        model_flux = sim_map.flux(kwargs_sim["theta"])
        simulated_flux = model_flux + noise[noisetype]

        allchips_flux.append(simulated_flux)

    allchips_flux = np.array(allchips_flux)

    # Plot fakemap
    plot_map = starry.Map(lazy=False, **kwargs_sim)
    plot_map.load(fakemap)
    dif = 1 - contrast

    if plot_IC14:
        fig = plt.figure(figsize=(5,3))
        ax2 = fig.add_subplot(111)
        image = plot_map.render(projection="moll")
        im = ax2.imshow(image, cmap=cmap, aspect=0.5, origin="lower", interpolation="nearest")
        ax2.axis("off")
        if colorbar:
            fig.colorbar(im, ax=ax2, fraction=0.023, pad=0.045)
        ax = fig.add_subplot(111, projection='mollweide')
        ax.patch.set_alpha(0)
        yticks = np.linspace(-np.pi/2, np.pi/2, 7)[1:-1]
        xticks = np.linspace(-np.pi, np.pi, 13)[1:-1]
        ax.set_yticks(yticks, labels=[f'{deg:.0f}˚' for deg in yticks*180/np.pi], fontsize=7, alpha=0.5)
        ax.set_xticks(xticks, labels=[f'{deg:.0f}˚' for deg in xticks*180/np.pi], fontsize=7, alpha=0.5)
        if colorbar:
            fig.colorbar(im, ax=ax, fraction=0.023, pad=0.04, alpha=0)
        ax.grid('major', color='k', linewidth=0.25, alpha=0.7)
        for item in ax.spines.values():
            item.set_linewidth(1.2)
    
    else:
        fig, ax = plt.subplots()
        sim_map.show(ax=ax, projection="moll", colorbar=colorbar)

    plt.savefig(paths.figures / f"{savedir}/fakemap.png", bbox_inches="tight", dpi=100, transparent=True)

    if plot_ts:
        plot_timeseries(sim_map, model_flux, kwargs_sim["theta"], obsflux=simulated_flux, overlap=2)

    observed = np.transpose(allchips_flux, axes=(1,0,2))

    return observed, fakemap

def make_LSD_profile(instru, template, observed, wav_nm, goodchips, pmod, line_file, cont_file, nk, vsini, rv, period, timestamps, savedir, pad=100, cut=30, plotspec=False, colorbar=False):
    global wav_angs, err_LSD_profiles, dbeta
    print(instru)
    nobs = observed.shape[0]
    nchip = len(goodchips)
    # Read daospec linelist
    lineloc, lineew, _ = dao_getlines(line_file)
    pspec_cont = fits.getdata(cont_file)
    hdr_pspec_cont = fits.getheader(cont_file)
    wspec = hdr_pspec_cont['crval1'] + np.arange(pspec_cont.size)*hdr_pspec_cont['cdelt1']
    factor = 1e11 if "t1" in pmod else 1e5 # don't know why different model needs scaling with a factor
    pspec_cont = pspec_cont/factor
    spline = interpolate.UnivariateSpline(wspec, pspec_cont, s=0.0, k=1.0) #set up interpolation over the continuum measurement
    #plt.figure(figsize=(6,1))
    #plt.plot(wspec, pspec_cont)
    #plt.title("daospec continuum")
    #plt.show()

    wav_angs = np.array(wav_nm) * 10 #convert nm to angstroms

    # Compute LSD velocity grid:
    dbeta = np.diff(wav_angs).mean()/wav_angs.mean()
    print("dbeta", dbeta)
    dx = - dbeta * np.arange(np.floor(-nk/2.+.5), np.floor(nk/2.+.5))
    dv = const.c*dx / 1e3 # km/s

    # Compute LSD:
    kerns = np.zeros((nobs, nchip, nk), dtype=float)
    modkerns = np.zeros((nobs, nchip, nk), dtype=float)
    deltaspecs = np.zeros((nobs, nchip, npix), dtype=float)
    for i, jj in enumerate(goodchips): 
        print("chip", jj)
        for kk in range(nobs):
            shift = 1. + rv  # best rv shift for Callie is 9e-5
            deltaspec = make_deltaspec(lineloc*shift, lineew, wav_angs[i], verbose=False, cont=spline(wav_angs[i]))
            m,kerns[kk,i],b,c = dsa(deltaspec, observed[kk,i], nk)
            m,modkerns[kk,i],b,c = dsa(deltaspec, template[kk,i,pad:-pad], nk) 
            deltaspecs[kk,i] = deltaspec

    # Plot lines vs. model
    if plotspec:
        plt.figure(figsize=(15, 2*nchip))
        t=0
        for i, jj in enumerate(goodchips):
            plt.subplot(nchip,1,i+1)
            plt.plot(wav_angs[i], deltaspecs[t,i], linewidth=0.5, color='C0', label="deltaspec")
            plt.plot(wav_angs[i], template[t,i,pad:-pad], linewidth=0.6, color='C1', label="chipmodnobroad")
            plt.plot(wav_angs[i], observed[t,i], linewidth=0.6, color='k', label="obs")
            plt.text(x=wav_angs[i].min()-10, y=0, s=f"order={jj}")
            if i==0:
                plt.title(f"{pmod} model vs. lines at t={t}")
        plt.legend(loc=4, fontsize=9)
        #plt.show()
        plt.savefig(paths.output / "LSD_deltaspecs.png", transparent=True)
        
    # shift kerns to center
    modkerns, kerns = shift_kerns_to_center(modkerns, kerns, instru, goodchips, dv)
    
    # plot kerns
    #plot_kerns_timeseries(kerns, goodchips, dv, gap=0.02)
    #plot_kerns_timeseries(modkerns, goodchips, dv, gap=0.1)

    err_LSD_profiles = np.median(kerns.mean(1).std(0)) 
    # the error level across different obs of the chip-avged profile, median over nk pixels


    # normalize kerns
    obskerns_norm = cont_normalize_kerns(kerns, instru)
    
    # plot kerns + intrinsic_profile
    plot_kerns_timeseries(modkerns, goodchips, dv, gap=0.1)
    intrinsic_profiles = np.array([modkerns[:,i].mean(0) for i in range(nchip)])
    plot_kerns_timeseries(obskerns_norm, goodchips, dv, gap=0.02, normed=True, intrinsic_profiles=intrinsic_profiles)
    
    ### Plot averaged line shapes
    plot_chipav_kern_timeseries(obskerns_norm, dv, timestamps, savedir, gap=0.02, cut=int(cut/2+1))

    ### Plot deviation map for each chip and mean deviation map
    plot_deviation_map(obskerns_norm, goodchips, dv, vsini, timestamps, savedir, meanby="median", cut=cut, colorbar=colorbar)

    return intrinsic_profiles, obskerns_norm, dbeta

def solve_LSD_starry_lin(intrinsic_profiles, obskerns_norm, kwargs_run, kwargs_fig, annotate=False, colorbar=True):
    print("*** Using solver LSD+starry_lin ***")

    mean_profile = np.median(intrinsic_profiles, axis=0) # mean over chips
    observation_2d = np.median(obskerns_norm, axis=1)

    # preprae data
    pad = 100
    model_profile = 1. - np.concatenate((np.zeros(pad), mean_profile, np.zeros(pad)))
    dlam = np.diff(wav_angs[0]).mean() / 10 # angstrom to nm
    lam_ref = wav_angs[0].mean() / 10 # angstrom to nm
    nw = len(model_profile)
    wav0_lsd = np.linspace(start=lam_ref-0.5*dlam*nw, stop=lam_ref+0.5*dlam*nw, num=nw)
    wav_lsd = wav0_lsd[pad:-pad]
    
    map_av = starry.DopplerMap(lazy=False, wav=wav_lsd, wav0=wav0_lsd, **kwargs_run)
    map_av.spectrum = model_profile
    map_av[1] = kwargs_run['u1']

    soln = map_av.solve(
        flux=np.flip(observation_2d, axis=1),
        theta=kwargs_run['theta'],
        normalized=True,
        fix_spectrum=True,
        flux_err=flux_err,
        quiet=os.getenv("CI", "false") == "true",
    )

    image = map_av.render(projection="moll")         
    fig, ax = plt.subplots(figsize=(7,3))
    map_av.show(ax=ax, projection="moll", image=image, colorbar=colorbar)
    if annotate:
        ax.annotate(f"""
            chip={kwargs_fig['goodchips']}
            solver=LSD+starry_lin
            noise={kwargs_fig['noisetype']} 
            err_level={flux_err} 
            contrast={kwargs_fig['noisetype']} 
            limbdark={kwargs_run['u1']}""",
        xy=(-2, -1), fontsize=8)
    plt.savefig(paths.figures / f"{kwargs_fig['savedir']}/solver2.png", bbox_inches="tight", dpi=100, transparent=True)

    return map_av

def solve_LSD_starry_opt(intrinsic_profiles, obskerns_norm, kwargs_run, kwargs_fig, lr=0.001, niter=5000, annotate=False, colorbar=True):
    print("*** Using solver LSD+starry_opt ***")
    print(f"ydeg = {kwargs_run['ydeg']}")

    mean_profile = np.median(intrinsic_profiles, axis=0) # mean over chips
    observation_2d = np.median(obskerns_norm, axis=1)

    # preprae data
    pad = 100
    model_profile = 1. - np.concatenate((np.zeros(pad), mean_profile, np.zeros(pad)))
    dlam = np.diff(wav_angs[0]).mean() / 10 # angstrom to nm
    lam_ref = wav_angs[0].mean() / 10 # angstrom to nm
    nw = len(model_profile)
    wav0_lsd = np.linspace(start=lam_ref-0.5*dlam*nw, stop=lam_ref+0.5*dlam*nw, num=nw)
    wav_lsd = wav0_lsd[pad:-pad]

    with pm.Model() as model:
        # The surface map
        A = starry.DopplerMap(ydeg=kwargs_run['ydeg']).sht_matrix(smoothing=0.075)
        p = pm.Uniform("p", lower=0.0, upper=1.0, shape=(A.shape[1],))
        y = tt.dot(A, p)

        map_pm = starry.DopplerMap(lazy=True, wav=wav_lsd, wav0=wav0_lsd, **kwargs_run)

        map_pm[:, :] = y
        map_pm.spectrum = model_profile
        map_pm[1] = kwargs_run['u1']

        model_flux = map_pm.flux(theta=kwargs_run['theta'], normalize=True)

        # Likelihood term
        pm.Normal(
            f"obs",
            mu=tt.reshape(model_flux, (-1,)),
            sd=flux_err,
            observed=np.flip(observation_2d,axis=1).reshape(-1,)
        )

    # Optimize!
    loss = []
    best_loss = np.inf
    map_soln = model.test_point
    iterator = tqdm(
        pmx.optim.optimize_iterator(pmx.optim.Adam(lr=lr), niter, start=map_soln),
        total=niter,
        disable=os.getenv("CI", "false") == "true",
    )
    with model:
        for obj, point in iterator:
            iterator.set_description(
                "loss: {:.3e} / {:.3e}".format(obj, best_loss)
            )
            loss.append(obj)
            if obj < best_loss:
                best_loss = obj
                map_soln = point

    # Plot the loss
    fig, ax = plt.subplots(1, figsize=(3, 3))
    ax.plot(loss[int(len(loss)/20):])
    ax.set_xlabel("iteration number")
    ax.set_ylabel("loss")
    plt.savefig(paths.figures / f"{kwargs_fig['savedir']}/solver3_loss.png", bbox_inches="tight", dpi=100, transparent=True)

    # Plot the MAP map
    map_res = starry.Map(kwargs_run['ydeg'], kwargs_run['udeg'], inc=kwargs_run['inc'], lazy=False)
    map_res[1] = kwargs_run['u1']
    image = map_res.render(projection="rect")

    with model:
        y_map = pmx.eval_in_model(y, point=map_soln)

    map_res[:, :] = y_map   # The spherical harmonic coefficient vector. 
    fig, ax = plt.subplots()
    map_res.show(ax=ax, projection="moll", colorbar=colorbar, cmap="plasma")
    if annotate:
        ax.annotate(f"""
            chip={kwargs_fig['goodchips']}
            solver=LSD+starry_opt(lr={lr}) 
            noise={kwargs_fig['noisetype']} 
            err_level={flux_err} 
            contrast={kwargs_fig['contrast']} 
            limbdark={kwargs_run['u1']}""",
        xy=(-2, -1), fontsize=8)
    plt.savefig(paths.figures / f"{kwargs_fig['savedir']}/solver3.png", bbox_inches="tight", dpi=100, transparent=True)
    
    return map_res, image

def solve_starry_lin(mean_spectrum, observed, wav_nm, wav0_nm, kwargs_run, kwargs_fig, annotate=False, colorbar=True):
    print("*** Using solver starry_lin***")
    goodchips = kwargs_fig['goodchips']
    nchip = len(goodchips)

    maps = [None for i in range(nchip)]
    images = []
    recimages = []
    successchips = []
    for i, jj in enumerate(goodchips):
        maps[i] = starry.DopplerMap(lazy=False, wav=wav_nm[i], wav0=wav0_nm[i], **kwargs_run)
        maps[i].spectrum = mean_spectrum[i]
        maps[i][1] = kwargs_run['u1']

        try:
            print(f"Solving chip {jj}... [{i+1}/{nchip}]")
            soln = maps[i].solve(
                flux=observed[:,i,:],
                theta=kwargs_run['theta'],
                normalized=True,
                fix_spectrum=True,
                flux_err=flux_err,
                quiet=os.getenv("CI", "false") == "true",
            )
            imag = maps[i].render(projection="moll")
            recimag = maps[i].render(projection="rect")
            images.append(imag)
            recimages.append(recimag)
            successchips.append(jj)

            print("Success!")
        
        except:
            print(f"Solver failed for chip {jj}, moving onto next chip...")
            continue

    # plot map of each chip
    if nchip > 1:
        fig, axs = plt.subplots(len(successchips), 1)
        for i, jj in enumerate(successchips):
            maps[0].show(ax=axs[i], projection="moll", image=images[i], colorbar=False)
            axs[i].annotate(f"chip {jj}", xy=(-1.6, -1))
        plt.savefig(paths.figures / f"{kwargs_fig['savedir']}/solver4_each.png", bbox_inches="tight", dpi=100, transparent=True)

    # plot chip-averaged map
    images = np.array(images)
    recimages = np.array(recimages)

    fig, ax = plt.subplots()
    maps[0].show(ax=ax, projection="moll", image=np.mean(images, axis=0), colorbar=colorbar)
    if annotate:
        ax.annotate(f"""
            chip=median{goodchips} 
            solver=starry_lin 
            noise={kwargs_fig['noisetype']} 
            err_level={flux_err} 
            contrast={kwargs_fig['contrast']} 
            limbdark={kwargs_run['u1']}""",
        xy=(-2, -1), fontsize=8)
    plt.savefig(paths.figures / f"{kwargs_fig['savedir']}/solver4.png", bbox_inches="tight", dpi=100, transparent=True)

    return np.mean(images, axis=0), np.mean(recimages, axis=0)

def solve_starry_opt(mean_spectrum, observed, wav_nm, wav0_nm, kwargs_run, kwargs_fig, lr=0.05, niter=5000, annotate=False, colorbar=True):
    print("*** Using solver starry_opt ***")
    goodchips = kwargs_fig['goodchips']
    nchip = len(goodchips)

    with pm.Model() as model:
        # The surface map
        A = starry.DopplerMap(ydeg=kwargs_run['ydeg']).sht_matrix(smoothing=0.075)
        p = pm.Uniform("p", lower=0.0, upper=1.0, shape=(A.shape[1],))
        y = tt.dot(A, p)

        maps = [None for i in range(nchip)]
        flux_models = [None for i in range(nchip)]
        for i, jj in enumerate(goodchips):
            print(f"Setting chip {jj} ({i+1}/{nchip})...")
            maps[i] = starry.DopplerMap(lazy=True, wav=wav_nm[i], wav0=wav0_nm[i], **kwargs_run)
            maps[i][:, :] = y
            maps[i].spectrum = mean_spectrum[i]
            maps[i][1] = kwargs_run['u1']

            flux_models[i] = maps[i].flux(theta=kwargs_run['theta'], normalize=True)

            # Likelihood term
            pm.Normal(
                f"obs{i}",
                mu=tt.reshape(flux_models[i], (-1,)),
                sd=flux_err,
                observed=observed[:,i,:].reshape(-1,))

    # Optimize!
    loss = []
    best_loss = np.inf
    map_soln = model.test_point
    iterator = tqdm(
        pmx.optim.optimize_iterator(pmx.optim.Adam(lr=lr), niter, start=map_soln),
        total=niter,
        disable=os.getenv("CI", "false") == "true",
    )
    with model:
        for obj, point in iterator:
            iterator.set_description(
                "loss: {:.3e} / {:.3e}".format(obj, best_loss)
            )
            loss.append(obj)
            if obj < best_loss:
                best_loss = obj
                map_soln = point

    # Plot the loss
    fig, ax = plt.subplots(1, figsize=(3, 3))
    ax.plot(loss[int(len(loss)/10):])
    ax.set_xlabel("iteration number")
    ax.set_ylabel("loss")
    plt.savefig(paths.figures / f"{kwargs_fig['savedir']}/solver5_loss.png", bbox_inches="tight", dpi=100, transparent=True)

    # Plot the MAP map
    map_res = starry.Map(**kwargs_run)
    map_res[1] = kwargs_run['u1']

    with model:
        y_map = pmx.eval_in_model(y, point=map_soln)

    map_res[:, :] = y_map   # The spherical harmonic coefficient vector. 
    fig, ax = plt.subplots()
    map_res.show(ax=ax, projection="moll", colorbar=colorbar)
    if annotate:
        ax.annotate(f"""
            chip={goodchips}
            solver=starry_opt(lr={lr}) 
            noise={kwargs_fig['noisetype']} 
            err_level={flux_err} 
            contrast={kwargs_fig['contrast']} 
            limbdark={kwargs_run['u1']}""",
        xy=(-2, -1), fontsize=8)
    plt.savefig(paths.figures / f"{kwargs_fig['savedir']}/solver5.png", bbox_inches="tight", dpi=100, transparent=True)

    return map_res


################################################################################
####################   Utils    ################################################
################################################################################

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

    #EB - this loop gets the form needed
    dat = np.zeros([len(raw), 2], dtype=float)                                                 
    for i, line in enumerate(raw):                                         
        dat[i,:]= list(map(float, line.split()[0:2]))

    lineloc = dat[:,0]
    lineew = dat[:,1]/1e3
    linespec = [line.split()[-1] for line in raw]
    return (lineloc, lineew, linespec)

def make_deltaspec(loc, ew, win, **kw):
    """
    Create a delta-function line spectrum based on a wavelength grid
    and a list of line locations and equivalent widths.

    :INPUTS:
       loc -- location of lines in the emission frame of reference

       ew  -- equivalent widths of lines, in units of wavelength grid.
               Positive values are emission lines.

       w_in -- wavelength grid in the emission frame, with values
              monotonically increasing (best if it is linearly spaced)

       All inputs should be lists or one-dimensional arrays of scalars

    :OPTIONAL_INPUTS:
       cont=None -- set continuum values in the emission frame;

       nearest=False  -- if True, use full pixels instead of partial

       verbose=False  -- if True, print out various messages

    :OUTPUTS:
      s  -- delta-function line spectrum, with a continuum level of zero
    
    :EXAMPLE: (NEEDS TO BE UPDATED!):
       ::

          w   = [2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7]
          loc = [2.1, 2.35, 2.62]
          ew  = [0.1, .02, .01]
          s = linespec(loc, ew, w)
          print s  #  --->  [0, 1, 0, 0.1, 0.1, 0, 0.08, 0.02]

    :NOTE:  This may give incorrect results for saturated lines.
    """
    # 2008-12-05 13:31 IJC: Created
    # 2008-12-10 13:30 IJC: Added continuum option, reworked code some.
    # 2008-12-12 12:33 IJC: Removed RV option

    #from pylab import find
    from numpy import where as find

    # Check inputs
    loc = np.array(loc).copy().ravel()
    ew  = np.array(ew ).copy().ravel()
    win = np.array(win).copy().ravel()

    defaults = dict(cont=None, nearest=False, verbose=False)
    for key in defaults:
        #if (not kw.has_key(key)): #EB - This method does not work in python3
        if (not key in kw): #EB update
            kw[key] = defaults[key]
    verbose = bool(kw['verbose'])
    nearest = bool(kw['nearest'])
    contset = kw['cont']!=None

    if contset.all(): #EB update .all() to work in python3
        cont = np.array(kw['cont']).copy()
        if len(cont)!=len(win):
            print( "Wavelength grid and continuum must have the same length!")
            return -1
    else:
        cont = np.ones(win.shape)

    nlines = len(loc)
    if nlines != len(ew):
        if verbose:  print( "len(loc)>>" + str(len(loc)))
        if verbose:  print( "len(ew)>>" + str(len(ew)))
        print( "Line locations and equivalent widths must have same length!")
        return -1

    #Only use lines in the proper wavelength range
    nlineinit = len(loc)
    lind = (loc>=win.min()) * (loc<=win.max())
    loc = loc[lind]
    ew  =  ew[lind]
    nlines = len(loc)

    s = cont.copy()
    d = np.diff(win).mean()

    if verbose:  print( "s>>" + str(s))

    for ii in range(nlines):
        lineloc = loc[ii]
        lineew  = ew[ii]
        index = (win<lineloc).sum() - 1
        if nearest:
            s[index+1] = s[index]-cont[index]*lineew/d
        elif index==len(win):
            s[index] = s[index] - cont[index]*lineew/d
        else:
            s[index] = s[index] - lineew*cont[index]* \
                (win[index+1] - lineloc)/d/d
            s[index+1] = s[index+1] - lineew*cont[index+1] * \
                (lineloc - win[index])/d/d
        
        if verbose:  
            print( "(lineloc, lineew)>>" + str((lineloc, lineew)))
            print( "(index, d)>>" + str((index,d)))

    if verbose:
        print( "(nlineinit, nline)>>" + str((nlineinit, nlines)))
    return s

def rconvolve1d(a, b, mode='valid', extend='nearest'):
    """
    Compute a 1D reverse convolution in the style of Bramich 2008.

    :INPUTS:
        'a' should be longer than 'b' -- i.e., 'b' is the kernel.
        'extend' tells how to extend the boundaries -- either
        'nearest'-neighbor or a number

    :NOTES:
      This is "reversed" from the canonical definition of the convolution.

    :SEE_ALSO:   
      :func:`dsa`
    """
    # 2008-11-14 18:26 IJC: Created
    na = len(a)
    nb = len(b)
    n = max(na, nb)
    
    dx = int(np.floor(nb/2))

    if extend=='nearest':
        X = a[-1]
    else:
        X = extend

    a2 = X + np.zeros(na+nb-1, dtype=float)
    a2[dx:dx+na] = a
    #a2 = concatenate((a, X + zeros(nb-1)))
    #c = zeros(n, dtype='float')
    
    bmat = np.tile(b, (n,1))
    amat = np.zeros((n, nb), dtype='float')
    for ii in range(na):
        amat[ii,:] = a2[range(ii,ii+nb)]

    c = np.sum(amat * bmat, axis=1)
        
    return c

def dsa(r, i, Nk, **kw): #, w=None, verbose=False, noback=False):
    """
    Computational tool for Difference Spectral Analysis (DSA)
    
    :INPUTS:
       R -- reference spectrum.  This should have the highest possible
            signal-to-noise and the highest spectral resolution.
       I -- Current spectrum to be analysed.
       Nk -- number of pixels in the desired convolution kernel

    :OPTIONS:
       w       -- weights of the pixel values in I; typically (sigma)^-2
           (Not HANDELED CORRECTLY?!?!?)
       noback  -- do not fit for a variable background; assume constant.
       tol=1e-10 -- if matrix determinant is less than tol, use
                    pseudoinverse rather than straight matrix
                    inversion
       verbose -- Print output statements and make a plot or two
       retinv -- return a fourth output, the Least Squares inverse
                 matrix (False by default)

    :OUTPUTS:       (M, K, B, C):
       M -- R, convolved to match I
       K -- kernel used in convolution
       B -- background offset
       C -- chisquared of fit. If no weights were specified, weights
            are set to unity for this calculation.

    :OPTIONS:
       I -- inverse matrix

    :NOTES:
        Best results are obtained with proper registration of the spectra.
        Also, beware of edge effects.  As a general rule, anything within
        a kernel width of the edges is suspect.
        Also

    :SEE_ALSO:  
       :func:`dsamulti`
    
    Based on the 2D Bramich (2008) DIA algorithm
    -----
    2008-11-14 10:56 IJC: Created @ UCLA.
    2008-11-18 11:12 IJC: Registration now works correctly
    2008-12-09 16:10 IJC: Somewhat optimized
    2009-02-26 22:06 IJC: Added retinv, changed optional input format
    """

    defaults = dict(verbose=False, w=None, noback=False, tol=1e-10, \
                        retinv=False)
    for key in defaults:
        if (not (key in kw)):
            kw[key] = defaults[key]
    verbose = bool(kw['verbose'])
    noback = bool(kw['noback'])
    retinv = bool(kw['retinv'])
    w = kw['w']
    if verbose:
        print( "kw>>" + str(kw))

    if noback:
        if verbose: print( "Not fitting for a variable background...")

    tol = 1e-10  # tolerance for singularity

    r = np.array(r, copy=True)
    i = np.array(i, copy=True)
    Nk = int(Nk)  # length of kernel
    dx = int(np.floor(Nk/2))

    if w==None:
        w = np.ones(len(r), dtype=float)

    Nr = len(r)  # length of Referene
    ind = np.arange(Nr-Nk+1, dtype=int)
    wind = w[ind]
        
    if noback:    
        U = np.zeros((Nk,Nk), dtype=float)
        b = np.zeros(Nk, dtype=float)
    else:
        U = np.zeros((Nk+1,Nk+1), dtype=float)
        b = np.zeros(Nk+1, dtype=float)

    # Build the b vector and U matrix
    tempval0 = w[ind+dx] * i[ind+dx]
    for p in range(Nk):
        b[p] = (tempval0 * r[ind+p]).sum()
        tempval2 = wind*r[ind+p]
        for q in range(p, Nk):
            U[p,q] = (tempval2 * r[ind+q]).sum()
            U[q,p] = U[p,q]

    if not noback:
        b[Nk] = (w[ind+dx] * i[ind+dx]).sum()
        for q in range(Nk):
            U[Nk, q] = (wind * r[ind+q]).sum()
            U[q, Nk] = U[Nk, q]

        U[Nk,Nk] = wind.sum()
    
    detU = np.linalg.det(U)
    if verbose: print( "det(U) is:  " + str(detU))

    if detU<tol:
        print( "Singular matrix: det(U) < tol.  Using pseudoinverse...")
        if verbose: 
            print( 'U>>',U)
        invmat = np.linalg.pinv(U)
    else:
        invmat = np.linalg.inv(U)

    a = np.dot(invmat, b)

    if noback:
        K = a
        B0 = 0.0
    else:
        K = a[0:len(a)-1]
        B0 = a[-1]

    m = rconvolve1d(r, K, mode='valid') + B0

    chisq  = ( wind * (i[ind] - m[ind])**2 ).sum()

    if verbose:
        chisq0 = ( wind * (i[ind] - r[ind])**2 ).sum()
        #print "Kernel is:  " + str(K)
        print( "Background: " + str(B0))
        print( "For the (" + str(Nr) + " - " + str(Nk+1) + ") = " + str(Nr-Nk-1) + " DOF:")
        print( "Red. Chisquared (I-R): " + str(chisq0/(Nr-Nk-1)))
        print( "Red. Chisquared (I-M): " + str(chisq/(Nr-Nk-1)))
    
        plt.figure(); plt.subplot(311)
        plt.plot(r, '--'); plt.plot(i, '-x'); plt.plot(m, '-..'); plt.legend('RIM'); 
        plt.subplot(312); plt.plot(r - i, '--'); plt.plot(m - i); plt.legend(['R-I', 'M-I'])
        plt.subplot(313); plt.plot(K, '-o'); plt.grid('on'); plt.legend(['Kernel']); 

    if retinv:
        return (m, K, B0, chisq, invmat)
    else:
        return (m, K, B0, chisq)

def plot_timeseries(map, modelspec, theta, obsflux=None, overlap=8.0, figsize=(5, 11.5)):
    # Plot the "Joy Division" graph
    fig = plt.figure(figsize=figsize)
    ax_img = [
        plt.subplot2grid((map.nt, 8), (t, 0), rowspan=1, colspan=1)
        for t in range(map.nt)
    ]
    ax_f = [plt.subplot2grid((map.nt, 8), (0, 1), rowspan=1, colspan=7)]
    ax_f += [
        plt.subplot2grid(
            (map.nt, 8),
            (t, 1),
            rowspan=1,
            colspan=7,
            sharex=ax_f[0],
            sharey=ax_f[0],
        )
        for t in range(1, map.nt)
    ]

    for t in range(map.nt):
        map.show(theta=theta[t], ax=ax_img[t], res=300)

        for l in ax_img[t].get_lines():
            if l.get_lw() < 1:
                l.set_lw(0.5 * l.get_lw())
                l.set_alpha(0.75)
            else:
                l.set_lw(1.25)
        ax_img[t].set_rasterization_zorder(100)

        # plot the obs data points
        if obsflux is not None:
            ax_f[t].plot(obsflux[t] - modelspec[t, 0], "k.",
                        ms=0.5, alpha=0.75, clip_on=False, zorder=-1)
        # plot the spectrum
        ax_f[t].plot(modelspec[t] - modelspec[t, 0], "C1-", lw=0.5, clip_on=False)

        ax_f[t].set_rasterization_zorder(0)
    fac = (np.max(modelspec) - np.min(modelspec)) / overlap
    ax_f[0].set_ylim(-fac, fac)
 
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

def shift_kerns_to_center(modkerns, kerns, instru, goodchips, dv, sim=False, shiftkerns=False):
    '''shift modkerns to center at dv=0 and shift kerns for same amount.'''
    nobs, nchip, nk = modkerns.shape
    cen_modkerns = np.zeros_like(modkerns)
    cen_kerns = np.zeros_like(kerns)
    for i,jj in enumerate(goodchips):
        for k in range(nobs):
            systematic_rv_offset = (modkerns[k,i]==modkerns[k,i].max()).nonzero()[0][0] - (dv==0).nonzero()[0][0] # find the rv offset
            print("chip:", jj , "obs:", k, "offset:", systematic_rv_offset)
            cen_modkerns[k,i] = np.interp(np.arange(nk), np.arange(nk) - systematic_rv_offset, modkerns[k,i]) # shift ip to center at dv=0
            if k ==0:
                print("modkerns shifted to center.")
            cen_kerns[k,i] = kerns[k,i]
            if not sim: # shift kerns with same amount if not simulation
                if instru != 'CRIRES': # don't shift kerns if crires
                    if shiftkerns:
                        cen_kerns[k,i] = np.interp(np.arange(nk), np.arange(nk) - systematic_rv_offset, kerns[k,i])
                        if k ==0:
                            print("kerns shifted to same amount.")
    return cen_modkerns, cen_kerns

def cont_normalize_kerns(cen_kerns, instru):
    '''Continuum-normalize kerns by fitting a line at the flat edges of kern.'''
    nobs, nchip, nk = cen_kerns.shape
    obskerns = 1. - cen_kerns
    obskerns_norm = np.zeros_like(obskerns)
    continuumfit = np.zeros((nobs, nchip, 2))
    side = 15 if instru != "CRIRES" else 7
    for i in range(nchip):
        for n in range(nobs):
            inds = np.concatenate((np.arange(0, side), np.arange(nk-side, nk)))
            continuumfit[n,i] = np.polyfit(inds, obskerns[n,i,inds], 1)
            obskerns_norm[n,i] = obskerns[n,i] / np.polyval(continuumfit[n,i], np.arange(nk))
    return obskerns_norm

def plot_kerns_timeseries(kerns, goodchips, dv, gap=0.03, normed=False, intrinsic_profiles=None):
    '''Plot time series of kerns.'''
    nobs, nchip, nk = kerns.shape
    colors = [cm.gnuplot_r(x) for x in np.linspace(0, 1, nobs+4)]
    plt.figure(figsize=(nchip*3,4))
    for i, jj in enumerate(goodchips):
        plt.subplot(1, nchip, i+1)
        for n in range(nobs):
            if not normed:
                plt.plot(dv, 1 - kerns[n,i] - gap*n, color=colors[n])
            else:
                plt.plot(dv, kerns[n,i] - gap*n, color=colors[n])
        if intrinsic_profiles is not None:
            plt.plot(dv, 1-intrinsic_profiles[i], color='k')
        plt.title(f"chip={jj}")
        plt.xlabel("dv")
    plt.tight_layout()

def plot_chipav_kern_timeseries(obskerns_norm, dv, timestamps, savedir, gap=0.025, cut=17):
    '''Plot time series of chip-averaged kerns.'''
    nobs = obskerns_norm.shape[0]
    colors = [cm.gnuplot_r(x) for x in np.linspace(0, 1, nobs+4)]
    fig, ax = plt.subplots(figsize=(4, 5))
    for n in range(nobs):
        ax.plot(dv[cut:-cut], obskerns_norm.mean(axis=0).mean(axis=0)[cut:-cut] - gap*n, "--", color="gray", alpha=0.5)
        ax.plot(dv[cut:-cut], obskerns_norm[n].mean(axis=0)[cut:-cut] - gap*n, color=colors[n+1])
        #plt.text(dv[cut] + 10, 1 - gap/4 - gap*n, f"{timestamps[n]:.1f}h")
    #plt.plot(dv, 1-intrinsic_profiles.mean(axis=0), color='k', label="intrinsic profile")
    ax.set_xlabel("velocity (km/s)")
    ax.set_xticks([-50, -25, 0, 25, 50])
    ax.set_ylabel("Line intensity")
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ybound())
    ax2.set_yticks([1- gap*n for n in range(nobs)], labels=[f"{t:.1f}h" for t in timestamps], fontsize=9)
    #plt.axvline(x=vsini/1e3, color="k", linestyle="dashed", linewidth=1)
    #plt.axvline(x=-vsini/1e3, color="k", linestyle="dashed", linewidth=1)
    #plt.legend(loc=4, bbox_to_anchor=(1,1))
    plt.savefig(paths.figures / f"{savedir}/tsplot.png", bbox_inches="tight", dpi=300, transparent=True)

def plot_deviation_map(obskerns_norm, goodchips, dv, vsini, timestamps, savedir, meanby="median",colorbar=False, cut=30, lim=0.003):
    '''Plot deviation map for each chip and mean deviation map'''
    nobs, nchip, nk = obskerns_norm.shape
    uniform_profiles = np.zeros((nchip, nk))
    ratio = 1.3 if nobs < 10 else 0.7 #if nchip != 4 else 0.5

    # plot deviation map for each chip
    plt.figure(figsize=(nchip*4,3))
    for i, jj in enumerate(goodchips):
        uniform_profiles[i] = obskerns_norm[:,i].mean(axis=0) # is each chip's mean kern over epoches
        plt.subplot(1,nchip,i+1)
        plt.imshow(obskerns_norm[:,i]-uniform_profiles[i], 
            extent=(dv.max(), dv.min(), timestamps[-1], 0),
            aspect=int(ratio*29),
            cmap='YlOrBr') # positive diff means dark spot
        plt.xlim(dv.min()+cut, dv.max()-cut),
        plt.xlabel("velocity (km/s)")
        plt.ylabel("Elapsed time (h)")
        plt.title(f"chip={jj}")
    plt.tight_layout()
    plt.savefig(paths.figures / f"{savedir}/tvplot_full.png", bbox_inches="tight", dpi=100, transparent=True)

    # plot only the chip-mean map
    if meanby == "median":
        mean_dev = np.median(np.array([obskerns_norm[:,i]-uniform_profiles[i] for i in range(nchip)]), axis=0) # mean over chips
    elif meanby == "median_each":
        mean_dev = np.median(obskerns_norm, axis=1) - np.median(uniform_profiles,axis=0)
    elif meanby == "mean":
        mean_dev = np.mean(np.array([obskerns_norm[:,i]-uniform_profiles[i] for i in range(nchip)]), axis=0) # mean over chips
    plt.figure(figsize=(5,3))
    plt.imshow(mean_dev, 
        extent=(dv.max(), dv.min(), timestamps[-1], 0),
        aspect=int(ratio* 29),
        cmap='YlOrBr',
        vmin=-lim, vmax=lim) # positive diff means dark spot
    plt.xlim(dv.min()+cut, dv.max()-cut),
    plt.xlabel("velocity (km/s)", fontsize=8)
    plt.xticks([-50, -25, 0, 25, 50], fontsize=8)
    plt.ylabel("Elapsed time (h)", fontsize=8)
    plt.yticks([0, 1, 2, 3, 4, 5], fontsize=8)
    plt.vlines(x=vsini/1e3, ymin=0, ymax=timestamps[-1], colors="k", linestyles="dashed", linewidth=1)
    plt.vlines(x=-vsini/1e3, ymin=0, ymax=timestamps[-1], colors="k", linestyles="dashed", linewidth=1)
    if colorbar:
        cb = plt.colorbar(fraction=0.06, pad=0.28, aspect=15, orientation="horizontal", label="%")
        cb_ticks = cb.ax.get_xticks()
        cb.ax.set_xticklabels([f"{t*100:.1f}" for t in cb_ticks])
        cb.ax.tick_params(labelsize=8)
    #plt.title(f"{meanby} deviation")
    #plt.text(dv.min()+5, 0.5, f"chips={goodchips}", fontsize=8)
    plt.tight_layout()
    plt.savefig(paths.figures / f"{savedir}/tvplot.png", bbox_inches="tight", dpi=150, transparent=True)

def make_gif_map(bestparamgrid, inc, period, savedir, step=15, fps=4, vmax=110):
    fig = plt.figure(figsize=(10,5))
    y, x = bestparamgrid.shape

    # create an interpolation smoothed map
    plt.imshow(bestparamgrid, interpolation='bicubic', cmap='gray', aspect=x/y*0.5)
    plt.axis('off')
    fig.patch.set_visible(False)
    plt.savefig(paths.figures / f"{savedir}/solver1_map.png", bbox_inches='tight', dpi=200, pad_inches=0)
    img = Image.open(paths.figures / f"{savedir}/solver1_map.png")
    img = np.array(img.convert('L'), dtype='float64')
    img /= vmax

    # make GIF with PIL
    Nlat, Nlon = img.shape
    Lon, Lat = np.meshgrid(np.linspace(-180, 180, Nlon), np.linspace(-90, 90, Nlat))

    from matplotlib.animation import FuncAnimation
    num_frames = int(360/step)
    fig = plt.figure(figsize=(4, 4))
    ax = plt.axes(projection=ccrs.Orthographic(0, 10))
    time_text = ax.text(0.85, 0.05,'', transform=ax.transAxes)
    def update(frame):
        ax = plt.axes(projection=ccrs.Orthographic(frame*step, 90-inc))
        gl = ax.gridlines(xlocs=range(-180, 180, 30), ylocs=range(-90, 90, 30), color='gray', linewidth=0.3)
        im = ax.imshow(img, origin="lower", extent=(-180, 180, -90, 90), transform=ccrs.PlateCarree(), cmap=plt.cm.plasma, vmin=0.77, vmax=1)
        time_text.set_text(f'{frame*period/num_frames+0.2:.1f}h')
        for item in ax.spines.values():
            item.set_linewidth(1.5)
        return (ax, gl, im, time_text)

    # Create the animation
    ani = FuncAnimation(plt.gcf(), update, frames=num_frames)

    # Save the animation as a GIF
    output_gif_path = paths.figures / f"{savedir}/solver1_map.gif"
    ani.save(output_gif_path, dpi=100, fps=fps)
    '''
    frames = []
    for view in range(0, 360, step):
        fig = plt.figure(figsize=(4, 4))
        ax = plt.axes(projection=ccrs.Orthographic(view, 90-inc))
        for item in ax.spines.values():
            item.set_linewidth(1.5)
        gl = ax.gridlines(xlocs=range(-180,180,30), ylocs=range(-90,90,30), color='gray', linewidth=0.3)
        ax.imshow(img, origin="lower", extent=(-180, 180, -90, 90),transform=ccrs.PlateCarree(), cmap=plt.cm.plasma, vmin=0, vmax=1)  # Important

        # save fig in bytes format
        temp = paths.figures / f'temp/frame_{view}.png'
        plt.savefig(temp, format='png', dpi=150, bbox_inches='tight', transparent=True)
        #frames.append(imageio.imread(temp))
        frames.append(Image.open(temp))

    gif_file = paths.figures / f"{savedir}/solver1_map.gif"
    frames[0].save(gif_file, format='GIF', append_images=frames[1:], save_all=True, duration=duration, loop=0)
    '''

############################################################################################################
### retired functions ###
############################################################################################################

def solve_IC14new(intrinsic_profiles, obskerns_norm, kwargs_IC14, kwargs_fig, clevel=7,
                  ret_both=True, annotate=False, colorbar=False, plot_cells=False, plot_starry=False, plot_fit=False,
                  spotfit=False, create_obs_from_diff=True, vmin=85, vmax=110):
    print("*** Using solver IC14new ***")
    nobs, nk = obskerns_norm.shape[0], obskerns_norm.shape[2]

    bestparamgrid, res = solve_DIME(
        obskerns_norm, intrinsic_profiles,
        dbeta, nk, nobs, **kwargs_IC14, plot_cells=plot_cells, spotfit=spotfit,
        create_obs_from_diff=create_obs_from_diff
    )

    bestparamgrid_r = np.roll(
        np.flip(bestparamgrid, axis=1), int(0.5*bestparamgrid.shape[1]), axis=1)
    # TODO: derotate map??? seems like Ic14 maps are flipped and rolled 180 deg

    if plot_starry:
        fig, ax = plt.subplots(figsize=(7,3))
        showmap = starry.Map(ydeg=7)
        showmap.load(bestparamgrid_r)
        showmap.show(ax=ax, projection="moll", colorbar=colorbar)
    
    else:
        #pass
        #plot_IC14_map(bestparamgrid_r, clevel=clevel, sigma=2., colorbar=colorbar) # smoothed contour lines
        plot_IC14_map(bestparamgrid_r, colorbar=colorbar, vmin=vmin, vmax=vmax)

    map_type = "eqarea" if kwargs_IC14['eqarea'] else "latlon"
    if annotate:
        plt.text(-3.5, -1, f"""
            chip=averaged{kwargs_fig['goodchips']} 
            solver=IC14new {map_type} 
            noise={kwargs_fig['noisetype']} 
            err_level={flux_err} 
            contrast={kwargs_fig['contrast']} 
            limbdark={kwargs_IC14['LLD']}""",
        fontsize=8)
    plt.savefig(paths.figures / f"{kwargs_fig['savedir']}/solver1.png", bbox_inches="tight", dpi=100, transparent=True)

    # Plot fit result
    if plot_fit:
        obs_2d = np.reshape(res['sc_observation_1d'], (nobs, nk))
        bestmodel_2d = np.reshape(res['model_observation'], (nobs, nk))
        flatmodel_2d = np.reshape(res['flatmodel'], (nobs, nk))

        plt.figure(figsize=(5, 7))
        for i in range(nobs):
            plt.plot(res['dv'], obs_2d[i] - 0.02*i, color='k', linewidth=1)
            #plt.plot(obs[i] - 0.02*i, '.', color='k', markersize=2)
            plt.plot(res['dv'], bestmodel_2d[i] - 0.02*i, color='r', linewidth=1)
            plt.plot(res['dv'], flatmodel_2d[i] - 0.02*i, '--', color='gray', linewidth=1)
        plt.legend(labels=['obs', 'best-fit map', 'flat map'])
    try:
        plt.savefig(paths.figures / f"{kwargs_fig['savedir']}/solver1_ts.png", bbox_inches="tight", dpi=150, transparent=True)
    except:
        pass

    if ret_both:
        return bestparamgrid_r, res
    else:
        return bestparamgrid_r

def solve_DIME(
        obskerns_norm: np.ndarray, 
        intrinsic_profiles: np.ndarray,
        dbeta: float, 
        nk: int, nobs: int, 
        phases: np.ndarray, 
        inc: float, vsini: float, LLD: float, 
        eqarea: bool = True,
        nlat: int = 20, nlon: int = 40,
        alpha: int = 4500, ftol: float = 0.01,
        plot_cells: bool = False,
        plot_unstretched_map: bool = False,
        spotfit: bool = False,
        create_obs_from_diff: bool = True
    ) -> np.ndarray:
    """
    Copied from IC14orig except kerns used to compute weights should take 
    input from cen_kerns (profiles centered to rv=0).
    ***inc in degrees (90 <-> equator-on).***

    Parameters
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

    Returns
    -------
    bestparamgrid: 2darray, shape=(nlon, nlat)
        Optimized surface map. 
        Cells corresponding to longitude 0~2*pi, latitude 0~pi.

    """
    # Safely take means over chips
    mean_profile = np.median(intrinsic_profiles, axis=0) # mean over chips
    observation_2d = np.median(obskerns_norm, axis=1)
    observation_1d = observation_2d.ravel() # mean over chips and ravel to 1d

    # calc error for each obs
    smoothed = savgol_filter(obskerns_norm, 31, 3)
    resid = obskerns_norm - smoothed
    err_pix = np.array([np.abs(resid[:,:,pix] - np.median(resid, axis=2)) for pix in range(nk)]) # error of each pixel in LP by MAD, shape=(nk, nobs, nchips)
    err_LP = 1.4826 * np.median(err_pix, axis=0) # error of each LP, shape=(nobs, nchips)
    err_each_obs = err_LP.mean(axis=1) # error of each obs, shape=(nobs)
    err_observation_1d = np.tile(err_each_obs[:, np.newaxis], (1,nk)).ravel() # look like a step function over different times

    ### Prepare data for DIME
    modIP = 1. - np.concatenate((np.zeros(300), mean_profile, np.zeros(300)))
    modDV = - np.arange(np.floor(-modIP.size/2.+.5), np.floor(modIP.size/2.+.5)) * dbeta * const.c / 1e3
    modelfunc = interpolate.UnivariateSpline(modDV[::-1], modIP[::-1], k=1., s=0.) # function that returns the intrinsic profile
    dv = -dbeta * np.arange(np.floor(-nk/2.+.5), np.floor(nk/2.+.5)) * const.c / 1e3 # km/s

    ### Reconstruct map

    # initialize Doppler map object
    inc_ = (90 - inc) * np.pi / 180 # IC14 defined 0 <-> equator-on, pi/2 <-> face-on
    if eqarea:
        mmap = ELL_map.Map(nlat=nlat, nlon=nlon, type='eqarea', inc=inc_, verbose=True)
    else:
        mmap = ELL_map.Map(nlat=nlat, nlon=nlon, inc=inc_) #ELL_map.map returns a class object
    if plot_cells:
        mmap.plot_map_cells()
    ncell = mmap.ncell
    nx = ncell
    flatguess = 100*np.ones(nx)
    bounds = [(1e-6, 300)]*nx
    allfits = []

    # Compute R matrix
    Rmatrix = np.zeros((ncell, nobs*dv.size), dtype=np.float32)
    uncovered = list(range(ncell))
    for kk, rot in enumerate(phases):
        speccube = np.zeros((ncell, dv.size), dtype=np.float32) 
        if eqarea:
            this_map = ELL_map.Map(nlat=nlat, nlon=nlon, type='eqarea', inc=inc_, deltaphi=-rot)
        else:
            this_map = ELL_map.Map(nlat=nlat, nlon=nlon, inc=inc_, deltaphi=-rot)
        this_doppler = 1. + vsini*this_map.visible_rvcorners.mean(1)/const.c/np.cos(inc_) # mean rv of each cell in m/s
        good = (this_map.projected_area>0) * np.isfinite(this_doppler)    
        for ii in good.nonzero()[0]:
            if ii in uncovered:
                uncovered.remove(ii) # remove cells that are visible at this rot
            speccube[ii,:] = modelfunc(dv + (this_doppler[ii]-1)*const.c/1000.)
        limbdarkening = (1. - LLD) + LLD * this_map.mu
        Rblock = speccube * ((limbdarkening*this_map.projected_area).reshape(this_map.ncell, 1)*np.pi/this_map.projected_area.sum())
        Rmatrix[:,dv.size*kk:dv.size*(kk+1)] = Rblock

    dime = MaxEntropy(alpha, nk, nobs)
    flatmodel = dime.normalize_model(np.dot(flatguess, Rmatrix))
    flatmodel_2d = np.reshape(flatmodel, (nobs, nk))
    
    # create diff+flat profile
    nchip = obskerns_norm.shape[1]
    uniform_profiles = np.zeros((nchip, nk))
    for c in range(nchip):
        uniform_profiles[c] = obskerns_norm[:,c].mean(axis=0) # time-avged LP for each chip
    mean_dev = np.median(np.array([obskerns_norm[:,c]-uniform_profiles[c] for c in range(nchip)]), axis=0) # mean over chips
    new_observation_2d = mean_dev + flatmodel_2d
    new_observation_1d = new_observation_2d.ravel()

    # Properly scale measurement weights:
    # Mask out non-surface velocity space with weight=0
    width = int(vsini/1e3/np.abs(np.diff(dv).mean())) + 15 # vsini edge plus uncert=3
    central_indices = np.arange(nobs) * nk + int(nk/2)
    mask = np.zeros_like(observation_1d, dtype=bool)
    for central_idx in central_indices:
        mask[central_idx - width:central_idx + width + 1] = True
    w_observation = (mask==True).astype(float) / err_observation_1d**2

    if create_obs_from_diff:
        observation_1d = new_observation_1d

    # Scale the observations to match the model's equivalent width:
    out, eout = an.lsq((observation_1d, np.ones(nobs*nk)), flatmodel, w=w_observation)
    sc_observation_1d = observation_1d * out[0] + out[1]

    ### Solve!
    dime.set_data(sc_observation_1d, w_observation, Rmatrix)
    bfit = an.gfit(dime.entropy_map_norm_sp, flatguess, fprime=dime.getgrad_norm_sp, args=(), ftol=ftol, disp=1, maxiter=1e4, bounds=bounds)
    allfits.append(bfit)
    bestparams = bfit[0]
    model_observation = dime.normalize_model(np.dot(bestparams, Rmatrix))
    metric, chisq, entropy = dime.entropy_map_norm_sp(bestparams, retvals=True)
    print("metric:", metric, "chisq:", chisq, "entropy:", entropy)

    bestparams[uncovered] = np.nan # set completely uncovered cells to nan

    # reshape into grid
    if eqarea:
        # reshape into list
        start=0
        bestparamlist = []
        for m in range(this_map.nlat):
            bestparamlist.append(bestparams[start:start+this_map.nlon[m]])
            start = start + this_map.nlon[m]
        # interp into rectangular array
        max_length = max([len(x) for x in bestparamlist])
        stretched_arrays = []
        for array in bestparamlist:
            x_old = np.arange(len(array))
            x_new = np.linspace(0, len(array) - 1, max_length)
            y_new = np.interp(x_new, x_old, array)
            stretched_arrays.append(y_new)

        bestparamgrid = np.vstack(stretched_arrays)

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
                #plt.show()

    else:
        bestparamgrid = np.reshape(bestparams, (-1, nlon))

    res = dict(
        bestparams=bestparams,
        bestparamgrid=bestparamgrid, 
        Q=metric, chisq=chisq, entropy=entropy,
        Rmatrix=Rmatrix,
        model_observation=model_observation,
        sc_observation_1d=sc_observation_1d,
        w_observation=w_observation,
        flatmodel=flatmodel,
        flineSpline=modelfunc,
        mmap=mmap, dv=dv, dbeta=dbeta
    )
        
    return bestparamgrid, res

def plot_IC14_map(bestparamgrid, colorbar=False, clevel=5, sigma=1, vmax=None, vmin=None, cmap=plt.cm.plasma):
    '''Plot doppler map from an array.'''
    cmap = plt.cm.plasma.copy()
    cmap.set_bad('gray', 1)
    fig = plt.figure(figsize=(5,3))
    ax = fig.add_subplot(111, projection='mollweide')
    lon = np.linspace(-np.pi, np.pi, bestparamgrid.shape[1])
    lat = np.linspace(-np.pi/2., np.pi/2., bestparamgrid.shape[0])
    Lon, Lat = np.meshgrid(lon,lat)
    if vmax is None:
        im = ax.pcolormesh(Lon, Lat, bestparamgrid, cmap=cmap, shading='gouraud')
    else:
        im = ax.pcolormesh(Lon, Lat, bestparamgrid, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
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
