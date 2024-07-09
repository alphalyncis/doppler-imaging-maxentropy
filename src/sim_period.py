from doppler_imaging import *
import numpy as np
import paths
import os

##############################################################################
####################    Configs     ##########################################
##############################################################################

from config_sim import *

target = "W1049A"
contrast = 0.7

modelmap = "testspots"
tobs = 5

maps = []
for period_true in [3, 5, 7, 10, 12]:
    period = period_true
    savedir = f"sim_period/{period_true}"

    if not os.path.exists(paths.figures / savedir):
        os.makedirs(paths.figures / savedir)


        # set time and period parameters

        # simualte with true time points
        nobs_true = int(nobs * period / tobs + 0.5) # number of observed time points in simulation
        timestamp_true = np.linspace(0, period, nobs_true)  # simulated obs time points in hours
        if period < tobs:
            nobs_extra = nobs - len(timestamp_true) + 1
            extra_times = timestamp_true[1:nobs_extra] + period
            timestamp_true = np.concatenate((timestamp_true, extra_times))
            nobs_true = len(timestamp_true)
        theta_true = 360.0 * timestamp_true / period_true

        # solve with actual observed time points
        timestamp = timestamp_true[:nobs] # observed time points in hours
        theta = theta_true[:nobs]          # observed time points in degree (0 ~ 360)
        phases = theta * np.pi / 180.0    # observed time points in rad (0 ~ 2*pi)


        kwargs_sim = dict(
            ydeg=ydeg_sim,
            udeg=udeg,
            nc=nc,
            veq=veq,
            inc=inc,
            nt=nobs_true,
            vsini_max=vsini_max,
            u1=u1,
            theta=theta_true)

        kwargs_IC14 = dict(
            phases=phases, 
            inc=inc, 
            vsini=vsini, 
            LLD=LLD, 
            eqarea=use_eqarea, 
            nlat=nlat, 
            nlon=nlon,
            alpha=alpha,
            ftol=ftol
        )


    ##############################################################################
    ####################      Run!      ##########################################
    ##############################################################################


    # Load data from fit pickle
    mean_spectrum, template, observed, residual, error, wav_nm, wav0_nm = load_data(model_datafile, instru, nobs, goodchips)

    # Make mock observed spectra
    observed, fakemap = spectra_from_sim(modelmap, contrast, roll, smoothing, mean_spectrum, wav_nm, wav0_nm, error, residual, 
                            noisetype, kwargs_sim, savedir, plot_ts=False, plot_IC14=False, colorbar=False)
    # Compute LSD mean profile
    intrinsic_profiles, obskerns_norm = make_LSD_profile(instru, template, observed[:nobs], wav_nm, goodchips, pmod, line_file, cont_file, nk, vsini, rv, 
                                                         period, timestamp, savedir, cut=cut)

    bestparamgrid_r, res = solve_IC14new(intrinsic_profiles, obskerns_norm, kwargs_IC14, kwargs_fig, annotate=False, colorbar=False)
    maps.append(bestparamgrid_r)

    #LSDlin_map = solve_LSD_starry_lin(intrinsic_profiles, obskerns_norm, kwargs_run, kwargs_fig, annotate=False, colorbar=False)

    #LSDopt_map = solve_LSD_starry_opt(intrinsic_profiles, obskerns_norm, kwargs_run, kwargs_fig, lr=lr_LSD, niter=niter_LSD, annotate=True)

    #lin_map = solve_starry_lin(mean_spectrum, observed, wav_nm, wav0_nm, kwargs_run, kwargs_fig, annotate=True)
    #plt.figure(figsize=(5,3))
    #plt.savefig(paths.figures / f"{savedir}/solver4.pdf", bbox_inches="tight", dpi=300)

    #opt_map = solve_starry_opt(mean_spectrum, observed, wav_nm, wav0_nm, kwargs_run, kwargs_fig, lr=lr, niter=niter, annotate=True)

