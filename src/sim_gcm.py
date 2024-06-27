import numpy as np
import paths
import os

from dime import DopplerImaging, load_data_from_pickle, make_fakemap, simulate_data
from config import load_config

##############################################################################
####################    Configs     ##########################################
##############################################################################

modelmap = "gcm"
ydeg_sim = 25
savedir = "sim_gcm"

contrast = 0.7


# Load data from fit pickle
mean_spectrum, template, observed, residual, error, wav_nm, wav0_nm = load_data(model_datafile, instru, nobs, goodchips)

#bestparamgrid_rs = []
#for i in range(5):
# Make mock observed spectra
observed, fakemap = spectra_from_sim(modelmap, contrast, roll, smoothing, mean_spectrum, wav_nm, wav0_nm, error, residual, noisetype, kwargs_sim, savedir, 
                            lon_deg=90, plot_ts=False, plot_IC14=False, colorbar=False)

# Compute LSD mean profile
intrinsic_profiles, obskerns_norm = make_LSD_profile(instru, template, observed, wav_nm, goodchips, pmod, line_file, cont_file, nk, 
                                                    vsini, rv, period, timestamp, savedir, cut=cut)

bestparamgrid_r, res = solve_IC14new(intrinsic_profiles, obskerns_norm, kwargs_IC14, kwargs_fig, annotate=False, colorbar=False)
#bestparamgrid_rs.append(bestparamgrid_r)

#bestparamgrid_r = np.mean(np.array(bestparamgrid_rs), axis=0)
#plot_IC14_map(bestparamgrid_r)
#plt.savefig(paths.figures / f"{kwargs_fig['savedir']}/solver1.png", bbox_inches="tight", dpi=100, transparent=True)


#LSDlin_map = solve_LSD_starry_lin(intrinsic_profiles, obskerns_norm, kwargs_run, kwargs_fig, annotate=False, colorbar=False)

#LSDopt_map = solve_LSD_starry_opt(intrinsic_profiles, obskerns_norm, kwargs_run, kwargs_fig, lr=lr_LSD, niter=niter_LSD, annotate=False, colorbar=False)

#lin_map = solve_starry_lin(mean_spectrum, observed, wav_nm, wav0_nm, kwargs_run, kwargs_fig, annotate=True)
#plt.figure(figsize=(5,3))
#plt.savefig(paths.figures / f"{savedir}/solver4.pdf", bbox_inches="tight", dpi=300)

#opt_map = solve_starry_opt(mean_spectrum, observed, wav_nm, wav0_nm, kwargs_run, kwargs_fig, lr=lr, niter=niter, annotate=True)

