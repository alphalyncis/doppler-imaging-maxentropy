import numpy as np
import paths
import os

from dime import *

##############################################################################
####################    Configs     ##########################################
##############################################################################

from config_sim import *

maptype = "2spot"
savedir = "sim_2spot"
contrast = 0.8

if not os.path.exists(paths.figures / savedir):
    os.makedirs(paths.figures / savedir)

##############################################################################
####################      Sim!      ##########################################
##############################################################################

assert simulation_on == True

# Load data from fit pickle
wav_nm, template, observed, error = load_data(model_datafile, goodchips)

# Make mock observed spectra
fakemap = make_fakemap(maptype, contrast, 
    lat_deg=0, lon_deg=90, r1_deg=25, lat1_deg=45, lon1_deg=-60)

mean_spectrum = np.median(template, axis=0)
observed = simulate_data(fakemap, mean_spectrum, wav_nm, flux_err, kwargs_sim)

# Compute LSD mean profile
#intrinsic_profiles, obskerns_norm = make_LSD_profile(instru, template, observed, wav_nm, goodchips, pmod, line_file, cont_file, nk, 
#                                                   vsini, rv, period, timestamp, savedir, cut=cut)

#bestparamgrid_r, bestparamgrid = solve_IC14new(intrinsic_profiles, obskerns_norm, kwargs_IC14, kwargs_fig, annotate=False, colorbar=False)


#plot_IC14_map(bestparamgrid_r, colorbar=False, vmin=85, vmax=110) # derotated
#plt.savefig(paths.figures / f"{kwargs_fig['savedir']}/solver1.png", bbox_inches="tight", dpi=100, transparent=True)

#LSDlin_map = solve_LSD_starry_lin(intrinsic_profiles, obskerns_norm, kwargs_run, kwargs_fig, annotate=False, colorbar=False)

#LSDopt_map = solve_LSD_starry_opt(intrinsic_profiles, obskerns_norm, kwargs_run, kwargs_fig, lr=lr_LSD, niter=niter_LSD, annotate=False, colorbar=False)

#lin_map = solve_starry_lin(mean_spectrum, observed, wav_nm, wav0_nm, kwargs_run, kwargs_fig, annotate=False, colorbar=False)
#plt.figure(figsize=(5,3))
#plt.savefig(paths.figures / f"{savedir}/solver4.pdf", bbox_inches="tight", dpi=300)

#opt_map = solve_starry_opt(mean_spectrum, observed, wav_nm, wav0_nm, kwargs_run, kwargs_fig, lr=lr, niter=niter, annotate=False, colorbar=False)

