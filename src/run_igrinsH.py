import paths
import os
from dime import DopplerImaging, load_data
from config_run import *

use_eqarea = True
savedir = "igrinsH"
band = "H"
goodchips = [1,2,3,4]
nchip = len(goodchips)

modelspec = "t1400g1000f8"
model_datafile = paths.data/f'{instru}_{target}_{band}_{modelspec}.pickle'
pmod = f'linbroad_{modelspec}'

if not os.path.exists(paths.figures/savedir):
    os.makedirs(paths.figures/savedir)

##############################################################################
####################      Run!      ##########################################
##############################################################################

assert simulation_on == False

# Load data from pickle fit
wav_nm, template, observed, error = load_data(model_datafile, goodchips)

# Compute LSD mean profile
#intrinsic_profiles, obskerns_norm, dbeta = make_LSD_profile(instru, template, observed, wav_nm, goodchips, pmod, line_file, cont_file, nk, 
#                                                     vsini, rv, period, timestamps[target], savedir, cut=cut)

mapB_H = DopplerImaging(wav_nm, observed, template, error, timestamps, goodchips, kwargs_IC14=kwargs_IC14)
mapB_H.make_lsd_profile(line_file, cont_file, plot_lsd_profiles=False, plot_deviation_map=False)
mapB_H.solve(create_obs_from_diff=True, solver='scipy')


mapB_H.plot_mollweide_map(vmin=85, vmax=110, savedir=paths.figures/f"{savedir}/solver1.png")

#mapB_H.plot_fit_results()



#LSDlin_map = solve_LSD_starry_lin(intrinsic_profiles, obskerns_norm, kwargs_run, kwargs_fig, annotate=False, colorbar=False)
#LSDopt_map = solve_LSD_starry_opt(intrinsic_profiles, obskerns_norm, kwargs_run, kwargs_fig, lr=lr_LSD, niter=5000, annotate=False, colorbar=False)
#lin_map = solve_starry_lin(mean_spectrum, observed, wav_nm, wav0_nm, kwargs_run, kwargs_fig, annotate=False, colorbar=False)
#opt_map = solve_starry_opt(mean_spectrum, observed, wav_nm, wav0_nm, kwargs_run, kwargs_fig, lr=lr, niter=5000, annotate=False, colorbar=False)

print("Run success.")

