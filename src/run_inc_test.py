import paths
import os
import numpy as np
from dime import DopplerImaging, load_data_from_pickle
from config import load_config

instru = "IGRINS"
target = "W1049A"
band = "H"
savedir = f"{instru}_inctest"

params, goodchips, modelspec = load_config(instru, target, band)
model_datafile = paths.data/f'fitted/{instru}_{target}_{band}_{modelspec}.pickle'
#goodchips = [1,2,3,4]

if not os.path.exists(paths.figures/savedir):
    os.makedirs(paths.figures/savedir)

##############################################################################
####################      Run!      ##########################################
##############################################################################

# Load data from pickle fit
wav_nm, template, observed, error = load_data_from_pickle(model_datafile, goodchips)

dmap = DopplerImaging(wav_nm, goodchips, params_dict=params)
dmap.load_data(observed, template, error)
dmap.make_lsd_profile(modelspec, plot_lsd_profiles=True, plot_deviation_map=True)

for inc in [30, 50, 70]:
    dmap.inc = inc
    dmap.inc_ = (90 - dmap.inc) * np.pi / 180
    dmap.solve(create_obs_from_diff=True, solver='scipy')
    dmap.plot_mollweide_map(vmin=85, vmax=110, savedir=paths.figures/f"{savedir}/inc_{inc}.png")




#LSDlin_map = solve_LSD_starry_lin(intrinsic_profiles, obskerns_norm, kwargs_run, kwargs_fig, annotate=False, colorbar=False)
#LSDopt_map = solve_LSD_starry_opt(intrinsic_profiles, obskerns_norm, kwargs_run, kwargs_fig, lr=lr_LSD, niter=5000, annotate=False, colorbar=False)
#lin_map = solve_starry_lin(mean_spectrum, observed, wav_nm, wav0_nm, kwargs_run, kwargs_fig, annotate=False, colorbar=False)
#opt_map = solve_starry_opt(mean_spectrum, observed, wav_nm, wav0_nm, kwargs_run, kwargs_fig, lr=lr, niter=5000, annotate=False, colorbar=False)

print("Run success.")

