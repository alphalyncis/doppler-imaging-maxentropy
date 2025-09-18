import paths
import os
import numpy as np
from dime import DopplerImaging, load_data_from_pickle
from config import load_config
from matplotlib import pyplot as plt

instru = "CRIRES"
target = "W1049B"
band = "K"
savedir = f"{instru}_{band}_{target}_test"

params, goodchips, modelspec = load_config(instru, target, band)
model_datafile = paths.data/f'fitted/{instru}_{target}_{band}_{modelspec}.pickle'
#goodchips = [1,2,3,4]

if not os.path.exists(paths.figures/savedir):
    os.makedirs(paths.figures/savedir)

##############################################################################
####################      Run!      ##########################################
##############################################################################

# Load data from pickle fit
wav_nm, template, observed, error = load_data_from_pickle(model_datafile, goodchips, instru=instru)

dmap_cr = DopplerImaging(wav_nm, goodchips, params_dict=params)
dmap_cr.load_data(observed, template, error)
dmap_cr.make_lsd_profile(modelspec, plot_lsd_profiles=True, plot_deviation_map=True, colorbar=True, savedir=paths.figures/f"{savedir}/")
dmap_cr.solve(create_obs_from_diff=True, solver='scipy')

dmap_cr.bestparamgrid = np.roll(dmap_cr.bestparamgrid, shift=int(dmap_cr.bestparamgrid.shape[1]*0.75), axis=1) # rotate to phase 0.75
dmap_cr.plot_mollweide_map(vmin=85, vmax=110, colorbar=True, savedir=paths.figures/f"{savedir}/solver1.png")

#dmap.plot_fit_results_2d(gap=0.01)

dmap_cr.plot_fit_results_1d(savedir=paths.figures/f"{savedir}/fit1d_{target}_{band}.png")

plt.figure(figsize=(18,4.6))
sz=20
plt.subplot(131)
dmap_cr.plot_deviation_map(dmap_cr.obs_2d - dmap_cr.flatmodel_2d, newfig=False)
plt.title("Observed", fontsize=sz)
plt.subplot(132)
dmap_cr.plot_deviation_map(dmap_cr.bestmodel_2d - dmap_cr.flatmodel_2d, newfig=False)
plt.title("Modelled", fontsize=sz)
plt.subplot(133)
dmap_cr.plot_deviation_map(dmap_cr.obs_2d - dmap_cr.bestmodel_2d, newfig=False)
plt.title("Residual", fontsize=sz)
plt.colorbar()
plt.savefig(paths.figures/f"{savedir}/devs_{target}_{band}.png", bbox_inches="tight", dpi=150, transparent=True)


print("Run success.")

