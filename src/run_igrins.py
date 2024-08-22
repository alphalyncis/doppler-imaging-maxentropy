import paths
import os
import numpy as np
from dime import DopplerImaging, load_data_from_pickle
from config import load_config
from matplotlib import pyplot as plt

instru = "IGRINS"
target = "W1049A"
band = "K"
savedir = f"{instru}_{band}_{target}"

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
dmap.make_lsd_profile(modelspec, plot_lsd_profiles=True, plot_deviation_map=True, savedir=paths.figures/f"{savedir}/")
dmap.solve(create_obs_from_diff=True, solver='scipy')


dmap.plot_mollweide_map(vmin=85, vmax=110, savedir=paths.figures/f"{savedir}/solver1.png")

dmap.plot_mecator_map(vmin=90, vmax=110, savedir=paths.figures/f"{savedir}/mecator.png")

#dmap.plot_fit_results_2d(gap=0.01)

dmap.plot_fit_results_1d(savedir=paths.figures/f"{savedir}/fit1d_{target}_{band}.png")

plt.figure(figsize=(18,4.6))
sz=20
plt.subplot(131)
dmap.plot_deviation_map(dmap.obs_2d - dmap.flatmodel_2d, newfig=False)
plt.title("Observed", fontsize=sz)
plt.subplot(132)
dmap.plot_deviation_map(dmap.bestmodel_2d - dmap.flatmodel_2d, newfig=False)
plt.title("Modelled", fontsize=sz)
plt.subplot(133)
dmap.plot_deviation_map(dmap.obs_2d - dmap.bestmodel_2d, newfig=False)
plt.title("Residual", fontsize=sz)
plt.colorbar()
plt.savefig(paths.figures/f"{savedir}/devs_{target}_{band}.png", bbox_inches="tight", dpi=150, transparent=True)


print("Run success.")

