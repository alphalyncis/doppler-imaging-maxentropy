import paths
import os
import numpy as np
from dime import DopplerImaging, load_data_from_pickle
from config import load_config
from matplotlib import pyplot as plt
import pandas as pd

instru = "IGRINS"
target = "W1049B_0209"
band = "K"
savedir = f"{instru}_{band}_{target}_202607"

params, goodchips, modelspec = load_config(instru, target, band)
model_datafile = paths.data/f'fitted/{instru}_{target}_{band}_{modelspec}.pickle'
#goodchips = [1,2,3,4]

if not os.path.exists(paths.output/savedir):
    os.makedirs(paths.output/savedir)

##############################################################################
####################      Run!      ##########################################
##############################################################################

# Load data from pickle fit
wav_nm, template, observed, error = load_data_from_pickle(model_datafile, goodchips)

dmap = DopplerImaging(wav_nm, goodchips, params_dict=params)
dmap.load_data(observed, template, error)
dmap.make_lsd_profile(modelspec, plot_lsd_profiles=True, plot_deviation_map=True, savedir=paths.output/f"{savedir}/")
dmap.solve(create_obs_from_diff=True, solver='scipy')

# save line profile data
df = pd.DataFrame({
    "obs_index":      np.repeat(np.arange(dmap.nobs), dmap.nk),
    "time":           np.repeat(dmap.timestamps, dmap.nk),
    "dv":             np.tile(dmap.dv, dmap.nobs),
    "observed_1d":    dmap.observed_1d,
    "observed_1d_sc": dmap.observed_1d_sc,
})
df.to_csv(paths.output/f"{savedir}/observed1d_{target}_{band}.csv", index=False)

dmap.plot_mollweide_map(vmin=85, vmax=110, savedir=paths.output/f"{savedir}/solver1.png")

dmap.plot_mecator_map(vmin=90, vmax=110, savedir=paths.output/f"{savedir}/mecator.png")

#dmap.plot_fit_results_2d(gap=0.01)

dmap.plot_fit_results_1d(savedir=paths.output/f"{savedir}/fit1d_{target}_{band}.png")

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
plt.savefig(paths.output/f"{savedir}/devs_{target}_{band}.png", bbox_inches="tight", dpi=150, transparent=True)


print("Run success.")

