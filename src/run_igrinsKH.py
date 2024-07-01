import paths
import os
import numpy as np
from dime import DopplerImaging, load_data_from_pickle
from config import load_config

instru = "IGRINS"
target = "W1049B"
savedir = f"{instru}_HK_{target}"
modelspecK = "t1500g1000f8"
modelspecH = "t1500g1000f8"

params, goodchipsH, modelspecH = load_config(instru, target, "H")
model_datafileH = paths.data/f'fitted/{instru}_{target}_H_{modelspecH}.pickle'

params, goodchipsK, modelspecK = load_config(instru, target, "K")
model_datafileK = paths.data/f'fitted/{instru}_{target}_K_{modelspecK}.pickle'

if not os.path.exists(paths.figures/savedir):
    os.makedirs(paths.figures/savedir)

##############################################################################
####################      Run!      ##########################################
##############################################################################

# Load data from pickle fit
print("H band:")
wav_nmH, templateH, observedH, errorH = load_data_from_pickle(model_datafileH, goodchipsH)

print("K band:")
wav_nmK, templateK, observedK, errorK = load_data_from_pickle(model_datafileK, goodchipsK)

#mean_spectrumK, templateK, observedK, residualK, errorK, wav_nmK, wav0_nmK = load_data(model_datafileK, instru, nobs, goodchipsK)
#print("mean_spetrumK:", mean_spectrumK.shape)
#print("observedK:", observedK.shape)

wav_nm = np.concatenate((wav_nmH, wav_nmK), axis=0)
template = np.concatenate((templateH, templateK), axis=1)
observed = np.concatenate((observedH, observedK), axis=1)
error = np.concatenate((errorH, errorK), axis=1)
goodchips = goodchipsH + goodchipsK

print("\nAfter stacking:")
print("observed:", observed.shape)
print("wav:", wav_nm.shape)
print("goodchips:", goodchips)

dmap = DopplerImaging(wav_nm, goodchips, params_dict=params)
dmap.load_data(observed, template, error)
dmap.make_lsd_profile(modelspecK, plot_lsd_profiles=True, plot_deviation_map=True)
dmap.solve(create_obs_from_diff=True, solver='scipy')

dmap.plot_mollweide_map(vmin=85, vmax=110, savedir=paths.figures/f"{savedir}/solver1.png")

dmap.plot_fit_results_2d(gap=0.01)

print("Run success.")

