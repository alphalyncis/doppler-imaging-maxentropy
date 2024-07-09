import paths
import os
import numpy as np
from matplotlib import pyplot as plt

from dime import DopplerImaging, load_data_from_pickle
from config import load_config
from scipy.signal import savgol_filter

instru = "CRIRES"
target = "W1049B"
band = "K"
savedir = f"nktest_{instru}_{band}_{target}"

params, goodchips, modelspec = load_config(instru, target, band)
model_datafile = paths.data/f'fitted/{instru}_{target}_{band}_{modelspec}.pickle'

if not os.path.exists(paths.figures/savedir):
    os.makedirs(paths.figures/savedir)

start = 72 if band == "K" else 99 # starting order number
linemark = 100

##############################################################################
####################      Run!      ##########################################
##############################################################################

# Load data from pickle fit
wav_nm, template, observed, error = load_data_from_pickle(model_datafile, goodchips, instru=instru)

# Run LSD for different nk
nks = [53,75,101,125,151,175,201]
std_LPs = []
snr_LPs = []
for nk in nks:
    print("Running nk=", nk)
    #cut = nk - 50
    params['nk'] = nk

    dmap = DopplerImaging(wav_nm, goodchips, params_dict=params)
    dmap.load_data(observed, template, error)
    dmap.make_lsd_profile(modelspec, plot_lsd_profiles=False, plot_deviation_map=True)
    dmap.solve(create_obs_from_diff=True, solver='scipy')
    dmap.plot_mollweide_map(vmin=85, vmax=110)

    smoothed = savgol_filter(dmap.obskerns_norm, 31, 3)
    resid = dmap.obskerns_norm - smoothed
    err_pix = np.array([np.abs(resid[:,:,pix] - np.median(resid, axis=2)) for pix in range(nk)]) # error of each pixel in LP by MAD
    err_pix_linecen = err_pix[int(nk/2-30):int(nk/2+30)]
    err_LP = 1.4826 * np.median(err_pix_linecen, axis=0) # error of each LP (at each t and chip)

    signal = 1 - smoothed.min(axis=2).mean(axis=0) # signal = line depth
    noise = np.median(err_LP, axis=0) # mean error of a chip
    snr_LPs.append(signal/noise) # S/N of LP of each chip
    std_LPs.append(noise)

# Plot S/N of LPs
plt.figure(figsize=(5,4))

snr_LPs = np.array(snr_LPs)
for i in range(4):  
    plt.plot(nks, snr_LPs[:, i], marker=".")
plt.xlabel("nk")
plt.ylim(5, 35)
#plt.axvline(linemark, color="k", linestyle="--", alpha=0.4)
plt.ylabel("S/N of line profile")
plt.text(0.75, 0.93, f"{instru} {band}", fontsize=12, transform=plt.gca().transAxes)
plt.legend(labels=[f"order {c+1}" for c in goodchips], fontsize=7, bbox_to_anchor=(1,1))

plt.savefig(paths.figures/f"{savedir}/snr_LPs.png", dpi=100, transparent=True)

# Plot noise of LPs
plt.figure(figsize=(5,4))
std_LPs = np.array(std_LPs)
for i in range(4):  
    plt.plot(nks, std_LPs[:, i], marker=".")
plt.xlabel("nk")
plt.ylabel("noise level of line profile")
plt.legend(labels=[f"order {c+1}" for c in goodchips], fontsize=8, bbox_to_anchor=(1,1))

# Find good chips
print("goodchips=", np.where(snr_LPs[3]>20))
print("Run success.")

