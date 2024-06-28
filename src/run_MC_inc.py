import numpy as np
import matplotlib.pyplot as plt
import paths
import os

savedir = "MC_inc"
if not os.path.exists(paths.figures/savedir):
    os.makedirs(paths.figures/savedir)

### Measurements
vsini =     {"WISE1049B": {"K": 27.3, "H": 31.9, "avg": 28.6},
		     "WISE1049A": {"K": 19.4, "H": 19.2, "avg": 19.4}}
vsini_err = {"WISE1049B": {"K": 4.5, "H": 10.9, "avg": 7.0},
		 	 "WISE1049A": {"K": 5.2, "H": 16.2, "avg": 9.2}}
period =    {"WISE1049B": 5,
			 "WISE1049A": 7}
period_err = {"WISE1049B": 0.3,
			 "WISE1049A": 1}
radius, radius_err = 1.00, 0.1

# generate Gaussian samples
target = "WISE1049B"
tag = "avg"
vsini_arr = np.random.normal(vsini[target][tag], vsini_err[target][tag], 100000)
radius_arr = np.random.normal(radius,radius_err, 100000)
period_arr = np.random.normal(period[target], period_err[target], 100000)

Rjup = 69911.   # radius of jupiter in km
hourtosec = 3600.  # conversion from hours to seconds

# generate range of equatorial velocities from radius and period
veq_arr = (radius_arr * 2.*np.pi*Rjup) / (period_arr * hourtosec)

# calculate array of sin i
sini_arr = vsini_arr / veq_arr

def arcsin(theta_arr):
	ans = np.zeros_like(theta_arr)
	for i, theta in enumerate(theta_arr):
		if 0 <= theta <= 1:
			ans[i] = np.arcsin(theta)
		elif theta < 0:
			ans[i] = 0
		else:
			ans[i] = np.pi/2
	return ans

hfont = {'size':'10'}
fig = plt.figure(figsize=(10,3.5))
fig.suptitle(f"{target}, period = {period[target]:.1f}±{period_err[target]:.1f} h, radius = {radius:.1f}±{radius_err:.1f} R_jup")
plt.subplot(141)
plt.hist(vsini_arr, 30, density=True, color='green')
plt.xlabel('v sin i (km/s)', **hfont)
plt.axvline(np.median(vsini_arr), color='k')
plt.axvline(np.mean(vsini_arr), linestyle='--', color='k')
plt.title(f"v sin i = {np.mean(vsini_arr):.1f}±{np.std(vsini_arr):.1f}", transform=plt.gca().transAxes, **hfont)

plt.subplot(142)
plt.hist(veq_arr, 30, density=True, color='blue')
plt.xlabel('equatorial velocity (km/s)', **hfont)
plt.axvline(np.median(veq_arr), color='k')
plt.axvline(np.mean(veq_arr), linestyle='--', color='k')
plt.title(f"veq = {np.mean(veq_arr):.1f}±{np.std(veq_arr):.1f}", transform=plt.gca().transAxes, **hfont)

plt.subplot(143)
plt.hist(sini_arr, 30, density=True, color='red')
plt.xlabel('sin i', **hfont)
plt.axvspan(xmin=1, xmax=plt.xlim()[1], color='gray', alpha=0.4, label="unphysical")
plt.text(0.54, 0.9, "unphysical", transform=plt.gca().transAxes, alpha=0.7, **hfont)
plt.axvline(np.median(sini_arr), color='k')
plt.axvline(np.mean(sini_arr), linestyle='--', color='k')
plt.title(f"sin i = {np.mean(sini_arr):.1f}±{np.std(sini_arr):.1f}", transform=plt.gca().transAxes, **hfont)

plt.subplot(144)
inc_arr = arcsin(sini_arr)*(180./np.pi)
plt.hist(inc_arr, 30, density=True, color='magenta')
plt.xlabel('i (degrees)', **hfont)
plt.axvline(np.median(inc_arr), color='k', label=f"median: {np.median(inc_arr):.1f}")
plt.axvline(np.mean(inc_arr), linestyle='--', color='k', label=f"mean: {np.mean(inc_arr):.1f}")
#plt.errorbar(np.mean(inc_arr), plt.ylim()[1]*0.8, xerr=np.std(inc_arr), color="k", capsize=2)
plt.title(f"i = {np.mean(inc_arr):.1f}±{np.std(inc_arr):.1f}", transform=plt.gca().transAxes, **hfont)
plt.xticks(np.linspace(0, 90, 10))
plt.xlim(0, 93)

plt.tight_layout()
plt.legend()

print(f"inc median: {np.median(inc_arr):.1f}")
print(f"inc mean: {np.mean(inc_arr):.1f}")
print(f"inc std: {np.std(inc_arr):.1f}")

plt.savefig(paths.figures / f"{savedir}/{target}_inc.png", dpi=150, transparent=True)