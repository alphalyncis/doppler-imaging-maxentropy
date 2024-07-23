from matplotlib import pyplot as plt
import numpy as np
import paths
import os
homedir = os.path.expanduser('~')

target = "W1049A_0209"
crirestarget = target.split("_")[0]
targetname = f"WISE 1049{target[5]}"
date = "Feb 09" if "0209" in target else "Feb 11"

bands = ["H", "K", "HK", "K"]
bandname = ["H", "K", "H+K", "K \n2014"]
instru = ["igrins", "igrins", "igrins", "crires"]
instruname = ["IGRINS", "IGRINS", "IGRINS", "CRIRES"]
savepath = f"{homedir}/uoedrive/result/paper1/ref"

fig, axes = plt.subplots(4, 3, figsize=(20, 20))
fontsize = 30

for row in range(4):
    folder = f"{instruname[row]}_{bands[row]}_{target}"
    if row == 3:
        folder = f"{instruname[row]}_{bands[row]}_{crirestarget}"
    lp_fig = plt.imread(paths.figures / f"{folder}/LSD_profiles.png")
    dev_fig = plt.imread(paths.figures / f"{folder}/deviation_map.png")
    map_fig = plt.imread(paths.figures / f"{folder}/solver1.png")
    hspace = 0.20
    axes[row, 0].imshow(lp_fig)
    axes[row, 0].text(-0.55, 0.5, f"{instruname[row]} {bandname[row]}", fontsize=fontsize, transform=axes[row, 0].transAxes, ha="center")
    axes[row, 0].set_position([0, 0.01-row*hspace, 0.18, 0.18])
    axes[row, 1].imshow(dev_fig)
    axes[row, 1].set_position([0.23, 0-row*hspace, 0.2, 0.2]) # [left, bottom, width, height]
    axes[row, 2].imshow(map_fig)
    axes[row, 2].set_position([0.48, -0.04-row*hspace, 0.3, 0.3])
    if row == 3:
        axes[row, 1].set_position([0.224, -0.038-row*hspace, 0.22, 0.22])
        axes[row, 2].set_position([0.48, -0.081-row*hspace, 0.3, 0.3])
    if row == 0:
        titleheight = 1.15
        axes[row, 0].text(0.53, titleheight, "LSD line profiles", fontsize=fontsize, transform=axes[row, 0].transAxes, ha="center")
        axes[row, 1].text(1.91, titleheight, "deviations", fontsize=fontsize, transform=axes[row, 0].transAxes, ha="center")
        axes[row, 2].text(3.52, titleheight, f"{targetname} {date}", fontsize=fontsize, transform=axes[row, 0].transAxes, ha="center")
    for ax in axes[row]:
        ax.axis("off")

fig.savefig(f"{savepath}/maps_{target}.png", bbox_inches="tight", dpi=150)

