# Doppler Imaging with Maximum Entropy

Doppler imaging of brown dwarfs from high-resolution spectra using the maximum-entropy method. Repository for **Chen et al. 2024** (IGRINS Doppler imaging of WISE1049AB) plus follow-up work on CRIRES and on ELT/METIS L- and M-band simulations.

> Lineage: Ian Crossfield (Python 2.7) → Emma Bubb (Python 3) → Xueqing Chen
> (class-based rewrite, 2024). Tested on **Python 3.9.6 only**.

---

## 1. Quick start

To run the codes, it is recommended to create a virtual environment and install the required packages in `requirements.txt`. This can be done by running the set up script `setup_env.sh`:

```bash
chmod +x setup_env.sh
./setup_env.sh                 # creates venv/ and installs requirements.txt
source venv/bin/activate
```

To run a Doppler imaging run (e.g. `run_igrins.py`) you need the fitted `.pickle` files — these are **not** in the repo and are shared by hand.

---

## 2. The two-stage pipeline

```
 raw spectra (.fits, EXTERNAL)
        │
        │   stage 1: MODEL FITTING
        │   src/fit_*.py
        │   - fit a model/template spectrum to each order & epoch
        │   - solve for vsini, limb darkening, RV, continuum
        ▼
 data/fitted/<INSTRU>_<TARGET>_<BAND>_<MODELSPEC>.pickle   ← intermediate product
        │
        │   stage 2: DOPPLER IMAGING
        │   src/run_*.py  or  src/run_*.ipynb
        │   - load pickle → DopplerImaging (src/dime.py)
        │   - extract LSD line profiles
        │   - maximum-entropy solve → surface map
        ▼
 output/<...>/   maps, line-profile & fit plots
```

**Stage 1 (model fitting)** reads the *original* spectra and needs the external raw data. Its `.pickle` outputs land in `data/fitted/` (gitignored, **not** in the repo). Because stage 2 only needs those pickles, the usual way to let someone else reproduce the maps is to **send them the fitted pickles** so they can skip stage 1 entirely.

**Stage 2 (Doppler imaging)** is the scientific core. Start from `src/run_igrins.py` for a minimal script example, or the `run_*.ipynb` notebooks for the interactive /
figure-producing versions.

---

## 3. Naming conventions

These strings appear everywhere (filenames, config keys, function args):

| Token | Meaning | Examples |
|-------|---------|----------|
| `INSTRU`    | instrument | `IGRINS`, `CRIRES`, `METIS`, `ELT` |
| `TARGET`    | object (+ optional obs-date suffix) | `W1049A`/`W1049B` (Luhman 16 A/B), `W1049B_0209`, `2M0036_1103`, `case1` (sim) |
| `BAND`      | spectral band | `K`, `H`, `L`, `M` |
| `MODELSPEC` | template model atmosphere | `t1500g1000f8` = Teff 1500 K, logg 1000, fsed 8; `gcm` = GCM-derived |

All per-target settings (period, inclination, vsini, RV, usable orders, best
model) live in **`src/config.py`** → `load_config(instru, target, band)`.

---

## 4. Data: what's here vs. what you must fetch

**Committed in the repo:**
- `data/modelmaps/` — input surface maps for simulations (spot, GCM, jupiter…).
- `data/linelists/` — line lists for LSD.

**Local only — NOT in the repo (gitignored); share by hand:**
- `data/fitted/*.pickle` — fitted spectra, the input to stage 2. I have archived these files as `datafiles-dime.zip`, please contact me at xueqingchen.astro @ gmail.com for this file if you want to run stage 2 on your machine.

---

## 5. File map

### Core library (`src/`)
| File | Role |
|------|------|
| `dime.py`            | **Main engine.** `DopplerImaging` class (load → LSD → solve → plot) and `MaxEntropy` solver. Well-documented class docstring with array shapes. |
| `modelfitting.py`    | Spectrum-fitting routines. |
| `ELL_map_class.py`   | Planetary surface map / equal-area grid (from Crossfield). |
| `lsd_utils.py`       | Least-Squares Deconvolution helpers. |
| `config.py`          | All target/instrument parameters + `load_config()`. |
| `paths.py`           | Repo path constants (`paths.data`, `paths.output`, …). |

### Stage-1 fitting (`src/`)
- `fit_IGRINS.py`, `fit_CRIRES.py`, `fit_Kband.py`, `fit_Lband.py`,
  `fit_Mband.py`

### Stage-2 Doppler imaging (`src/`)
- `run_igrins.py`, `run_crires.py` — minimal scripted runs.
- `run_gcm_Lband.ipynb`, `run_gcm_Mband.ipynb`, `run_gcmhires.ipynb`,
  `run_metis_L.ipynb`, `run_metis_M.ipynb` — runs from GCM simulation (in progress).
- `sim_***.ipynb` — Injected map simulations.

### Plotting / analysis notebooks
- `plot_*.ipynb`, `src/make_gifs.ipynb`,
  `src/plot_skematic.ipynb` — paper figures.

### Yield / ELT forecasting (current work, experimental)
- `src/di_yield_calc.py`, `src/di_yield.ipynb`, `src/di_yield_plot.ipynb` —
  DI yield estimates for ELT.

---

## 6. Reference
Chen et al. 2024 — IGRINS Doppler imaging of WISE1049AB. See `LICENSE` for terms.
