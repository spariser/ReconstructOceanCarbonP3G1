### Configuration file for the project
### This file contains all the paths and parameters used in the project

SOCAT_PATH = "gs://leap-persistent/abbysh/zarr_files_/socat_mask_feb1982-dec2023.zarr"
PATH_SEEDS = (
    "gs://leap-persistent/abbysh/pickles/random_seeds.npy"  # random seeds for ML
)
GRID_SEARCH_APPROACH = "nmse"
MLINPUTS_PATH = "gs://leap-persistent/Mukkke/pco2_residual/post01_xgb_inputs"
ENSEMBLE_DIR = "gs://leap-persistent/abbysh/pco2_all_members_1982-2023/00_regridded_members"  # path to regridded data

FEATURES_SEL = [
    "sst",
    "sst_anom",
    "sss",
    "sss_anom",
    "mld_clim_log",
    "chl_log",
    "chl_log_anom",
    "xco2",
    "A",
    "B",
    "C",
    "T0",
    "T1",
]

# the target variable we reconstruct:
TARGET_SEL = [
    "pco2_residual"
]  # this represents pCO2 - pCO2-T (calculated in notebook 00)
