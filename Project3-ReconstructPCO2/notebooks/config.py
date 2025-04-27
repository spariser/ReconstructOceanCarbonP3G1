### Configuration file for the project
### This file contains all the paths and parameters used in the project

socat_path = "gs://leap-persistent/abbysh/zarr_files_/socat_mask_feb1982-dec2023.zarr"
path_seeds = (
    "gs://leap-persistent/abbysh/pickles/random_seeds.npy"  # random seeds for ML
)
grid_search_approach = "nmse"
MLinputs_path = "gs://leap-persistent/Mukkke/pco2_residual/post01_xgb_inputs"
ensemble_dir = "gs://leap-persistent/abbysh/pco2_all_members_1982-2023/00_regridded_members"  # path to regridded data

features_sel = [
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
target_sel = [
    "pco2_residual"
]  # this represents pCO2 - pCO2-T (calculated in notebook 00)
