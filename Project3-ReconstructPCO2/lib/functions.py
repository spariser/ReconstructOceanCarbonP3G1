import numpy as np
import xarray as xr
import pandas as pd
import scipy.stats
import statsmodels.nonparametric.smoothers_lowess
from joblib import Parallel, delayed
import os
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

def plot_mask(mask, title):
    with plt.style.context("seaborn-v0_8-talk"):
        fig = plt.figure(figsize=(8.5, 11)) # fig = plt.figure(dpi=300)
        worldmap = SpatialMap2(fig=fig, region='world', 
                    cbar_mode='each',
                    colorbar=True,
                    cbar_location='bottom',
                    nrows_ncols=[1,1])
        
        vrange = [0, 144, 12]
        cmap = cm.cm.rain
        data = xr_add_cyclic_point(mask, cyclic_coord='xlon')
        data = data.assign_coords(xlon=(((data.xlon + 180) % 360) ))
        sub = worldmap.add_plot(lon=data['xlon'], lat=data['ylat'], data=data, 
                        vrange=vrange[0:2], cmap=cmap, ax=0, linewidth_coast=0.5)
        
        col = worldmap.add_colorbar(sub, ax=0,extend='max')
        worldmap.set_cbar_xlabel(col, 'Number of months with data', fontsize=14)
        worldmap.set_ticks(col, vrange[0], vrange[1], vrange[2])
        col.ax.tick_params(labelsize=12)
        worldmap.set_title(title, ax=0, fontsize=14)
        plt.show()
        # save figure
        fig.savefig(f"mask_{title}.png", dpi=300, bbox_inches='tight')

def add_to_existing(non_zero_counts, socat_mask_data):
    mean_val_glob_loc = non_zero_counts.where((non_zero_counts > non_zero_counts.mean()) | (non_zero_counts == 0), 5)
    socat_mean_glob = socat_mask_data.where((non_zero_counts > non_zero_counts.mean()) | (non_zero_counts == 0),
                                            socat_mask_data.sel(xlon=-172.5, ylat=-75.5),)

    thirtyp_val_glob = non_zero_counts.where((non_zero_counts > 7) | (non_zero_counts == 0), 7)    
    socat_30p_glob = socat_mask_data.where((non_zero_counts > 7) | (non_zero_counts == 0),
                                           socat_mask_data.sel(xlon=171.5, ylat=-39.5),)
    
    fiftyp_val_glob = non_zero_counts.where((non_zero_counts > 10) | (non_zero_counts == 0), 10)
    socat_50p_glob = socat_mask_data.where((non_zero_counts > 9) | (non_zero_counts == 0),
                                           socat_mask_data.sel(xlon=132.5, ylat=-54.5),)

    return mean_val_glob_loc, socat_mean_glob, thirtyp_val_glob, socat_30p_glob, fiftyp_val_glob, socat_50p_glob 

def add_new(non_zero_counts, socat_mask_data):
    addmeanp_oceans = non_zero_counts.copy(deep=True)
    addmeanp_socat = socat_mask_data.copy(deep=True)


    addmeanp_oceans.loc[dict(xlon=slice(-97, -87), ylat=slice(-46, -36))] = 34  # Pacific 1
    addmeanp_oceans.loc[dict(xlon=slice(-130, -110), ylat=slice(-45, -35))] = 34  # Pacific 2
    addmeanp_oceans.loc[dict(xlon=slice(-141, -121), ylat=slice(-32, -27))] = 34  # Pacific 3
    addmeanp_oceans.loc[dict(xlon=slice(-60, -40), ylat=slice(-77, -67))] = 34  # Southern Ocean
    addmeanp_oceans.loc[dict(xlon=slice(75, 85), ylat=slice(-11, 9))] = 33  # Indian Ocean 1
    addmeanp_oceans.loc[dict(xlon=slice(70, 90), ylat=slice(-30, -25))] = 33  # Indian Ocean 2

    ref_val1 = socat_mask_data["socat_mask"].sel(xlon=-11.5, ylat=40.5)
    ref_val2 = socat_mask_data["socat_mask"].sel(xlon=154.5, ylat=43.5)

    addmeanp_socat["socat_mask"].loc[dict(ylat=slice(-46, -36), xlon=slice(-97, -87))] = ref_val1
    addmeanp_socat["socat_mask"].loc[dict(ylat=slice(-45, -35), xlon=slice(-130, -120))] = ref_val1
    addmeanp_socat["socat_mask"].loc[dict(ylat=slice(-32, -27), xlon=slice(-141, -121))] = ref_val1
    addmeanp_socat["socat_mask"].loc[dict(ylat=slice(-77, -67), xlon=slice(-60, -40))] = ref_val1

    addmeanp_socat["socat_mask"].loc[dict(ylat=slice(-11, 9), xlon=slice(75, 85))] = ref_val2
    addmeanp_socat["socat_mask"].loc[dict(ylat=slice(-30, -25), xlon=slice(70, 90))] = ref_val2

    # Adding 30% more values to undersampled oceans, in currently unsampled locations
    add30p_oceans = non_zero_counts.copy(deep=True)
    add30p_socat = socat_mask_data.copy(deep=True)
    add30p_oceans.loc[dict(xlon=slice(-97, -87), ylat=slice(-46, -36))] = 73  # pacific1
    add30p_oceans.loc[dict(xlon=slice(-130, -110), ylat=slice(-45, -35))] = 73  # pacific2
    add30p_oceans.loc[dict(xlon=slice(-141, -121), ylat=slice(-32, -27))] = 73  # pacific3
    add30p_oceans.loc[dict(xlon=slice(-60, -40), ylat=slice(-77, -67))] = 73  # southern ocean

    add30p_oceans.loc[dict(xlon=slice(75, 85), ylat=slice(-11, 9))] = 72 # indian ocean1
    add30p_oceans.loc[dict(xlon=slice(70, 90), ylat=slice(-30, -25))] = 72  # indian ocean2

    ref_val3 = socat_mask_data["socat_mask"].sel(xlon=-27.5, ylat=62.5)
    ref_val4 = socat_mask_data["socat_mask"].sel(xlon=-87.5, ylat=18.5)

    add30p_socat["socat_mask"].loc[dict(ylat=slice(-46, -36), xlon=slice(-97, -87))] = ref_val3
    add30p_socat["socat_mask"].loc[dict(ylat=slice(-45, -35), xlon=slice(-130, -120))] = ref_val3
    add30p_socat["socat_mask"].loc[dict(ylat=slice(-32, -27), xlon=slice(-141, -121))] = ref_val3
    add30p_socat["socat_mask"].loc[dict(ylat=slice(-77, -67), xlon=slice(-60, -40))] = ref_val3
    add30p_socat["socat_mask"].loc[dict(ylat=slice(-11, 9), xlon=slice(75, 85))] = ref_val3
    add30p_socat["socat_mask"].loc[dict(ylat=slice(-30, -25), xlon=slice(70, 90))] = ref_val4


    # Adding 52% more values to undersampled oceans, in currently unsampled locations
    add50p_oceans = non_zero_counts.copy(deep=True)
    add50p_socat = socat_mask_data.copy(deep=True)
    add50p_oceans.loc[dict(xlon=slice(-97, -87), ylat=slice(-46, -36))] = 125  # pacific1
    add50p_oceans.loc[dict(xlon=slice(-130, -110), ylat=slice(-45, -35))] = 125  # pacific2
    add50p_oceans.loc[dict(xlon=slice(-141, -121), ylat=slice(-32, -27))] = 125  # pacific3
    add50p_oceans.loc[dict(xlon=slice(-60, -40), ylat=slice(-77, -67))] = 125  # southern ocean
    add50p_oceans.loc[dict(xlon=slice(75, 85), ylat=slice(-11, 9))] = 125  # indian ocean1
    add50p_oceans.loc[dict(xlon=slice(70, 90), ylat=slice(-30, -25))] = 125  # indian ocean2

    ref_val5 = socat_mask_data["socat_mask"].sel(xlon=-76.5, ylat=25.5)
    ref_val6 = socat_mask_data["socat_mask"].sel(xlon=-77.5, ylat=25.5)

    add50p_socat["socat_mask"].loc[dict(ylat=slice(-46, -36), xlon=slice(-97, -87))] = ref_val5
    add50p_socat["socat_mask"].loc[dict(ylat=slice(-45, -35), xlon=slice(-130, -120))] = ref_val5
    add50p_socat["socat_mask"].loc[dict(ylat=slice(-32, -27), xlon=slice(-141, -121))] = ref_val5
    add50p_socat["socat_mask"].loc[dict(ylat=slice(-77, -67), xlon=slice(-60, -40))] = ref_val6
    add50p_socat["socat_mask"].loc[dict(ylat=slice(-11, 9), xlon=slice(75, 85))] = ref_val6
    add50p_socat["socat_mask"].loc[dict(ylat=slice(-30, -25), xlon=slice(70, 90))] = ref_val6

    return addmeanp_oceans, addmeanp_socat, add30p_oceans, add30p_socat, add50p_oceans, add50p_socat 


