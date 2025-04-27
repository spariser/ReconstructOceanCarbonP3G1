
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import gcsfs 


try:
    import cmocean.cm as cm
    DEFAULT_CMAP = cm.thermal # Keep the original if cmocean is available
except ImportError:
    print("Warning: cmocean not found. Using matplotlib's 'viridis' as default colormap.")
    from matplotlib import cm as cm_mpl
    DEFAULT_CMAP = cm_mpl.viridis # Fallback colormap


def plot_reconstruction_comparison(
    fs: gcsfs.GCSFileSystem,
    ensemble_name: str,
    member_name: str,
    ensemble_dir: str,
    recon_output_dir: str,
    mask_data_dict: dict,
    mask_name: str,
    chosen_time: str,
    init_date: str,
    fin_date: str,
    dates: list, # Or specify type more precisely, e.g., pd.DatetimeIndex
    plot_style: str = "seaborn-v0_8-talk",
    cmap = DEFAULT_CMAP,
    cbar_title: str = 'pCO₂ (µatm)',
    vrange: list = [280, 440],
    SpatialMap2 = None # Pass the class or an instance creator if needed
):
    """
    Plots a side-by-side comparison of original masked ESM pCO2 data
    and the reconstructed pCO2 data for a specific ensemble member and time.

    Parameters:
        fs (gcsfs.GCSFileSystem): Authenticated GCS filesystem object.
        ensemble_name (str): Name of the ensemble (e.g., 'ACCESS-ESM1-5').
        member_name (str): Name of the member (e.g., 'member_r10i1p1f1').
        ensemble_dir (str): Base GCS path to the raw ensemble output directories.
        recon_output_dir (str): Base GCS path to the reconstructed output directories.
        mask_data_dict (dict): Dictionary where keys are mask names (str) and
                               values are xarray Datasets containing 'socat_mask'.
        mask_name (str): The key from mask_data_dict specifying which mask to use.
        chosen_time (str): The time slice to plot (e.g., '2021-01').
        init_date (str): Start date string used in reconstructed filenames (e.g., '200401').
        fin_date (str): End date string used in reconstructed filenames (e.g., '202312').
        dates (list): Time range (e.g., list of datetime objects or pd.DatetimeIndex)
                      used for slicing the original member data.
        plot_style (str): Matplotlib style context.
        cmap: Matplotlib colormap or compatible object.
        cbar_title (str): Label for the colorbar.
        vrange (list): Min and max values for the color range [vmin, vmax].
        SpatialMap2: The SpatialMap2 class needed for plotting. Must be provided.

    Returns:
        matplotlib.figure.Figure: The figure object containing the plot.
    """
    if SpatialMap2 is None:
        raise ValueError("SpatialMap2 class must be provided via the 'SpatialMap2' argument.")

    # --- 1. Load Data ---
    # Load original member data
    member_dir = f"{ensemble_dir}/{ensemble_name}/{member_name}"
    # Use fs.glob to find the specific zarr file if the name isn't fixed
    try:
        member_path_pattern = f"{member_dir}/*.zarr"
        member_paths = fs.glob(member_path_pattern)
        if not member_paths:
            raise FileNotFoundError(f"No Zarr store found at {member_path_pattern}")
        # Assuming only one matching zarr store per member directory
        member_path = member_paths[0]
        print(f"Loading original data from: gs://{member_path}")
        # Ensure dates are correctly formatted for slicing if needed
        # Example assumes dates[0] and dates[-1] are suitable for str()
        member_data = xr.open_zarr(f'gs://{member_path}').sel(time=slice(str(dates[0]), str(dates[-1])))
    except Exception as e:
        print(f"Error loading original member data for {ensemble_name}/{member_name}: {e}")
        return None

    # Load reconstructed pCO₂ data
    recon_dir = f"{recon_output_dir}/{ensemble_name}/{member_name}"
    recon_fname = f"recon_pCO2_{ensemble_name}_{member_name}_mon_1x1_{init_date}_{fin_date}.zarr"
    recon_path = f"{recon_dir}/{recon_fname}"
    try:
        print(f"Loading reconstructed data from: {recon_path}")
        # Use consolidated=True if metadata is consolidated
        full = xr.open_zarr(f'gs://{recon_path}', consolidated=True)["pCO2_recon_full"]
    except Exception as e:
        print(f"Error loading reconstructed data from {recon_path}: {e}")
        return None

    # --- 2. Select Data and Mask for Plotting ---
    try:
        raw_data = member_data["spco2"].sel(time=chosen_time).squeeze()
        recon_data = full.sel(time=chosen_time).squeeze() # Assumes first dim is singleton if present
        if recon_data.ndim > 2: # Handle potential extra dimensions if squeeze didn't remove them
             recon_data = recon_data[0, ...] # Example: take the first slice if time was not squeezed

    except KeyError as e:
        print(f"Error selecting time '{chosen_time}': {e}. Check variable names and time coordinates.")
        return None
    except Exception as e:
         print(f"An unexpected error occurred during time selection: {e}")
         return None


    # Load and select the specified SOCAT mask
    if mask_name not in mask_data_dict:
        raise ValueError(f"Mask name '{mask_name}' not found in mask_data_dict keys: {list(mask_data_dict.keys())}")
    mask_dataset = mask_data_dict[mask_name]
    try:
        mask = mask_dataset["socat_mask"].sel(time=chosen_time).squeeze()
    except Exception as e:
        print(f"Error selecting time '{chosen_time}' from mask '{mask_name}': {e}")
        return None

    # --- 3. Prepare Data for Plotting ---
    # Shift longitudes from [0, 360] to [-180, 180] for global plotting
    # Check if rolling is necessary (if longitude is indeed 0-360)
    if raw_data['xlon'].min() >= 0:
        print("Rolling longitude for raw data...")
        raw_data = raw_data.roll(xlon=len(raw_data.xlon) // 2, roll_coords=True)
        raw_data['xlon'] = np.where(raw_data['xlon'] > 180, raw_data['xlon'] - 360, raw_data['xlon'])

    if recon_data['xlon'].min() >= 0:
        print("Rolling longitude for reconstructed data...")
        recon_data = recon_data.roll(xlon=len(recon_data.xlon) // 2, roll_coords=True)
        recon_data['xlon'] = np.where(recon_data['xlon'] > 180, recon_data['xlon'] - 360, recon_data['xlon'])

    if mask['xlon'].min() >= 0:
        print(f"Rolling longitude for mask '{mask_name}'...")
        mask = mask.roll(xlon=len(mask.xlon) // 2, roll_coords=True)
        mask['xlon'] = np.where(mask['xlon'] > 180, mask['xlon'] - 360, mask['xlon'])


    # Apply the selected mask to the original data
    # Ensure mask aligns with data after potential rolling
    try:
        # Reindex mask to match data coordinates precisely if necessary, using nearest neighbor
        mask_aligned = mask.reindex_like(raw_data, method='nearest')
        masked_raw = raw_data.where(mask_aligned == 1) # Use xarray's where for easier handling
    except Exception as e:
        print(f"Error applying mask '{mask_name}' to raw data: {e}")
        # Fallback or decide how to handle: plot unmasked raw? return error?
        # For now, let's create a masked array manually as a fallback like original code
        print("Attempting manual masking as fallback...")
        try:
             # This assumes coordinates align sufficiently after rolling
             masked_raw = np.ma.masked_array(raw_data.values, mask=(mask.values == 0))
        except Exception as inner_e:
             print(f"Manual masking also failed: {inner_e}. Cannot mask raw data.")
             masked_raw = raw_data # Plot unmasked as last resort


    # --- 4. Plotting ---
    with plt.style.context(plot_style):
        fig = plt.figure(figsize=(14, 5), dpi=150) # Adjusted size slightly
        worldmap = SpatialMap2(
            fig=fig, region='world',
            cbar_mode='single',
            colorbar=True,
            cbar_location='bottom',
            nrows_ncols=[1, 2]
        )

        # Plot original (masked) data
        sub0 = worldmap.add_plot(
            lon=raw_data['xlon'], lat=raw_data['ylat'], data=masked_raw,
            vrange=vrange, cmap=cmap, ax=0
        )
        # Plot reconstructed data
        sub1 = worldmap.add_plot(
            lon=recon_data['xlon'], lat=recon_data['ylat'], data=recon_data,
            vrange=vrange, cmap=cmap, ax=1
        )

        worldmap.set_title(f"Original pCO₂ ({chosen_time}, Mask: {mask_name})", ax=0, fontsize=13)
        worldmap.set_title(f"Reconstructed pCO₂ ({chosen_time})", ax=1, fontsize=13)

        colorbar = worldmap.add_colorbar(sub0, ax=0) # Add colorbar based on first plot
        worldmap.set_cbar_xlabel(colorbar, cbar_title, fontsize=12)

        plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust layout to prevent title overlap
        plt.show()

    return fig