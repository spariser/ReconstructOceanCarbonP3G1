import matplotlib.path as mpath
import matplotlib.cm as mpl_cm
import numpy as np
import xarray as xr
import cmocean as cm
import cartopy.crs as ccrs
import cartopy.feature
from cartopy.mpl.geoaxes import GeoAxes
from mpl_toolkits.axes_grid1 import AxesGrid
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import gcsfs


from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker


class SpatialMap2(object):
    """
    SpatialMap2 : class to plot plot nice spatial maps with a colorbar
                 correctly positioned in the figure

    Inputs
    ==============
    data     : Input 2D dataset [lon,lat] (default=None)
    lon      : longitude vector (default=np.arange(0.5,360,1))
    lat      : latitude vector (default=np.arange(-89.5,90,1))
    region   : 'world', 'southern-ocean' (default='world')
    fig      : figure handle (default=None)
    rect     : number of rows, columns, and position (default=111)
    cmap     : colormap (default=cm.cm.balance)
    colorbar : Toggle for colorbar (default=True)
    ncolors  : number of colors in colorbar (default=101)
    vrange   : colorbar range (default=[0,1])

    Returns
    ==============
    returns a colormap of your data within the specified region

    Methods
    ==============
    set_ticks()
    set_title()
    set_cbar_title()
    set_cbar_labels()

    Add at some point
    ==============
    # worldmap.cbar.ax.yaxis.set_ticks_position("left") # way to easily set tick location
    # worldmap.cbar.ax.yaxis.set_label_position('left') # set label position


    Example
    ==============
    # download WOA data
    ds = xr.open_dataset('https://data.nodc.noaa.gov/thredds/dodsC/ncei/woa/salinity/decav/1.00/woa18_decav_s00_01.nc', decode_times=False)
    data = ds['s_mn'].where(ds['depth']==0, drop=True).mean(['time','depth'])
    # plot spatial map
    worldmap = SpatialMap(data, lon=ds['lon'], lat=ds['lat'], fig=plt.figure(figsize=(7,7)), vrange=[30, 37], region='world')

    """

    def __init__(
        self,
        nrows_ncols=(1, 1),
        region="world",
        fig=None,
        rect=[1, 1, 1],
        colorbar=True,
        cbar_location="bottom",
        cbar_mode="single",
        cbar_orientation="horizontal",
        cbar_size="7%",
        cbar_pad=0.1,
        axes_pad=0.2,
    ):
        # cmap=cm.cm.balance,
        # ncolors=101,
        # vrange = [0, 1]):

        self.region = region
        self.cbar_orientation = cbar_orientation
        # self.vrange = vrange
        # self.ncolors = ncolors
        # self.cmap = cmap

        ### Setup figure and axes
        if fig is None:
            fig = plt.figure(figsize=(8.5, 11))

        # Define projection
        if self.region.upper() == "SOUTHERN-OCEAN":
            projection = ccrs.SouthPolarStereo()

        if self.region.upper() == "WORLD":
            projection = ccrs.Robinson(central_longitude=180)
            # projection=ccrs.InterruptedGoodeHomolosine(central_longitude=0)
            # projection=ccrs.Miller(central_longitude=0)

        # Setup axesgrid
        axes_class = (GeoAxes, dict(projection=projection))
        self.grid = AxesGrid(
            fig,
            rect=rect,
            axes_class=axes_class,
            share_all=False,
            nrows_ncols=nrows_ncols,
            axes_pad=axes_pad,
            cbar_location=cbar_location,
            cbar_mode=cbar_mode if colorbar == True else None,
            cbar_pad=cbar_pad if colorbar == True else None,
            cbar_size=cbar_size,
            label_mode="all",
        )  # note the empty label_mode

    def add_plot(
        self,
        lon=None,
        lat=None,
        data=None,
        ax=None,
        land=True,
        coastline=True,
        linewidth_coast=0.25,
        ncolors=101,
        vrange=[-25, 25],
        cmap=cm.cm.balance,
        facecolor=[0.25, 0.25, 0.25],
        *args,
        **kwargs,
    ):
        """
        add_plot(lon, lat, data, **kwargs)

        Inputs:
        ==============
        sub : subplot (this is returuned from add_plot())
        ax. : axis number to add colorbar to

        """

        self.vrange = vrange
        self.ncolors = ncolors
        self.cmap = cmap

        ### Set Longitude if none is given
        if lon is None:
            self.lon = np.arange(0.5, 360, 1)
        else:
            self.lon = lon

        ### Set latitude if none is given
        if lat is None:
            self.lat = np.arange(-89.5, 90, 1)
        else:
            self.lat = lat

        self.transform = ccrs.PlateCarree(central_longitude=0)
        self.bounds = np.linspace(self.vrange[0], self.vrange[1], self.ncolors)
        self.norm = mpl.colors.BoundaryNorm(self.bounds, self.cmap.N)

        # Define southern ocean region
        if self.region.upper() == "SOUTHERN-OCEAN":
            # Compute a circle in axes coordinates, which we can use as a boundary
            # for the map. We can pan/zoom as much as we like - the boundary will be
            # permanently circular.
            theta = np.linspace(0, 2 * np.pi, 100)
            center, radius = [0.5, 0.5], 0.5
            verts = np.vstack([np.sin(theta), np.cos(theta)]).T
            circle = mpath.Path(verts * radius + center)

            # Set extent
            self.grid[ax].set_boundary(circle, transform=self.grid[ax].transAxes)

            # Limit the map to -60 degrees latitude and below.
            self.grid[ax].set_extent([-180, 180, -90, -35], ccrs.PlateCarree())

        ### land mask
        # Add Contintents
        if land is True:
            self.grid[ax].add_feature(
                cfeature.NaturalEarthFeature(
                    "physical", "land", "110m", edgecolor="None", facecolor=facecolor
                )
            )

        ## add Coastline
        if coastline is True:
            self.grid[ax].coastlines(facecolor=facecolor, linewidth=linewidth_coast)

        sub = self.grid[ax].pcolormesh(
            self.lon,
            self.lat,
            data,
            norm=self.norm,
            transform=self.transform,
            # vmin = self.vrange[0],vmax = self.vrange[1]
            cmap=self.cmap,
            *args,
            **kwargs,
        )
        return sub

    # def add_colorbar(self, sub, ax=0, *args, **kwargs):
    #     """
    #     add_colorbar(sub, ax, **kwargs)

    #     Inputs:
    #     ==============
    #     sub : subplot (this is returuned from add_plot())
    #     ax. : axis number to add colorbar to

    #     """
    #     # Weird whitespace when you use 'extend'
    #     # The workaround is to make a colorbar
    #     # Help from : https://github.com/matplotlib/matplotlib/issues/9778

    #     # col = self.grid.cbar_axes[ax].colorbar(sub, *args, **kwargs)
    #     col = mpl.colorbar.ColorbarBase(
    #         self.grid.cbar_axes[ax],
    #         orientation=self.cbar_orientation,
    #         cmap=self.cmap,
    #         norm=mpl.colors.Normalize(vmin=self.vrange[0], vmax=self.vrange[1]),
    #         *args,
    #         **kwargs,
    #     )

    #     # cb2 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
    #     #                                norm=norm,
    #     #                                boundaries=[0] + bounds + [13],
    #     #                                extend='both',
    #     #                                ticks=bounds,
    #     #                                spacing='proportional',
    #     #                                orientation='horizontal')

    #     return col
    def add_colorbar(self, sub, ax=0, cmap=None, vrange=None, *args, **kwargs):
        """
        Add a colorbar to a subplot.

        Inputs:
        ==============
        sub : artist returned from add_plot()
        ax  : index of the subplot to add the colorbar to
        cmap: colormap to use (if None, use self.cmap)
        vrange: value range [vmin, vmax] (if None, use self.vrange)
        """
        if cmap is None:
            cmap = self.cmap
        if vrange is None:
            vrange = self.vrange

        col = mpl.colorbar.ColorbarBase(
            self.grid.cbar_axes[ax],
            orientation=self.cbar_orientation,
            cmap=cmap,
            norm=mpl.colors.Normalize(vmin=vrange[0], vmax=vrange[1]),
            *args,
            **kwargs,
        )

        return col

    ### Class methods
    def set_ticks(self, col, tmin, tmax, dt, *args, **kwargs):
        """
        set_ticks(tmin,tmax,dt, **kwargs)

        Inputs:
        ==============
        tmin : min tick value
        tmax : max tick value
        dt.  : delta tick value

        """
        # col.cbar_axis.set_ticks(np.arange(tmin, tmax+dt, dt), *args, **kwargs)
        col.set_ticks(ticks=np.arange(tmin, tmax + dt, dt), *args, **kwargs)

    def set_title(self, title, ax, *args, **kwargs):
        """
        set_title(title, *args, **kwargs)

        Inputs:
        ==============
        title : title value

        """
        self.grid[ax].set_title(title, *args, **kwargs)

    def set_cbar_title(self, col, title, *args, **kwargs):
        """
        set_cbar_title(title, *args, **kwargs)

        Inputs:
        ==============
        title : colorbar title value

        """
        col.ax.set_title(title, *args, **kwargs)

    def set_cbar_ylabel(self, col, ylabel, *args, **kwargs):
        """
        set_cbar_ylabel(title, *args, **kwargs)

        Inputs:
        ==============
        title : colorbar title value

        """
        col.ax.set_ylabel(ylabel, *args, **kwargs)

    def set_cbar_xlabel(self, col, ylabel, *args, **kwargs):
        """
        set_cbar_xlabel(title, *args, **kwargs)

        Inputs:
        ==============
        title : colorbar title value

        """
        col.ax.set_xlabel(ylabel, *args, **kwargs)

    def set_cbar_xticklabels(self, col, labels, *args, **kwargs):
        """
        set_cbar_labels(labels, *args, **kwargs)

        Inputs:
        ==============
        labels : custom colorbar labels

        """
        col.ax.set_xticklabels(labels, *args, **kwargs)

    def set_cbar_yticklabels(self, col, labels, *args, **kwargs):
        """
        set_cbar_labels(labels, *args, **kwargs)

        Inputs:
        ==============
        labels : custom colorbar labels

        """
        col.ax.set_yticklabels(labels, *args, **kwargs)


from cartopy.util import add_cyclic_point


def xr_add_cyclic_point(data, cyclic_coord=None):
    """
    cyclic_point : a wrapper for catopy's apply_ufunc

    Inputs
    =============
    data         : dataSet you want to add cyclic point to
    cyclic_coord : coordinate to apply cyclic to

    Returns
    =============
    cyclic_data : returns dataset with cyclic point added

    """
    return xr.apply_ufunc(
        add_cyclic_point,
        data.load(),
        input_core_dims=[[cyclic_coord]],
        output_core_dims=[["tmp_new"]],
    ).rename({"tmp_new": cyclic_coord})


####### Custom Functions created by Group 1 for Project 3 Data Story


def plot_mask(mask, title, vrange=[0, 144, 12]):
    """
    Plots a mask on a world map using a specified colormap and saves the figure.

    Parameters:
    -----------
    mask : xarray.DataArray
        The data array containing the mask to be plotted. It should have
        coordinates 'xlon' (longitude) and 'ylat' (latitude).
    title : str
        The title of the plot, which will also be used as part of the saved
        figure's filename.
    """
    with plt.style.context("seaborn-v0_8-talk"):
        fig = plt.figure(figsize=(8.5, 11))  # fig = plt.figure(dpi=300)
        worldmap = SpatialMap2(
            fig=fig,
            region="world",
            cbar_mode="each",
            colorbar=True,
            cbar_location="bottom",
            nrows_ncols=[1, 1],
        )

        cmap = cm.cm.rain
        data = xr_add_cyclic_point(mask, cyclic_coord="xlon")
        data = data.assign_coords(xlon=((data.xlon + 180) % 360))
        sub = worldmap.add_plot(
            lon=data["xlon"],
            lat=data["ylat"],
            data=data,
            vrange=vrange[0:2],
            cmap=cmap,
            ax=0,
            linewidth_coast=0.5,
        )

        col = worldmap.add_colorbar(sub, ax=0, extend="max")
        worldmap.set_cbar_xlabel(col, "Number of months with data", fontsize=14)
        worldmap.set_ticks(col, vrange[0], vrange[1], vrange[2])
        col.ax.tick_params(labelsize=12)
        worldmap.set_title(title, ax=0, fontsize=14)
        plt.show()
        # save figure
        # fig.savefig(f"mask_{title}.png", dpi=300, bbox_inches="tight")
def plot_maskd(mask, title, vrange=[0, 144, 12]):
    """
    Plots a mask on a world map using a specified colormap and saves the figure.

    Parameters:
    -----------
    mask : xarray.DataArray
        The data array containing the mask to be plotted. It should have
        coordinates 'xlon' (longitude) and 'ylat' (latitude).
    title : str
        The title of the plot, which will also be used as part of the saved
        figure's filename.
    """
    with plt.style.context("seaborn-v0_8-talk"):
        fig = plt.figure(figsize=(8.5, 11))  # fig = plt.figure(dpi=300)
        worldmap = SpatialMap2(
            fig=fig,
            region="world",
            cbar_mode="each",
            colorbar=True,
            cbar_location="bottom",
            nrows_ncols=[1, 1],
        )

        cmap = cm.cm.rain
        data = xr_add_cyclic_point(mask, cyclic_coord="xlon")
        data = data.assign_coords(xlon=((data.xlon + 180) % 360))
        sub = worldmap.add_plot(
            lon=data["xlon"],
            lat=data["ylat"],
            data=data,
            vrange=vrange[0:2],
            cmap=cmap,
            ax=0,
            linewidth_coast=0.5,
        )

        col = worldmap.add_colorbar(sub, ax=0, extend="max")
        worldmap.set_cbar_xlabel(col, "Change in number of months with data", fontsize=14)
        worldmap.set_ticks(col, vrange[0], vrange[1], vrange[2])
        col.ax.tick_params(labelsize=12)
        worldmap.set_title(title, ax=0, fontsize=14)
        plt.show()

def plot_reconstruction_vs_truth(
    mask_name,
    mask_data_dict,
    selected_mems_dict,
    ensemble_dir,
    output_dir,
    dates,  # Used to define time slice for truth data
    init_date,  # Used for finding recon file path
    fin_date,  # Used for finding recon file path
    # --- MODIFIED DEFAULTS ---
    vrange=[340, 420],  # Adjusted default vrange for typical mean pCO2 (µatm)
    cmap_data="viridis",  # Default colormap suitable for magnitude data
    plot_style="seaborn-v0_8-talk",  # Default plot style
    mask_threshold=0.01,  # Threshold for applying the averaged mask
    # --- END MODIFIED DEFAULTS ---
    # chosen_time parameter removed
):
    """
    Plots the comparison between the time-averaged reconstructed pCO₂ data
    and the time-averaged original (truth) data (masked by average sampling).

    Parameters:
    ----------
    mask_name : str
        Name of the mask scenario used for reconstruction & defining sampling.
    mask_data_dict : dict
        Dictionary containing mask datasets (needs 'socat_mask' variable).
        Expected structure: {'mask_name': xr.Dataset_or_DataArray}
    selected_mems_dict : dict
        Dictionary specifying selected ensemble members for reconstruction path.
    ensemble_dir : str
        Path to the directory containing the original ensemble (truth) data.
    output_dir : str
        Path to the output directory for reconstructions.
    dates : list or array-like
        Sequence of dates defining the time range for the truth data (e.g., from np.datetime64).
    init_date : str
        Initial date string defining the reconstruction period in the file path.
    fin_date : str
        Final date string defining the reconstruction period in the file path.
    vrange : list, optional
        Value range for the pCO₂ color scale. Default is [340, 420].
    cmap_data : str, optional
        Colormap name for the pCO₂ plots. Default is 'viridis'.
    plot_style : str, optional
        Matplotlib plot style to use. Default is 'seaborn-v0_8-talk'.
    mask_threshold : float, optional
        Minimum average sampling presence required to show truth data.
        Areas where `mask_avg < mask_threshold` will be masked out. Default is 0.01.
    """
    # Get the colormap object using the mpl_cm alias
    try:
        cmap_object = mpl_cm.get_cmap(cmap_data)
    except ValueError as e:
        print(f"Error getting colormap: {e}. Check if the name '{cmap_data}' is valid.")
        return
    except AttributeError:
        print(f"Error: Could not retrieve colormap '{cmap_data}' using mpl_cm.")
        return

    # --- Data Loading ---
    truth_avg, recon_avg, mask_avg = None, None, None  # Initialize
    lon_coords, lat_coords = None, None  # Initialize

    try:
        # --- Get Path Components ---
        first_ens = list(selected_mems_dict.keys())[0]
        first_mem = selected_mems_dict[first_ens][0]

        # --- Load Full Time Series: Truth Data ---
        print("Loading truth data...")
        fs = gcsfs.GCSFileSystem()  # Initialize GCS filesystem object
        member_dir = f"{ensemble_dir}/{first_ens}/{first_mem}"
        # Use fs.glob to find the zarr store; requires directory path exists
        glob_path = f"{member_dir}/*.zarr"
        print(f"  Globbing for truth data in: gs://{glob_path}")
        member_paths = fs.glob(glob_path)
        if not member_paths:
            raise FileNotFoundError(f"No .zarr files found matching gs://{glob_path}")
        member_path = member_paths[0]  # Take the first match
        print(f"  Found truth data path: gs://{member_path}")
        # Load and select time slice
        truth_data_full = xr.open_zarr(f"gs://{member_path}", consolidated=True)[
            "spco2"
        ].sel(
            time=slice(
                str(dates[0]), str(dates[-1])
            )  # Select full time range based on 'dates'
        )
        if "time" not in truth_data_full.dims:
            raise ValueError("Truth data missing 'time' dimension.")

        # --- Load Full Time Series: Reconstruction Data ---
        print("Loading reconstruction data...")
        recon_output_dir = f"{output_dir}/reconstructions/{mask_name}"
        recon_dir = f"{recon_output_dir}/{first_ens}/{first_mem}"
        recon_path = f"{recon_dir}/recon_pCO2_{first_ens}_{first_mem}_mon_1x1_{init_date}_{fin_date}.zarr"
        print(f"  Recon path: {recon_path}")  # Note: Assumes local path or mounted GCS
        recon_data_full_members = xr.open_zarr(recon_path, consolidated=True)[
            "pCO2_recon_full"
        ]
        if "time" not in recon_data_full_members.dims:
            raise ValueError("Reconstruction data missing 'time' dimension.")

        # Handle potential member dimension BEFORE averaging
        member_dim_name = "member"
        if member_dim_name in recon_data_full_members.dims:
            if recon_data_full_members.sizes[member_dim_name] > 1:
                print(f"  Selecting first member '{first_mem}' from reconstruction.")
                recon_data_full = recon_data_full_members.sel(
                    {member_dim_name: first_mem}, drop=True
                )  # Or use .isel({member_dim_name:0})
            elif recon_data_full_members.sizes[member_dim_name] == 1:
                print(f"  Squeezing single member dimension from reconstruction.")
                recon_data_full = recon_data_full_members.squeeze(
                    dim=member_dim_name, drop=True
                )
            else:  # Size 0 ?
                recon_data_full = recon_data_full_members  # Should not happen often
        else:
            print("  No member dimension found in reconstruction.")
            recon_data_full = recon_data_full_members  # Assume (time, lat, lon)

        # --- Load Full Time Series: Mask Data ---
        print("Loading mask data...")
        try:
            # Access the specific mask dataset/dataarray from the dictionary
            mask_dataset_or_da = mask_data_dict[mask_name]
            # Check if it's a Dataset or DataArray containing the mask
            if isinstance(mask_dataset_or_da, xr.Dataset):
                mask_data_full = mask_dataset_or_da["socat_mask"]
            elif isinstance(mask_dataset_or_da, xr.DataArray):
                mask_data_full = mask_dataset_or_da  # Assume it is the mask itself
            else:
                raise TypeError(
                    f"mask_data_dict['{mask_name}'] is not an xarray Dataset or DataArray."
                )
        except KeyError:
            raise KeyError(
                f"Mask name '{mask_name}' not found as a key in mask_data_dict."
            )

        if "time" not in mask_data_full.dims:
            raise ValueError("Mask data missing 'time' dimension.")

        # --- Time Averaging ---
        print("Calculating time averages...")
        truth_avg = truth_data_full.mean(dim="time", skipna=True).squeeze(drop=True)
        recon_avg = recon_data_full.mean(dim="time", skipna=True).squeeze(
            drop=True
        )  # Squeeze again just in case
        mask_avg = mask_data_full.mean(dim="time", skipna=True).squeeze(drop=True)

        # Verify results are 2D
        assert len(truth_avg.dims) == 2, (
            f"Time-averaged truth is not 2D: {truth_avg.dims}"
        )
        assert len(recon_avg.dims) == 2, (
            f"Time-averaged reconstruction is not 2D: {recon_avg.dims}"
        )
        assert len(mask_avg.dims) == 2, f"Time-averaged mask is not 2D: {mask_avg.dims}"
        print("  Averaging and dimension checks complete.")

        # Get coordinates for plotting
        lon_coords = (
            truth_avg["xlon"] if "xlon" in truth_avg.coords else truth_avg["lon"]
        )
        lat_coords = (
            truth_avg["ylat"] if "ylat" in truth_avg.coords else truth_avg["lat"]
        )

        # --- Align Longitudes (on the averaged data) ---
        print("Aligning longitudes...")
        lon_coord_name = (
            "xlon"
            if "xlon" in truth_avg.dims
            else "lon"
            if "lon" in truth_avg.dims
            else None
        )
        if lon_coord_name:
            lon_size = len(truth_avg[lon_coord_name])

            def adjust_lon(ds, lon_name, size):
                ds_rolled = ds.roll(**{lon_name: size // 2}, roll_coords=True)
                ds_rolled[lon_name] = (ds_rolled[lon_name] + 180) % 360 - 180
                return ds_rolled.sortby(lon_name)

            # Apply adjustment
            truth_avg = adjust_lon(truth_avg, lon_coord_name, lon_size)
            recon_avg = adjust_lon(recon_avg, lon_coord_name, lon_size)
            mask_avg = adjust_lon(mask_avg, lon_coord_name, lon_size)
            # Update coordinate variables if needed
            lon_coords = truth_avg[lon_coord_name]
        else:
            print("Warning: Could not find standard longitude coordinate for rolling.")

        # --- Mask the Averaged Truth Data ---
        print(
            f"Masking averaged truth data where average mask presence < {mask_threshold}..."
        )
        # Create a boolean mask: True where average mask value is below threshold
        mask_condition = mask_avg < mask_threshold
        # Apply the mask using numpy's masked array or xarray's .where()
        # Using np.ma.masked_array: requires numpy arrays
        masked_truth_avg_np = np.ma.masked_array(
            truth_avg.values, mask=mask_condition.values
        )
        # Or using xarray.where (keeps it as DataArray, often preferred):
        # masked_truth_avg_xr = truth_avg.where(mask_avg >= mask_threshold) # Keep where mask is sufficient
        # Let's use the numpy version as the original code did, but on averaged data
        masked_truth_avg = masked_truth_avg_np

    except FileNotFoundError as e:
        print(f"Error: Data file not found.\n{e}")
        return  # Stop execution
    except KeyError as e:
        print(f"Error: Variable or Mask Name {e} not found. Check keys/names.")
        return
    except ValueError as e:
        print(f"Error in data dimension or structure: {e}")
        return
    except TypeError as e:
        print(f"Error with mask data type: {e}")
        return
    except AssertionError as e:
        print(f"Error: Data dimension assertion failed after processing. {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during data loading or processing: {e}")
        return

    # --- Plotting ---
    print("Generating side-by-side plot...")
    try:
        # Check if SpatialMap2 class is available
        if "SpatialMap2" not in globals():
            raise NameError("SpatialMap2 class is not defined or imported.")

        with plt.style.context(plot_style):
            fig = plt.figure(figsize=(12, 4), dpi=200)  # Adjusted figsize
            worldmap = SpatialMap2(
                fig=fig,
                region="world",
                cbar_mode="single",  # Single colorbar for comparison
                colorbar=True,
                cbar_location="bottom",
                nrows_ncols=[1, 2],  # 1 row, 2 columns
            )

            # Plot Time-Averaged, Masked Truth Data on left axis (ax=0)
            sub0 = worldmap.add_plot(
                lon=lon_coords,  # Use aligned lon coords
                lat=lat_coords,  # Use lat coords
                data=masked_truth_avg,  # Plot the masked numpy array
                vrange=vrange,  # Use the specified value range
                cmap=cmap_object,  # Use the specified colormap object
                ax=0,  # Target the first subplot
            )

            # Plot Time-Averaged Reconstructed Data on right axis (ax=1)
            sub1 = worldmap.add_plot(
                lon=lon_coords,  # Use the same coordinates
                lat=lat_coords,
                data=recon_avg,  # Plot the averaged reconstruction DataArray
                vrange=vrange,  # Use the same value range
                cmap=cmap_object,  # Use the same colormap
                ax=1,  # Target the second subplot
            )

            # Set titles indicating time average
            worldmap.set_title(
                f"Mean Truth pCO₂ (Mask: {mask_name})", ax=0, fontsize=13
            )
            worldmap.set_title(
                f"Mean Reconstruction pCO₂ ({mask_name})", ax=1, fontsize=13
            )

            # Add and label the single colorbar
            try:
                colorbar = worldmap.add_colorbar(sub0, ax=0)  # Add based on one plot
                worldmap.set_cbar_xlabel(colorbar, "Mean pCO₂ (µatm)", fontsize=12)
            except Exception as cbar_err:
                print(
                    f"Note: Could not automatically add single colorbar. Error: {cbar_err}"
                )

            plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Adjust layout for colorbar
            plt.show()
            print("Plotting complete.")

    except NameError as e:
        print(f"Plotting Error: {e}. Ensure the SpatialMap2 class is defined/imported.")
        if "fig" in locals() and plt.fignum_exists(fig.number):
            plt.close(fig)
    except Exception as e:
        print(f"An unexpected error occurred during plotting: {e}")
        if "fig" in locals() and plt.fignum_exists(fig.number):
            plt.close(fig)


def plot_reconstruction_std(
    mask_name,
    selected_mems_dict,
    output_dir,
    init_date,
    fin_date,
    chosen_time="2021-01",
    vrange=[280, 440],
):
    """
    Plots the standard deviation of reconstructed pCO₂ data for a specific time period
    using NGBoost predictions.

    Parameters:
    -----------
    mask_name : str
        Name of the mask used for the reconstruction.
    selected_mems_dict : dict
        Dictionary containing ensemble names as keys and lists of member names as values.
    output_dir : str
        Path to the output directory where reconstruction data is stored.
    init_date : str
        Initial date of the reconstruction period in the format 'YYYY-MM'.
    fin_date : str
        Final date of the reconstruction period in the format 'YYYY-MM'.
    chosen_time : str, optional
        Specific time (month) to extract and plot, in the format 'YYYY-MM'. Default is "2021-01".
    vrange : list, optional
        Value range for the color scale in the plot. Default is [280, 440].

    Notes:
    ------
    This function assumes that the standard deviation of the reconstructed pCO₂ data
    is derived from NGBoost's probabilistic predictions. NGBoost provides uncertainty
    estimates (e.g., standard deviation) for each prediction, which can be visualized
    to assess the variability or confidence in the reconstruction.
    """
    cmap = cm.cm.thermal

    # Select the first ensemble and member
    first_ens = list(selected_mems_dict.keys())[0]
    first_mem = selected_mems_dict[first_ens][0]

    # Load reconstructed pCO₂ data
    recon_output_dir = f"{output_dir}/reconstructions/{mask_name}"
    recon_dir = f"{recon_output_dir}/{first_ens}/{first_mem}"
    recon_path = f"{recon_dir}/recon_pCO2_{first_ens}_{first_mem}_mon_1x1_{init_date}_{fin_date}.zarr"
    print("Recon path:", recon_path)
    full = xr.open_zarr(recon_path, consolidated=True)["pCO2_recon_full_std"]

    # Extract specific month
    recon_data = full.sel(time=chosen_time)[0, ...]

    # Shift longitudes for global plotting
    recon_data = recon_data.roll(xlon=len(recon_data.xlon) // 2, roll_coords=True)

    # Start plotting
    with plt.style.context("seaborn-v0_8-talk"):
        fig = plt.figure(figsize=(8, 3), dpi=200)
        worldmap = SpatialMap2(
            fig=fig,
            region="world",
            cbar_mode="single",
            colorbar=True,
            cbar_location="bottom",
            nrows_ncols=[1, 1],
        )

        sub1 = worldmap.add_plot(
            lon=recon_data["xlon"],
            lat=recon_data["ylat"],
            data=recon_data,
            vrange=vrange,
            cmap=cmap,
            ax=0,
        )

        worldmap.set_title(
            f"pCO₂ STD Reconstruction ({chosen_time})", ax=0, fontsize=13
        )

        colorbar = worldmap.add_colorbar(sub1, ax=0)
        worldmap.set_cbar_xlabel(colorbar, "pCO₂ (µatm)", fontsize=12)

        plt.show()


###


def plot_reconstruction_std_single(
    mask_name,
    selected_mems_dict,
    output_dir,
    init_date,
    fin_date,
    vrange=[0, 30],
    cmap_std="viridis",
    plot_style="seaborn-v0_8-talk",
):
    """
    Plot the time-averaged standard deviation (STD) of reconstructed pCO₂
    for a single masking strategy over the entire period.

    Parameters
    ----------
    mask_name : str
        Name of the mask used for the reconstruction.
    selected_mems_dict : dict
        Dictionary containing ensemble names as keys and lists of member names as values.
    output_dir : str
        Path to the output directory where reconstruction data is stored.
    init_date : str
        Initial date string defining the reconstruction period in the file path.
    fin_date : str
        Final date string defining the reconstruction period in the file path.
    vrange : list, optional
        Value range for the color scale in the plot. Default is [0, 30].
    cmap_std : str, optional
        Colormap name for the STD plots. Default is 'viridis'.
    plot_style : str, optional
        Matplotlib plot style to use. Default is 'seaborn-v0_8-talk'.
    """

    # Get the colormap object
    try:
        cmap_object = mpl_cm.get_cmap(cmap_std)
    except Exception as e:
        print(f"Error retrieving colormap '{cmap_std}': {e}")
        return

    # --- Data Loading ---
    try:
        first_ens = list(selected_mems_dict.keys())[0]
        first_mem = selected_mems_dict[first_ens][0]
        base_path = f"{output_dir}/reconstructions/{mask_name}/{first_ens}/{first_mem}/recon_pCO2_{first_ens}_{first_mem}_mon_1x1_{init_date}_{fin_date}.zarr"

        print(f"Loading STD from: {base_path}")
        full_std = xr.open_zarr(base_path, consolidated=True)["pCO2_recon_full_std"]

        if "time" not in full_std.dims:
            raise ValueError(f"{mask_name} STD data missing 'time' dimension.")

        # Time average
        std_avg = full_std.mean(dim="time", skipna=True).squeeze(drop=True)

        # Coordinates
        lon_coords = std_avg["xlon"] if "xlon" in std_avg.coords else std_avg["lon"]
        lat_coords = std_avg["ylat"] if "ylat" in std_avg.coords else std_avg["lat"]

        # Align Longitudes
        lon_coord_name = (
            "xlon"
            if "xlon" in std_avg.dims
            else "lon"
            if "lon" in std_avg.dims
            else None
        )
        if lon_coord_name:
            lon_size = len(std_avg[lon_coord_name])

            def adjust_lon(ds, lon_name, size):
                ds_rolled = ds.roll(**{lon_name: size // 2}, roll_coords=True)
                ds_rolled[lon_name] = (ds_rolled[lon_name] + 180) % 360 - 180
                return ds_rolled.sortby(lon_name)

            std_avg = adjust_lon(std_avg, lon_coord_name, lon_size)
            lon_coords = std_avg[lon_coord_name]
        else:
            print("Warning: Longitude coordinate not found for rolling.")

    except Exception as e:
        print(f"Error loading or processing data: {e}")
        return

    # --- Plotting ---
    print("Generating plot...")
    try:
        if "SpatialMap2" not in globals():
            raise NameError("SpatialMap2 class is not defined or imported.")

        with plt.style.context(plot_style):
            fig = plt.figure(figsize=(6, 4), dpi=200)
            worldmap = SpatialMap2(
                fig=fig,
                region="world",
                cbar_mode="single",
                colorbar=True,
                cbar_location="bottom",
                nrows_ncols=[1, 1],  # Single plot
            )

            sub = worldmap.add_plot(
                lon=lon_coords,
                lat=lat_coords,
                data=std_avg,
                vrange=vrange,
                cmap=cmap_object,
                ax=0,
            )

            worldmap.set_title(f"Mean STD {mask_name}", ax=0, fontsize=13)

            try:
                colorbar = worldmap.add_colorbar(sub, ax=0)
                worldmap.set_cbar_xlabel(colorbar, "Mean STD pCO₂ (µatm)", fontsize=12)
            except Exception as cbar_err:
                print(f"Colorbar issue: {cbar_err}")

            plt.tight_layout(rect=[0, 0.1, 1, 0.95])
            plt.show()
            print("Plotting complete.")

    except Exception as e:
        print(f"Error during plotting: {e}")
        if "fig" in locals() and plt.fignum_exists(fig.number):
            plt.close(fig)


###


def plot_reconstruction_std_side_by_side(
    mask_name_1,
    mask_name_2,
    selected_mems_dict,
    output_dir,
    init_date,  # Used for finding recon file path
    fin_date,  # Used for finding recon file path
    # chosen_time parameter removed
    # --- MODIFIED DEFAULTS ---
    vrange=[0, 30],  # Default vrange suitable for pCO2 STD (µatm)
    cmap_std="viridis",  # Default colormap suitable for STD (sequential)
    plot_style="seaborn-v0_8-talk",  # Default plot style
    # --- END MODIFIED DEFAULTS ---
):
    """
    Plot the time-averaged standard deviation (STD) of reconstructed pCO₂
    side-by-side for two masking strategies over the entire period.

    Parameters:
    ----------
    mask_name_1 : str
        Name of the first mask used for the reconstruction.
    mask_name_2 : str
        Name of the second mask used for the reconstruction.
    selected_mems_dict : dict
        Dictionary containing ensemble names as keys and lists of member names as values.
    output_dir : str
        Path to the output directory where reconstruction data is stored.
    init_date : str
        Initial date string defining the reconstruction period in the file path.
    fin_date : str
        Final date string defining the reconstruction period in the file path.
    vrange : list, optional
        Value range for the color scale in the plot. Default is [0, 30].
    cmap_std : str, optional
        Colormap name for the STD plots. Default is 'viridis'.
    plot_style : str, optional
        Matplotlib plot style to use. Default is 'seaborn-v0_8-talk'.
    """
    # Get the colormap object using the mpl_cm alias
    try:
        cmap_object = mpl_cm.get_cmap(cmap_std)
    except ValueError as e:
        print(f"Error getting colormap: {e}. Check if the name '{cmap_std}' is valid.")
        return
    except AttributeError:
        print(f"Error: Could not retrieve colormap '{cmap_std}' using mpl_cm.")
        return

    # --- Data Loading ---
    std1_avg, std2_avg = None, None  # Initialize
    lon_coords, lat_coords = None, None  # Initialize

    try:
        # Select the first ensemble and member to construct file paths
        first_ens = list(selected_mems_dict.keys())[0]
        first_mem = selected_mems_dict[first_ens][0]
        base_path_str = f"{output_dir}/reconstructions/{{mask_name}}/{first_ens}/{first_mem}/recon_pCO2_{first_ens}_{first_mem}_mon_1x1_{init_date}_{fin_date}.zarr"

        # --- Process Mask 1 ---
        path1 = base_path_str.format(mask_name=mask_name_1)
        print(f"Loading and processing {mask_name_1} STD from: {path1}")
        # Load the *full* time series of the STD
        full_std_1 = xr.open_zarr(path1, consolidated=True)["pCO2_recon_full_std"]
        if "time" not in full_std_1.dims:
            raise ValueError(f"{mask_name_1} STD data missing 'time' dimension.")
        # Calculate the mean over time and remove singleton dimensions
        std1_avg = full_std_1.mean(dim="time", skipna=True).squeeze(drop=True)
        # Verify result is 2D
        assert len(std1_avg.dims) == 2, (
            f"Time-averaged STD for {mask_name_1} is not 2D: {std1_avg.dims}"
        )
        print(f"  Finished processing {mask_name_1}. Dimensions: {std1_avg.dims}")
        # Get coordinates for plotting (once is enough)
        lon_coords = std1_avg["xlon"] if "xlon" in std1_avg.coords else std1_avg["lon"]
        lat_coords = std1_avg["ylat"] if "ylat" in std1_avg.coords else std1_avg["lat"]

        # --- Process Mask 2 ---
        path2 = base_path_str.format(mask_name=mask_name_2)
        print(f"Loading and processing {mask_name_2} STD from: {path2}")
        # Load the *full* time series of the STD
        full_std_2 = xr.open_zarr(path2, consolidated=True)["pCO2_recon_full_std"]
        if "time" not in full_std_2.dims:
            raise ValueError(f"{mask_name_2} STD data missing 'time' dimension.")
        # Calculate the mean over time and remove singleton dimensions
        std2_avg = full_std_2.mean(dim="time", skipna=True).squeeze(drop=True)
        # Verify result is 2D
        assert len(std2_avg.dims) == 2, (
            f"Time-averaged STD for {mask_name_2} is not 2D: {std2_avg.dims}"
        )
        print(f"  Finished processing {mask_name_2}. Dimensions: {std2_avg.dims}")

        # --- Align Longitudes (on the averaged data) ---
        print("Aligning longitudes...")
        lon_coord_name = (
            "xlon"
            if "xlon" in std1_avg.dims
            else "lon"
            if "lon" in std1_avg.dims
            else None
        )
        if lon_coord_name:
            lon_size = len(std1_avg[lon_coord_name])

            # Function to roll, adjust coords, and sort
            def adjust_lon(ds, lon_name, size):
                ds_rolled = ds.roll(**{lon_name: size // 2}, roll_coords=True)
                ds_rolled[lon_name] = (ds_rolled[lon_name] + 180) % 360 - 180
                return ds_rolled.sortby(lon_name)

            # Apply adjustment
            std1_avg = adjust_lon(std1_avg, lon_coord_name, lon_size)
            std2_avg = adjust_lon(std2_avg, lon_coord_name, lon_size)
            # Update coordinate variables if needed (adjust_lon modified them)
            lon_coords = std1_avg[lon_coord_name]

        else:
            print("Warning: Could not find standard longitude coordinate for rolling.")

    except FileNotFoundError as e:
        print(f"Error: Data file not found. Check path components.\n{e}")
        return  # Stop execution if files are missing
    except KeyError as e:
        print(f"Error: Variable 'pCO2_recon_full_std' not found in Zarr store: {e}")
        return
    except ValueError as e:
        print(f"Error in data dimension: {e}")
        return
    except AssertionError as e:
        print(f"Error: Data dimension assertion failed after processing. {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred during data loading or processing: {e}")
        return

    # --- Plotting ---
    print("Generating side-by-side plot...")
    try:
        # Check if SpatialMap2 class is available
        if "SpatialMap2" not in globals():
            raise NameError("SpatialMap2 class is not defined or imported.")

        with plt.style.context(plot_style):
            # Create figure for two subplots
            fig = plt.figure(
                figsize=(12, 4), dpi=200
            )  # Width allows for two plots + colorbar
            # Use the SpatialMap2 class for layout
            worldmap = SpatialMap2(
                fig=fig,
                region="world",
                cbar_mode="single",  # Use a single colorbar for both plots
                colorbar=True,
                cbar_location="bottom",  # Place colorbar at the bottom
                nrows_ncols=[1, 2],  # 1 row, 2 columns
            )

            # Plot time-averaged STD for Mask 1 on the left axis (ax=0)
            sub0 = worldmap.add_plot(
                lon=lon_coords,  # Use determined lon coords
                lat=lat_coords,  # Use determined lat coords
                data=std1_avg,  # Plot the averaged STD 1
                vrange=vrange,  # Use the specified value range
                cmap=cmap_object,  # Use the specified colormap object
                ax=0,  # Target the first subplot
            )

            # Plot time-averaged STD for Mask 2 on the right axis (ax=1)
            sub1 = worldmap.add_plot(
                lon=lon_coords,  # Use the same coordinates
                lat=lat_coords,
                data=std2_avg,  # Plot the averaged STD 2
                vrange=vrange,  # Use the same value range
                cmap=cmap_object,  # Use the same colormap
                ax=1,  # Target the second subplot
            )

            # Set titles for each subplot, indicating they show the MEAN STD
            worldmap.set_title(f"Mean STD {mask_name_1}", ax=0, fontsize=13)
            worldmap.set_title(f"Mean STD {mask_name_2}", ax=1, fontsize=13)

            # Add a single colorbar associated with one of the plots (they share the same scale)
            # If cbar_mode='single', SpatialMap2 might handle this automatically or require specifying the mappable
            # Assuming we need to explicitly add it based on one subplot's mappable:
            try:
                colorbar = worldmap.add_colorbar(
                    sub0, ax=0
                )  # Add colorbar based on first plot
                # If cbar_mode='single' places it centrally, label it once
                worldmap.set_cbar_xlabel(colorbar, "Mean STD pCO₂ (µatm)", fontsize=12)
            except Exception as cbar_err:
                # Fallback if single colorbar needs different handling by SpatialMap2
                print(
                    f"Note: Could not automatically add single colorbar, may need adjustment in SpatialMap2 or manual placement. Error: {cbar_err}"
                )
                # Try adding individually if SpatialMap2 setup needs it (less ideal for comparison)
                # colorbar0 = worldmap.add_colorbar(sub0, ax=0)
                # worldmap.set_cbar_xlabel(colorbar0, "Mean STD pCO₂ (µatm)", fontsize=12)
                # colorbar1 = worldmap.add_colorbar(sub1, ax=1) # This would add a second colorbar

            plt.tight_layout(
                rect=[0, 0.1, 1, 0.95]
            )  # Adjust layout, leave space for bottom colorbar
            plt.show()
            print("Plotting complete.")

    except NameError as e:
        print(
            f"Plotting Error: {e}. Ensure the SpatialMap2 class is defined or imported."
        )
        if "fig" in locals() and plt.fignum_exists(fig.number):
            plt.close(fig)
    except Exception as e:
        print(f"An unexpected error occurred during plotting: {e}")
        if "fig" in locals() and plt.fignum_exists(fig.number):
            plt.close(fig)


def plot_reconstruction_std_difference(
    mask_name_1,
    mask_name_2,
    selected_mems_dict,
    output_dir,
    init_date,  # Used for finding recon file path
    fin_date,  # Used for finding recon file path
    # chosen_time parameter removed
    diff_vrange=[-5, 5],  # Adjusted default vrange, difference in STD might be smaller
    cmap_diff="RdBu_r",  # Added cmap parameter, RdBu_r is suitable for differences
    plot_style="seaborn-v0_8-talk",  # Added plot style parameter
):
    """
    Visualize the difference between the time-averaged standard deviation (STD) maps
    of two reconstructed pCO₂ datasets derived from different masking strategies.

    This function computes the difference between the mean STDs of two reconstructions
    (calculated as Mask2 - Mask1) over their entire time period and displays the
    result on a world map.

    Parameters:
    ----------
    mask_name_1 : str
        Name of the first mask used for the reconstruction.
    mask_name_2 : str
        Name of the second mask used for the reconstruction.
    selected_mems_dict : dict
        Dictionary containing ensemble names as keys and lists of member names as values.
    output_dir : str
        Path to the output directory where reconstruction data is stored.
    init_date : str
        Initial date string defining the reconstruction period in the file path.
    fin_date : str
        Final date string defining the reconstruction period in the file path.
    diff_vrange : list, optional
        Value range for the color scale in the plot. Default is [-5, 5].
    cmap_diff : str, optional
        Colormap name for the difference plot. Default is "RdBu_r".
    plot_style : str, optional
        Matplotlib plot style to use. Default is "seaborn-v0_8-talk".
    """
    # Get the colormap object using the mpl_cm alias
    try:
        cmap_object = mpl_cm.get_cmap(cmap_diff)
    except ValueError as e:
        print(
            f"Error getting colormap: {e}. Check if the name '{cmap_diff}' is a valid matplotlib colormap."
        )
        return
    except AttributeError:
        print(f"Error: Could not retrieve colormap '{cmap_diff}' using mpl_cm.")
        return

    # --- Data Loading ---
    try:
        # Select the first ensemble and member (assuming STD is stored per member if generated that way)
        first_ens = list(selected_mems_dict.keys())[0]
        first_mem = selected_mems_dict[first_ens][0]
    except (IndexError, KeyError) as e:
        print(
            f"Error accessing selected members dictionary: {e}. Check the structure of selected_mems_dict."
        )
        return

    # Construct paths to the Zarr stores containing the standard deviation data
    recon_path_1 = f"{output_dir}/reconstructions/{mask_name_1}/{first_ens}/{first_mem}/recon_pCO2_{first_ens}_{first_mem}_mon_1x1_{init_date}_{fin_date}.zarr"
    recon_path_2 = f"{output_dir}/reconstructions/{mask_name_2}/{first_ens}/{first_mem}/recon_pCO2_{first_ens}_{first_mem}_mon_1x1_{init_date}_{fin_date}.zarr"

    print(f"Loading STD data from: {recon_path_1}")
    print(f"Loading STD data from: {recon_path_2}")

    try:
        # Load the *full* time series of the reconstructed pCO₂ standard deviation
        full_std_1 = xr.open_zarr(recon_path_1, consolidated=True)[
            "pCO2_recon_full_std"
        ]
        full_std_2 = xr.open_zarr(recon_path_2, consolidated=True)[
            "pCO2_recon_full_std"
        ]
        # Check for time dimension
        if "time" not in full_std_1.dims or "time" not in full_std_2.dims:
            raise ValueError("STD data must have a 'time' coordinate for averaging.")
    except FileNotFoundError as e:
        print(
            f"Error opening reconstruction file: {e}. Check paths and ensure files exist."
        )
        return
    except KeyError as e:
        print(f"Error: Variable 'pCO2_recon_full_std' not found in Zarr store: {e}")
        return
    except ValueError as e:
        print(f"Error with STD data structure: {e}")
        return
    except Exception as e:  # Catch other potential zarr/xarray errors
        print(f"An unexpected error occurred opening Zarr stores: {e}")
        return

    # --- Data Processing ---
    try:
        # Calculate the mean standard deviation over the entire time dimension
        print("Calculating time-mean standard deviation...")
        avg_std_1_temporal = full_std_1.mean(dim="time", skipna=True)
        avg_std_2_temporal = full_std_2.mean(dim="time", skipna=True)

        # Handle potential extra dimension (e.g., member) AFTER averaging time
        # This logic assumes the extra dimension, if present, is the first one
        # or can be identified as a singleton dimension not lat/lon.
        print("Checking dimensions after time averaging...")
        if len(avg_std_1_temporal.dims) > 2:
            potential_singleton_dims = [
                d
                for d in avg_std_1_temporal.dims
                if d not in ["ylat", "xlon", "lat", "lon"]
                and avg_std_1_temporal.sizes[d] == 1
            ]
            if potential_singleton_dims:
                dim_to_squeeze = potential_singleton_dims[0]
                print(f"Squeezing singleton dimension '{dim_to_squeeze}' for STD 1.")
                std_1 = avg_std_1_temporal.squeeze(dim=dim_to_squeeze, drop=True)
            else:
                # Fallback: Assume the first dimension needs selecting if no obvious singleton
                print(
                    f"Warning: STD 1 has >2 dims after averaging. Selecting index 0 from leading dimension '{avg_std_1_temporal.dims[0]}'."
                )
                std_1 = avg_std_1_temporal.isel(
                    {avg_std_1_temporal.dims[0]: 0}, drop=True
                )
        else:
            print(f"STD 1 is 2D: {avg_std_1_temporal.dims}")
            std_1 = avg_std_1_temporal  # Already 2D (lat, lon)

        if len(avg_std_2_temporal.dims) > 2:
            potential_singleton_dims = [
                d
                for d in avg_std_2_temporal.dims
                if d not in ["ylat", "xlon", "lat", "lon"]
                and avg_std_2_temporal.sizes[d] == 1
            ]
            if potential_singleton_dims:
                dim_to_squeeze = potential_singleton_dims[0]
                print(f"Squeezing singleton dimension '{dim_to_squeeze}' for STD 2.")
                std_2 = avg_std_2_temporal.squeeze(dim=dim_to_squeeze, drop=True)
            else:
                print(
                    f"Warning: STD 2 has >2 dims after averaging. Selecting index 0 from leading dimension '{avg_std_2_temporal.dims[0]}'."
                )
                std_2 = avg_std_2_temporal.isel(
                    {avg_std_2_temporal.dims[0]: 0}, drop=True
                )
        else:
            print(f"STD 2 is 2D: {avg_std_2_temporal.dims}")
            std_2 = avg_std_2_temporal  # Already 2D (lat, lon)

        # Align longitudes (roll to center on Pacific) for the averaged maps
        print("Aligning longitudes...")
        lon_coord_name = (
            "xlon" if "xlon" in std_1.dims else "lon" if "lon" in std_1.dims else None
        )
        if lon_coord_name:
            lon_size = len(std_1[lon_coord_name])
            std_1 = std_1.roll(**{lon_coord_name: lon_size // 2}, roll_coords=True)
            std_1[lon_coord_name] = (std_1[lon_coord_name] + 180) % 360 - 180
            std_1 = std_1.sortby(lon_coord_name)

            std_2 = std_2.roll(**{lon_coord_name: lon_size // 2}, roll_coords=True)
            std_2[lon_coord_name] = (std_2[lon_coord_name] + 180) % 360 - 180
            std_2 = std_2.sortby(lon_coord_name)
        else:
            print(
                "Warning: Could not find standard longitude coordinate ('xlon' or 'lon') for rolling."
            )

        # Compute the difference between the time-averaged standard deviations (Mask2 - Mask1)
        print("Calculating difference between mean STDs...")
        std_diff = std_2 - std_1

    except ValueError as e:
        print(
            f"Error during data processing (averaging, dim handling, lon align, diff): {e}"
        )
        return
    except Exception as e:
        print(f"An unexpected error occurred during data processing: {e}")
        return

    # --- Plotting ---
    print("Generating plot...")
    try:
        # Check if SpatialMap2 class is available
        if "SpatialMap2" not in globals():
            raise NameError("SpatialMap2 class is not defined or imported.")

        with plt.style.context(plot_style):
            fig = plt.figure(figsize=(8, 3), dpi=200)  # Adjusted figsize slightly
            worldmap = SpatialMap2(
                fig=fig,
                region="world",
                cbar_mode="single",
                colorbar=True,
                cbar_location="bottom",
                nrows_ncols=[1, 1],
            )

            # Determine coordinate names for plotting
            lon_coord_plot = (
                lon_coord_name
                if lon_coord_name
                else ("xlon" if "xlon" in std_diff.coords else "lon")
            )  # Best guess
            lat_coord_plot = "ylat" if "ylat" in std_diff.coords else "lat"

            # Add the difference map to the plot
            sub0 = worldmap.add_plot(
                lon=std_diff[lon_coord_plot],
                lat=std_diff[lat_coord_plot],
                data=std_diff,
                vrange=diff_vrange,
                cmap=cmap_object,  # Use the retrieved colormap object
                ax=0,
            )

            # Set the title to reflect the mean difference
            worldmap.set_title(
                f"Mean STD Difference ({mask_name_2} - {mask_name_1})",
                ax=0,
                fontsize=13,
            )

            # Add and label the colorbar
            colorbar = worldmap.add_colorbar(sub0, ax=0)
            worldmap.set_cbar_xlabel(
                colorbar, "Δ Mean STD pCO₂ (µatm)", fontsize=12
            )  # Updated label

            plt.tight_layout()  # Adjust layout
            plt.show()
            print("Plotting complete.")

    except NameError as e:
        print(
            f"Plotting Error: {e}. Ensure the SpatialMap2 class is defined or imported."
        )
        if "fig" in locals() and plt.fignum_exists(fig.number):
            plt.close(fig)
    except Exception as e:
        print(f"An unexpected error occurred during plotting: {e}")
        if "fig" in locals() and plt.fignum_exists(fig.number):
            plt.close(fig)


def plot_reconstruction_mean(
    mask_name,
    selected_mems_dict,
    output_dir,
    init_date,
    fin_date,
    chosen_time="2021-01",
    vrange=[280, 440],
):
    """
    Plots the mean of reconstructed pCO₂ data for a specific time period using NGBoost predictions.

    Parameters:
    ----------
    mask_name : str
        Name of the mask used for the reconstruction.
    selected_mems_dict : dict
        Dictionary containing ensemble names as keys and lists of member names as values.
    output_dir : str
        Path to the output directory where reconstruction data is stored.
    init_date : str
        Initial date of the reconstruction period in the format 'YYYY-MM'.
    fin_date : str
        Final date of the reconstruction period in the format 'YYYY-MM'.
    chosen_time : str, optional
        Specific time (month) to extract and plot, in the format 'YYYY-MM'. Default is "2021-01".
    vrange : list, optional
        Value range for the color scale in the plot. Default is [280, 440].

    Notes:
    ------
    This function assumes that the mean of the reconstructed pCO₂ data is derived
    from NGBoost's probabilistic predictions. NGBoost provides the mean as the
    central tendency of its probabilistic output, which can be visualized to
    assess the reconstructed values.
    """
    cmap = cm.cm.thermal

    # Select the first ensemble and member
    first_ens = list(selected_mems_dict.keys())[0]
    first_mem = selected_mems_dict[first_ens][0]

    # Load reconstructed pCO₂ data
    recon_output_dir = f"{output_dir}/reconstructions/{mask_name}"
    recon_dir = f"{recon_output_dir}/{first_ens}/{first_mem}"
    recon_path = f"{recon_dir}/recon_pCO2_{first_ens}_{first_mem}_mon_1x1_{init_date}_{fin_date}.zarr"
    print("Recon path:", recon_path)
    full = xr.open_zarr(recon_path, consolidated=True)["pCO2_recon_full_mean"]

    # Extract specific month
    recon_data = full.sel(time=chosen_time)[0, ...]

    # Shift longitudes for global plotting
    recon_data = recon_data.roll(xlon=len(recon_data.xlon) // 2, roll_coords=True)

    # Start plotting
    with plt.style.context("seaborn-v0_8-talk"):
        fig = plt.figure(figsize=(8, 3), dpi=200)
        worldmap = SpatialMap2(
            fig=fig,
            region="world",
            cbar_mode="single",
            colorbar=True,
            cbar_location="bottom",
            nrows_ncols=[1, 1],
        )

        sub1 = worldmap.add_plot(
            lon=recon_data["xlon"],
            lat=recon_data["ylat"],
            data=recon_data,
            vrange=vrange,
            cmap=cmap,
            ax=0,
        )

        worldmap.set_title(
            f"pCO₂ Reconstruction Mean ({chosen_time})", ax=0, fontsize=13
        )

        colorbar = worldmap.add_colorbar(sub1, ax=0)
        worldmap.set_cbar_xlabel(colorbar, "pCO₂ (µatm)", fontsize=12)

        plt.show()


### Difference plots


def plot_masking_strategy_difference(
    mask_name_1,
    mask_name_2,
    selected_mems_dict,
    output_dir,
    init_date,
    fin_date,
    diff_vrange=[-30, 30],
    chosen_time="2021-01",
):
    """
    Plot the difference between reconstructions from two different masking strategies.
    Parameters:
    ----------
    mask_name_1 : str
        Name of the first mask used for the reconstruction.
    mask_name_2 : str
        Name of the second mask used for the reconstruction.
    selected_mems_dict : dict
        Dictionary containing ensemble names as keys and lists of member names as values.
    output_dir : str
        Path to the output directory where reconstruction data is stored.
    init_date : str
        Initial date of the reconstruction period in the format 'YYYY-MM'.
    fin_date : str
        Final date of the reconstruction period in the format 'YYYY-MM'.
    diff_vrange : list, optional
        Value range for the color scale in the plot. Default is [-30, 30].
    chosen_time : str, optional
        Specific time (month) to extract and plot, in the format 'YYYY-MM'. Default is "2021-01".
    """

    # Select the first ensemble and member
    first_ens = list(selected_mems_dict.keys())[0]
    first_mem = selected_mems_dict[first_ens][0]

    # Load reconstructed pCO₂ for both masks
    recon_path_1 = f"{output_dir}/reconstructions/{mask_name_1}/{first_ens}/{first_mem}/recon_pCO2_{first_ens}_{first_mem}_mon_1x1_{init_date}_{fin_date}.zarr"
    recon_path_2 = f"{output_dir}/reconstructions/{mask_name_2}/{first_ens}/{first_mem}/recon_pCO2_{first_ens}_{first_mem}_mon_1x1_{init_date}_{fin_date}.zarr"

    recon_1 = xr.open_zarr(recon_path_1, consolidated=True)["pCO2_recon_full"].sel(
        time=chosen_time
    )[0, ...]
    recon_2 = xr.open_zarr(recon_path_2, consolidated=True)["pCO2_recon_full"].sel(
        time=chosen_time
    )[0, ...]

    # Align longitudes
    recon_1 = recon_1.roll(xlon=len(recon_1.xlon) // 2, roll_coords=True)
    recon_2 = recon_2.roll(xlon=len(recon_2.xlon) // 2, roll_coords=True)

    # Calculate the difference
    diff = recon_2 - recon_1

    # Convert colormap string to colormap object
    cmap_diff = mpl_cm.get_cmap("RdBu_r")

    # Start plotting
    with plt.style.context("seaborn-v0_8-talk"):
        fig = plt.figure(figsize=(6, 3), dpi=200)
        worldmap = SpatialMap2(
            fig=fig,
            region="world",
            cbar_mode="single",
            colorbar=True,
            cbar_location="bottom",
            nrows_ncols=[1, 1],
        )

        sub0 = worldmap.add_plot(
            lon=recon_1["xlon"],
            lat=recon_1["ylat"],
            data=diff,
            vrange=diff_vrange,
            cmap=cmap_diff,  # Pass the colormap object
            ax=0,
        )

        worldmap.set_title(
            f"Reconstruction Difference\n({mask_name_2} - {mask_name_1}) ({chosen_time})",
            ax=0,
            fontsize=13,
        )
        colorbar = worldmap.add_colorbar(sub0, ax=0)
        worldmap.set_cbar_xlabel(colorbar, "Δ pCO₂ (µatm)", fontsize=12)

        plt.show()


def plot_masking_strategy_total_difference(
    mask_name_1,
    mask_name_2,
    mask_data_dict,
    mask_diff_vrange=[-20, 20],
):
    """
    Plot the total difference in number of months sampled between two SOCAT sampling strategies, summed over time.
    Parameters:
    ----------
    mask_name_1 : str
        Name of the first SOCAT mask.
    mask_name_2 : str
        Name of the second SOCAT mask.
    mask_data_dict : dict
        Dictionary containing SOCAT mask datasets.
    mask_diff_vrange : list, optional
        Value range for the color scale in the plot. Default is [-20, 20].
    """

    # Sum masks over time dimension
    mask1 = mask_data_dict[mask_name_1]["socat_mask"].sum(dim="time")
    mask2 = mask_data_dict[mask_name_2]["socat_mask"].sum(dim="time")

    # Align longitudes
    mask1 = mask1.roll(xlon=len(mask1.xlon) // 2, roll_coords=True)
    mask2 = mask2.roll(xlon=len(mask2.xlon) // 2, roll_coords=True)

    # Calculate difference
    mask_diff = mask2 - mask1
    cmap_mask_diff = mpl_cm.get_cmap("coolwarm")

    # Start plotting
    with plt.style.context("seaborn-v0_8-talk"):
        fig = plt.figure(figsize=(6, 3), dpi=200)
        worldmap = SpatialMap2(
            fig=fig,
            region="world",
            cbar_mode="single",
            colorbar=True,
            cbar_location="bottom",
            nrows_ncols=[1, 1],
        )

        sub0 = worldmap.add_plot(
            lon=mask1["xlon"],
            lat=mask1["ylat"],
            data=mask_diff,
            vrange=mask_diff_vrange,
            cmap=cmap_mask_diff,
            ax=0,
        )

        worldmap.set_title(
            f"Total Sampling Difference\n({mask_name_2} - {mask_name_1})",
            ax=0,
            fontsize=13,
        )
        colorbar = worldmap.add_colorbar(sub0, ax=0)
        worldmap.set_cbar_xlabel(colorbar, "Δ Total Months Sampled", fontsize=12)

        plt.show()


def plot_monthly_comparison_panel(
    mask_name_1,
    mask_name_2,
    mask_data_dict,
    selected_mems_dict,
    output_dir,
    init_date,
    fin_date,
    chosen_time="2021-01",
    mask_vrange=[0, 1],
    diff_vrange=[-30, 30],
):
    """
    Plot the sampling masks and the difference in reconstruction between two masking strategies.
    Parameters:
    ----------
    mask_name_1 : str
        Name of the first SOCAT mask.
    mask_name_2 : str
        Name of the second SOCAT mask.
    mask_data_dict : dict
        Dictionary containing SOCAT mask datasets.
    selected_mems_dict : dict
        Dictionary containing ensemble names as keys and lists of member names as values.
    output_dir : str
        Path to the output directory where reconstruction data is stored.
    init_date : str
        Initial date of the reconstruction period in the format 'YYYY-MM'.
    fin_date : str
        Final date of the reconstruction period in the format 'YYYY-MM'.
    chosen_time : str, optional
        Specific time (month) to extract and plot, in the format 'YYYY-MM'. Default is "2021-01".
    mask_vrange : list, optional
        Value range for the color scale of the masks. Default is [0, 1].
    diff_vrange : list, optional
        Value range for the color scale of the difference in reconstruction. Default is [-30, 30].
    Notes:
    ------
    This function visualizes the sampling masks for two different SOCAT masking strategies
    and the difference in reconstruction between them. The masks are displayed on the left
    and center, while the difference in reconstruction is shown on the right.
    """
    cmap_mask = mpl_cm.get_cmap("Blues")
    cmap_diff = mpl_cm.get_cmap("RdBu_r")

    # Select first ensemble member
    first_ens = list(selected_mems_dict.keys())[0]
    first_mem = selected_mems_dict[first_ens][0]

    # Load masks
    mask1 = mask_data_dict[mask_name_1].sel(time=chosen_time)["socat_mask"].squeeze()
    mask2 = mask_data_dict[mask_name_2].sel(time=chosen_time)["socat_mask"].squeeze()

    # Load reconstructions
    recon_path_1 = f"{output_dir}/reconstructions/{mask_name_1}/{first_ens}/{first_mem}/recon_pCO2_{first_ens}_{first_mem}_mon_1x1_{init_date}_{fin_date}.zarr"
    recon_path_2 = f"{output_dir}/reconstructions/{mask_name_2}/{first_ens}/{first_mem}/recon_pCO2_{first_ens}_{first_mem}_mon_1x1_{init_date}_{fin_date}.zarr"

    recon1 = xr.open_zarr(recon_path_1, consolidated=True)["pCO2_recon_full"].sel(
        time=chosen_time
    )[0, ...]
    recon2 = xr.open_zarr(recon_path_2, consolidated=True)["pCO2_recon_full"].sel(
        time=chosen_time
    )[0, ...]

    # Align longitude (roll at 180°)
    mask1 = mask1.roll(xlon=len(mask1.xlon) // 2, roll_coords=True)
    mask2 = mask2.roll(xlon=len(mask2.xlon) // 2, roll_coords=True)
    recon1 = recon1.roll(xlon=len(recon1.xlon) // 2, roll_coords=True)
    recon2 = recon2.roll(xlon=len(recon2.xlon) // 2, roll_coords=True)

    # Difference in reconstructions
    diff = recon2 - recon1

    # Start plotting
    with plt.style.context("seaborn-v0_8-talk"):
        fig = plt.figure(figsize=(15, 4), dpi=200)
        worldmap = SpatialMap2(
            fig=fig,
            region="world",
            cbar_mode="each",
            colorbar=True,
            cbar_location="bottom",
            nrows_ncols=[1, 3],
        )

        # Left: sampling mask 1
        sub0 = worldmap.add_plot(
            lon=mask1["xlon"],
            lat=mask1["ylat"],
            data=mask1,
            vrange=mask_vrange,
            cmap=cmap_mask,
            ax=0,
        )
        worldmap.set_title(f"Mask: {mask_name_1}", ax=0, fontsize=13)

        # Center: sampling mask 2
        sub1 = worldmap.add_plot(
            lon=mask2["xlon"],
            lat=mask2["ylat"],
            data=mask2,
            vrange=mask_vrange,
            cmap=cmap_mask,
            ax=1,
        )
        worldmap.set_title(f"Mask: {mask_name_2}", ax=1, fontsize=13)

        # Right: difference in reconstruction
        sub2 = worldmap.add_plot(
            lon=diff["xlon"],
            lat=diff["ylat"],
            data=diff,
            vrange=diff_vrange,
            cmap=cmap_diff,
            ax=2,
        )
        worldmap.set_title(
            f"Reconstruction Difference\n({mask_name_2} - {mask_name_1})",
            ax=2,
            fontsize=13,
        )

        # Colorbars
        cbar0 = worldmap.add_colorbar(sub0, ax=0)
        cbar1 = worldmap.add_colorbar(sub1, ax=1)
        cbar2 = worldmap.add_colorbar(sub2, ax=2)

        worldmap.set_cbar_xlabel(cbar0, "Sampling Presence", fontsize=11)
        worldmap.set_cbar_xlabel(cbar1, "Sampling Presence", fontsize=11)
        worldmap.set_cbar_xlabel(cbar2, "Δ pCO₂ (µatm)", fontsize=11)

        plt.tight_layout()
        plt.show()


def plot_masking_strategy_month_difference(
    mask_name_1,
    mask_name_2,
    mask_data_dict,
    chosen_time="2021-02",
    mask_diff_vrange=[-2, 2],
):
    """
    Plot the SOCAT mask difference between two strategies for a single month.
    Parameters:
    ----------
    mask_name_1 : str
        Name of the first SOCAT mask.
    mask_name_2 : str
        Name of the second SOCAT mask.
    mask_data_dict : dict
        Dictionary containing SOCAT mask datasets.
    chosen_time : str, optional
        Time period for the SOCAT masks (default is "2021-02").
    mask_diff_vrange : list, optional
        Value range for the color scale in the plot (default is [-2, 2]).
    Notes:
    ------
    This function visualizes the difference in SOCAT sampling masks between two
    different strategies for a specific month. The difference is calculated as
    Mask2 - Mask1, and the result is displayed on a world map with a specified
    color range.
    """
    cmap_mask_diff = mpl_cm.get_cmap("coolwarm")

    # Load single-month SOCAT mask for both strategies
    mask1 = mask_data_dict[mask_name_1].sel(time=chosen_time)["socat_mask"].squeeze()
    mask2 = mask_data_dict[mask_name_2].sel(time=chosen_time)["socat_mask"].squeeze()

    # Align longitudes for visualization
    mask1 = mask1.roll(xlon=len(mask1.xlon) // 2, roll_coords=True)
    mask2 = mask2.roll(xlon=len(mask2.xlon) // 2, roll_coords=True)

    # Calculate difference (Mask2 - Mask1)
    mask_diff = mask2 - mask1

    # Start plotting
    with plt.style.context("seaborn-v0_8-talk"):
        fig = plt.figure(figsize=(6, 3), dpi=200)
        worldmap = SpatialMap2(
            fig=fig,
            region="world",
            cbar_mode="single",
            colorbar=True,
            cbar_location="bottom",
            nrows_ncols=[1, 1],
        )

        sub0 = worldmap.add_plot(
            lon=mask1["xlon"],
            lat=mask1["ylat"],
            data=mask_diff,
            vrange=mask_diff_vrange,
            cmap=cmap_mask_diff,
            ax=0,
        )

        worldmap.set_title(
            f"SOCAT Sampling Difference\n({mask_name_2} - {mask_name_1}) ({chosen_time})",
            ax=0,
            fontsize=13,
        )
        colorbar = worldmap.add_colorbar(sub0, ax=0)
        worldmap.set_cbar_xlabel(colorbar, "Δ Observations", fontsize=12)

        plt.show()


def plot_masking_meanstrategy_difference(
    mask_name_1,
    mask_name_2,
    mask_data_dict,  # This parameter is not actually used in the function body, but kept for signature consistency
    selected_mems_dict,
    ensemble_dir,
    output_dir,
    dates,  # Not directly used for averaging, but kept for consistency. Assumed recon files cover the period.
    init_date,  # Used for finding recon file path
    fin_date,  # Used for finding recon file path
    # --- MODIFIED PARAMETERS ---
    start_time,  # Start time for averaging (e.g., "2004-01")
    end_time,  # End time for averaging (e.g., "2023-12")
    # --- END MODIFIED PARAMETERS ---
    plot_style="seaborn-v0_8-talk",
    cmap_diff="RdBu_r",  # Colormap as a string
    diff_vrange=[-30, 30],
    # chosen_time is removed, replaced by start_time and end_time
):
    """
    Plot the difference between the time-averaged reconstructions from
    two different masking strategies over a specified period.
    (Modified from original to show difference of time averages)
    """

    # --- Data Loading and Preparation ---
    try:
        # Select the first ensemble and member (original logic kept)
        first_ens = list(selected_mems_dict.keys())[0]
        first_mem = selected_mems_dict[first_ens][0]
    except (IndexError, KeyError) as e:
        print(
            f"Error accessing selected members dictionary: {e}. Check the structure of selected_mems_dict."
        )
        return

    # Construct paths (original logic kept)
    recon_path_1 = f"{output_dir}/reconstructions/{mask_name_1}/{first_ens}/{first_mem}/recon_pCO2_{first_ens}_{first_mem}_mon_1x1_{init_date}_{fin_date}.zarr"
    recon_path_2 = f"{output_dir}/reconstructions/{mask_name_2}/{first_ens}/{first_mem}/recon_pCO2_{first_ens}_{first_mem}_mon_1x1_{init_date}_{fin_date}.zarr"

    print(f"Loading recon 1: {recon_path_1}")
    print(f"Loading recon 2: {recon_path_2}")

    # Load full reconstructed data for the relevant variable
    try:
        recon_full_1 = xr.open_zarr(recon_path_1, consolidated=True)["pCO2_recon_full"]
        recon_full_2 = xr.open_zarr(recon_path_2, consolidated=True)["pCO2_recon_full"]
    except FileNotFoundError as e:
        print(
            f"Error opening reconstruction file: {e}. Check paths and ensure files exist."
        )
        print(f"Path 1: {recon_path_1}")
        print(f"Path 2: {recon_path_2}")
        return
    except KeyError as e:
        print(f"Error: Variable 'pCO2_recon_full' not found in Zarr store: {e}")
        return
    except Exception as e:  # Catch other potential zarr/xarray errors
        print(f"An unexpected error occurred opening Zarr stores: {e}")
        return

    # --- Select Time Slice and Average ---
    print(
        f"Calculating time average from {start_time} to {end_time} for both reconstructions..."
    )
    try:
        recon_slice_1 = recon_full_1.sel(time=slice(start_time, end_time))
        recon_slice_2 = recon_full_2.sel(time=slice(start_time, end_time))

        # Check if slices are empty
        if recon_slice_1.time.size == 0:
            print(
                f"Warning: No time data found for recon 1 between {start_time} and {end_time}."
            )
            return
        if recon_slice_2.time.size == 0:
            print(
                f"Warning: No time data found for recon 2 between {start_time} and {end_time}."
            )
            return

        recon_avg_1_temporal = recon_slice_1.mean(dim="time", skipna=True)
        recon_avg_2_temporal = recon_slice_2.mean(dim="time", skipna=True)
    except KeyError as e:
        print(
            f"Error selecting time slice: {e}. Check if 'time' coordinate exists and format of start/end times ({start_time}, {end_time}) matches data."
        )
        return
    except Exception as e:
        print(f"An error occurred during time slicing or averaging: {e}")
        return

    # Handle the potential extra dimension (like the original [0, ...]) AFTER averaging
    try:
        if len(recon_avg_1_temporal.dims) > 2:
            # Try to find a likely singleton dimension (e.g., 'member') if not the first one
            potential_singleton_dims = [
                d
                for d in recon_avg_1_temporal.dims
                if d not in ["ylat", "xlon", "lat", "lon"]
                and recon_avg_1_temporal.sizes[d] == 1
            ]
            if potential_singleton_dims:
                dim_to_squeeze = potential_singleton_dims[0]
                print(
                    f"Selecting index 0 from singleton dimension '{dim_to_squeeze}' for recon 1 after averaging."
                )
                recon_1 = recon_avg_1_temporal.isel({dim_to_squeeze: 0}, drop=True)
            else:
                # Fallback to original logic if no obvious singleton found besides the first
                print(
                    f"Warning: Recon 1 has >2 dims after averaging. Selecting index 0 from leading dimension '{recon_avg_1_temporal.dims[0]}'."
                )
                recon_1 = recon_avg_1_temporal.isel(
                    {recon_avg_1_temporal.dims[0]: 0}, drop=True
                )
        else:
            recon_1 = recon_avg_1_temporal  # Already 2D

        if len(recon_avg_2_temporal.dims) > 2:
            potential_singleton_dims = [
                d
                for d in recon_avg_2_temporal.dims
                if d not in ["ylat", "xlon", "lat", "lon"]
                and recon_avg_2_temporal.sizes[d] == 1
            ]
            if potential_singleton_dims:
                dim_to_squeeze = potential_singleton_dims[0]
                print(
                    f"Selecting index 0 from singleton dimension '{dim_to_squeeze}' for recon 2 after averaging."
                )
                recon_2 = recon_avg_2_temporal.isel({dim_to_squeeze: 0}, drop=True)
            else:
                print(
                    f"Warning: Recon 2 has >2 dims after averaging. Selecting index 0 from leading dimension '{recon_avg_2_temporal.dims[0]}'."
                )
                recon_2 = recon_avg_2_temporal.isel(
                    {recon_avg_2_temporal.dims[0]: 0}, drop=True
                )
        else:
            recon_2 = recon_avg_2_temporal  # Already 2D
    except Exception as e:
        print(f"Error handling potential extra dimensions after averaging: {e}")
        return
    # --- End Time Slice and Average ---

    # Align longitudes (applied to averaged data)
    try:
        # Ensure coordinate names are correct before rolling
        lon_coord_name = (
            "xlon"
            if "xlon" in recon_1.dims
            else "lon"
            if "lon" in recon_1.dims
            else None
        )
        if lon_coord_name:
            recon_1 = recon_1.roll(
                **{lon_coord_name: len(recon_1[lon_coord_name]) // 2}, roll_coords=True
            )
            recon_2 = recon_2.roll(
                **{lon_coord_name: len(recon_2[lon_coord_name]) // 2}, roll_coords=True
            )
        else:
            print(
                "Warning: Could not find standard longitude coordinate ('xlon' or 'lon') for rolling."
            )
    except Exception as e:
        print(f"Error rolling longitude coordinates: {e}")
        # Decide if you want to return or proceed with unrolled data
        # return

    # Calculate the difference between the averages
    try:
        diff = recon_2 - recon_1
    except ValueError as e:
        print(
            f"Error calculating difference between averaged reconstructions: {e}. Check if dimensions and coordinates align after processing."
        )
        return

    # Convert colormap string to colormap object (using new mpl_cm alias)
    try:
        # matplotlib >= 3.7 prefers this
        cmap_object = plt.colormaps[cmap_diff]
    except AttributeError:
        # Fallback for older matplotlib using mpl_cm
        try:
            cmap_object = mpl_cm.get_cmap(cmap_diff)
        except ValueError as e:
            print(
                f"Error getting colormap: {e}. Check if the name '{cmap_diff}' is a valid matplotlib colormap."
            )
            return
        except AttributeError:
            print(f"Error: Could not retrieve colormap '{cmap_diff}' using mpl_cm.")
            return

    # --- Plotting ---
    try:
        # Check if SpatialMap2 class is available
        if "SpatialMap2" not in globals():
            # Try to import it if you know the module name
            # from aplotpy import SpatialMap2 # Example
            # If it cannot be imported or defined, raise an error
            raise NameError("SpatialMap2 class is not defined or imported.")

        with plt.style.context(plot_style):
            fig = plt.figure(figsize=(6, 3), dpi=200)
            worldmap = SpatialMap2(
                fig=fig,
                region="world",
                cbar_mode="single",
                colorbar=True,
                cbar_location="bottom",
                nrows_ncols=[1, 1],
            )

            # Determine coordinate names for plotting
            lon_coord_plot = "xlon" if "xlon" in recon_1.coords else "lon"
            lat_coord_plot = "ylat" if "ylat" in recon_1.coords else "lat"

            sub0 = worldmap.add_plot(
                lon=recon_1[lon_coord_plot],  # Coordinates from either averaged recon
                lat=recon_1[lat_coord_plot],
                data=diff,  # Plot the difference of the averages
                vrange=diff_vrange,
                cmap=cmap_object,  # Pass the colormap object
                ax=0,
            )

            # Update title to reflect averaging period
            time_title = f"{start_time} to {end_time} Avg."
            worldmap.set_title(
                f"Reconstruction Difference ({time_title})\n({mask_name_2} - {mask_name_1})",
                ax=0,
                fontsize=13,
            )
            colorbar = worldmap.add_colorbar(sub0, ax=0)
            worldmap.set_cbar_xlabel(colorbar, "Δ pCO₂ (µatm)", fontsize=12)

            plt.tight_layout()  # Added tight layout
            plt.show()

    except NameError as e:
        print(
            f"Plotting Error: {e}. Ensure the SpatialMap2 class is defined or imported."
        )
    except Exception as e:
        print(f"An unexpected error occurred during plotting: {e}")
        # Optionally close the figure if it was created but plotting failed
        if "fig" in locals() and plt.fignum_exists(fig.number):
            plt.close(fig)


def plot_mean_comparison_panel(
    mask_name_1,
    mask_name_2,
    mask_data_dict,
    selected_mems_dict,
    ensemble_dir,
    output_dir,
    dates,
    init_date,
    fin_date,
    mask_vrange=[0, 1],
    diff_vrange=[-30, 30],
    cmap_mask="Blues",
    cmap_diff="RdBu_r",
    plot_style="seaborn-v0_8-talk",
):
    """
    Plots a 3-panel comparison: Mask 1, Mask 2, and the difference
    in their corresponding mean pCO2 reconstructions.

    Args:
        mask_name_1 (str): Name of the first mask scenario.
        mask_name_2 (str): Name of the second mask scenario.
        mask_data_dict (dict): Dictionary containing mask dataarrays.
                                Expected structure: {'mask_name': {'socat_mask': xr.DataArray}}
        selected_mems_dict (dict): Dictionary defining selected members for ensembles.
                                   Expected structure: {'ensemble_name': [member_list]}
        ensemble_dir (str): Path to the main ensemble directory (unused in current logic but kept for signature).
        output_dir (str): Path to the main output directory containing reconstructions.
        dates (array-like): Array of dates (unused in current logic but kept for signature).
        init_date (str): Start date string (YYYY-MM-DD) for file paths.
        fin_date (str): End date string (YYYY-MM-DD) for file paths.
        mask_vrange (list, optional): Value range for mask plots. Defaults to [0, 1].
        diff_vrange (list, optional): Value range for difference plot. Defaults to [-30, 30].
        cmap_mask (str, optional): Colormap name for mask plots. Defaults to "Blues".
        cmap_diff (str, optional): Colormap name for difference plot. Defaults to "RdBu_r".
        plot_style (str, optional): Matplotlib plot style. Defaults to "seaborn-v0_8-talk".
    """

    # Get colormaps using the new alias mpl_cm
    try:
        cmap_mask_obj = mpl_cm.get_cmap(cmap_mask)
        cmap_diff_obj = mpl_cm.get_cmap(cmap_diff)
    except ValueError as e:
        print(
            f"Error getting colormap: {e}. Check if the names '{cmap_mask}' and '{cmap_diff}' are valid matplotlib colormaps."
        )
        return
    except AttributeError:
        # Handle cases where get_cmap might return None or mpl_cm is not as expected (less likely)
        print(f"Error: Could not retrieve colormaps '{cmap_mask}' or '{cmap_diff}'.")
        return

    # --- Data Loading and Preparation ---
    try:
        first_ens = list(selected_mems_dict.keys())[0]
        first_mem = selected_mems_dict[first_ens][0]
    except (IndexError, KeyError) as e:
        print(
            f"Error accessing selected members dictionary: {e}. Check the structure of selected_mems_dict."
        )
        return

    try:
        mask1 = mask_data_dict[mask_name_1]["socat_mask"].mean(dim="time")
        mask2 = mask_data_dict[mask_name_2]["socat_mask"].mean(dim="time")
    except (KeyError, AttributeError, ValueError) as e:
        print(
            f"Error processing mask data from mask_data_dict: {e}. Check keys and data structure."
        )
        return

    recon_path_1 = f"{output_dir}/reconstructions/{mask_name_1}/{first_ens}/{first_mem}/recon_pCO2_{first_ens}_{first_mem}_mon_1x1_{init_date}_{fin_date}.zarr"
    recon_path_2 = f"{output_dir}/reconstructions/{mask_name_2}/{first_ens}/{first_mem}/recon_pCO2_{first_ens}_{first_mem}_mon_1x1_{init_date}_{fin_date}.zarr"

    # Load reconstructions and calculate the mean over time
    try:
        recon1_full = xr.open_zarr(recon_path_1, consolidated=True)["pCO2_recon_full"]
        recon2_full = xr.open_zarr(recon_path_2, consolidated=True)["pCO2_recon_full"]
    except FileNotFoundError as e:
        print(
            f"Error opening reconstruction file: {e}. Check paths and ensure files exist."
        )
        print(f"Path 1: {recon_path_1}")
        print(f"Path 2: {recon_path_2}")
        return
    except KeyError as e:
        print(f"Error: Variable 'pCO2_recon_full' not found in Zarr store: {e}")
        return
    except Exception as e:  # Catch other potential zarr/xarray errors
        print(f"An unexpected error occurred opening Zarr stores: {e}")
        return

    # Define member dimension name (adjust if necessary)
    member_dim_name = "member"

    # Process recon1
    try:
        if member_dim_name in recon1_full.dims:
            if recon1_full.dims[member_dim_name] > 1:
                print(f"Selecting first member for {mask_name_1} after time mean.")
                recon1 = recon1_full.mean(dim="time").isel({member_dim_name: 0})
            elif recon1_full.dims[member_dim_name] == 1:
                print(f"Squeezing single member dim for {mask_name_1} after time mean.")
                recon1 = recon1_full.mean(dim="time").squeeze(
                    dim=member_dim_name, drop=True
                )
            else:  # Should not happen if dim exists, but for completeness
                print(
                    f"Warning: Member dimension '{member_dim_name}' found with size 0 for {mask_name_1}. Taking time mean."
                )
                recon1 = recon1_full.mean(dim="time")
        else:
            print(
                f"No member dimension '{member_dim_name}' found for {mask_name_1}, taking time mean."
            )
            recon1 = recon1_full.mean(dim="time")
    except Exception as e:
        print(f"Error processing recon1 for {mask_name_1}: {e}")
        return

    # Process recon2
    try:
        if member_dim_name in recon2_full.dims:
            if recon2_full.dims[member_dim_name] > 1:
                print(f"Selecting first member for {mask_name_2} after time mean.")
                recon2 = recon2_full.mean(dim="time").isel({member_dim_name: 0})
            elif recon2_full.dims[member_dim_name] == 1:
                print(f"Squeezing single member dim for {mask_name_2} after time mean.")
                recon2 = recon2_full.mean(dim="time").squeeze(
                    dim=member_dim_name, drop=True
                )
            else:
                print(
                    f"Warning: Member dimension '{member_dim_name}' found with size 0 for {mask_name_2}. Taking time mean."
                )
                recon2 = recon2_full.mean(dim="time")
        else:
            print(
                f"No member dimension '{member_dim_name}' found for {mask_name_2}, taking time mean."
            )
            recon2 = recon2_full.mean(dim="time")
    except Exception as e:
        print(f"Error processing recon2 for {mask_name_2}: {e}")
        return

    # Align longitude (roll at 180°) - added checks for safety
    try:
        if "xlon" in mask1.dims:
            mask1 = mask1.roll(xlon=len(mask1.xlon) // 2, roll_coords=True)
        if "xlon" in mask2.dims:
            mask2 = mask2.roll(xlon=len(mask2.xlon) // 2, roll_coords=True)
        if "xlon" in recon1.dims:
            recon1 = recon1.roll(xlon=len(recon1.xlon) // 2, roll_coords=True)
        if "xlon" in recon2.dims:
            recon2 = recon2.roll(xlon=len(recon2.xlon) // 2, roll_coords=True)
    except Exception as e:
        print(f"Error rolling longitude coordinates: {e}")
        # Decide if you want to return or proceed with unrolled data
        # return

    # Difference in mean reconstructions
    try:
        diff = recon2 - recon1
    except ValueError as e:
        print(
            f"Error calculating difference between reconstructions: {e}. Check if dimensions and coordinates align."
        )
        return

    # --- Plotting ---
    try:
        # Check if SpatialMap2 class is available
        if "SpatialMap2" not in globals():
            # Try to import it if you know the module name
            # from your_spatialmap_module import SpatialMap2
            # If it cannot be imported or defined, raise an error
            raise NameError("SpatialMap2 class is not defined or imported.")

        with plt.style.context(plot_style):
            fig = plt.figure(figsize=(15, 4), dpi=200)
            worldmap = SpatialMap2(
                fig=fig,
                region="world",
                cbar_mode="each",
                colorbar=True,
                cbar_location="bottom",
                nrows_ncols=[1, 3],
            )

            # Left: mean sampling mask 1
            sub0 = worldmap.add_plot(
                lon=mask1["xlon"],
                lat=mask1["ylat"],
                data=mask1,
                vrange=mask_vrange,
                cmap=cmap_mask_obj,
                ax=0,  # Use cmap object
            )
            worldmap.set_title(f"Mask: {mask_name_1}", ax=0, fontsize=13)

            # Center: mean sampling mask 2
            sub1 = worldmap.add_plot(
                lon=mask2["xlon"],
                lat=mask2["ylat"],
                data=mask2,
                vrange=mask_vrange,
                cmap=cmap_mask_obj,
                ax=1,  # Use cmap object
            )
            worldmap.set_title(f"Mask: {mask_name_2}", ax=1, fontsize=13)

            # Right: difference in mean reconstruction
            sub2 = worldmap.add_plot(
                lon=diff["xlon"],
                lat=diff["ylat"],
                data=diff,
                vrange=diff_vrange,
                cmap=cmap_diff_obj,
                ax=2,  # Use cmap object
            )
            worldmap.set_title(
                f"Mean Reconstruction Difference\n({mask_name_2} - {mask_name_1})",
                ax=2,
                fontsize=13,
            )

            # Colorbars
            # cbar0 = worldmap.add_colorbar(sub0, ax=0)
            # cbar1 = worldmap.add_colorbar(sub1, ax=1)
            # cbar2 = worldmap.add_colorbar(sub2, ax=2)

            cbar0 = worldmap.add_colorbar(
                sub0, ax=0, cmap=cmap_mask_obj, vrange=mask_vrange
            )
            cbar1 = worldmap.add_colorbar(
                sub1, ax=1, cmap=cmap_mask_obj, vrange=mask_vrange
            )
            cbar2 = worldmap.add_colorbar(
                sub2, ax=2, cmap=cmap_diff_obj, vrange=diff_vrange
            )

            worldmap.set_cbar_xlabel(cbar0, "Sampling Presence", fontsize=11)
            worldmap.set_cbar_xlabel(cbar1, "Sampling Presence", fontsize=11)
            worldmap.set_cbar_xlabel(cbar2, "Δ pCO₂ (µatm)", fontsize=11)

            plt.tight_layout()
            plt.show()

    except NameError as e:
        print(
            f"Plotting Error: {e}. Ensure the SpatialMap2 class is defined or imported."
        )
    except Exception as e:
        print(f"An unexpected error occurred during plotting: {e}")
        # Optionally close the figure if it was created but plotting failed
        if "fig" in locals() and plt.fignum_exists(fig.number):
            plt.close(fig)


###


def compare_masks_pvalue_plot(
    mask_name_1, mask_name_2, selected_mems_dict, output_dir, init_date, fin_date
):
    """
    Compare two reconstruction masks by computing the p-value map between their time-averaged pCO₂ fields.

    Parameters:
    ----------
    mask_name_1 : str
        Name of the first mask (baseline).
    mask_name_2 : str
        Name of the second mask (e.g., densified sampling).
    selected_mems_dict : dict
        Dictionary with ensemble member structure.
    output_dir : str
        Base output directory containing reconstructions.
    init_date : str
        Start date of reconstruction (format: YYYYMM).
    fin_date : str
        End date of reconstruction (format: YYYYMM).
    calculate_p_value_map_fn : function
        Function that takes (mean1, std1, mean2, std2) arrays and returns a p-value map.

    Returns:
    -------
    None
    """
    # Initialize variables
    mean1_avg, std1_avg, mean2_avg, std2_avg = None, None, None, None
    lon_coords, lat_coords = None, None

    try:
        # --- Get data path components ---
        first_ens = list(selected_mems_dict.keys())[0]
        first_mem = selected_mems_dict[first_ens][0]
        base_path_str = f"{output_dir}/reconstructions/{{mask_name}}/{first_ens}/{first_mem}/recon_pCO2_{first_ens}_{first_mem}_mon_1x1_{init_date}_{fin_date}.zarr"

        # --- Process Mask 1 ---
        path1 = base_path_str.format(mask_name=mask_name_1)
        print(f"Processing {mask_name_1} from: {path1}")
        ds1 = xr.open_zarr(path1, consolidated=True)
        mean1_avg = (
            ds1["pCO2_recon_full_mean"].mean(dim="time", skipna=True).squeeze(drop=True)
        )
        std1_avg = (
            ds1["pCO2_recon_full_std"].mean(dim="time", skipna=True).squeeze(drop=True)
        )
        assert len(mean1_avg.dims) == 2, f"{mask_name_1} mean avg is not 2D"
        assert len(std1_avg.dims) == 2, f"{mask_name_1} std avg is not 2D"
        lon_coords = (
            mean1_avg["xlon"] if "xlon" in mean1_avg.coords else mean1_avg["lon"]
        )
        lat_coords = (
            mean1_avg["ylat"] if "ylat" in mean1_avg.coords else mean1_avg["lat"]
        )

        # --- Process Mask 2 ---
        path2 = base_path_str.format(mask_name=mask_name_2)
        print(f"Processing {mask_name_2} from: {path2}")
        ds2 = xr.open_zarr(path2, consolidated=True)
        mean2_avg = (
            ds2["pCO2_recon_full_mean"].mean(dim="time", skipna=True).squeeze(drop=True)
        )
        std2_avg = (
            ds2["pCO2_recon_full_std"].mean(dim="time", skipna=True).squeeze(drop=True)
        )
        assert len(mean2_avg.dims) == 2, f"{mask_name_2} mean avg is not 2D"
        assert len(std2_avg.dims) == 2, f"{mask_name_2} std avg is not 2D"

        # --- Calculate p-values ---
        print("Calculating p-value map...")
        p_value_map_avg = calculate_p_value_map(
            mean1_avg.values, std1_avg.values, mean2_avg.values, std2_avg.values
        )
        valid_count = np.isfinite(p_value_map_avg).sum()
        print(f"Number of valid (non-NaN) p-values calculated: {valid_count}")

        # --- Plotting ---
        print("Generating plot...")
        plt.figure(figsize=(10, 5))
        plt.pcolormesh(
            lon_coords,
            lat_coords,
            p_value_map_avg,
            shading="auto",
            cmap="viridis_r",
            vmin=0,
            vmax=0.1,
        )
        plt.colorbar(label="p-value (Two-sided)")
        plt.title(f"p-value map (Time Avg: {mask_name_2} vs {mask_name_1})")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.tight_layout()
        plt.show()
        print("Plotting complete.")

    except FileNotFoundError as e:
        print(f"Error: Data file not found. Check path components.\n{e}")
    except KeyError as e:
        print(f"Error: Variable {e} not found in Zarr store. Check variable names.")
    except AssertionError as e:
        print(f"Error: Data dimension assertion failed. {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


###


def plot_seasonal_comparison_panel(
    mask_name_1,
    mask_name_2,
    mask_data_dict,
    selected_mems_dict,
    ensemble_dir,  # Unused, but kept for signature consistency
    output_dir,
    init_date,  # Used for finding recon file path
    fin_date,  # Used for finding recon file path
    mask_vrange=[0, 1],
    diff_vrange=[-30, 30],
    cmap_mask="Blues",
    cmap_diff="RdBu_r",
    plot_style="seaborn-v0_8-talk",
):
    """
    Plots a comparison panel showing seasonal means (DJF, MAM, JJA, SON)
    of sampling masks and reconstruction differences. Panel layout is
    4 rows (Seasons) x 3 columns (Mask1, Mask2, Difference).
    """

    # Get colormaps using the new alias mpl_cm
    try:
        cmap_mask_obj = mpl_cm.get_cmap(cmap_mask)
        cmap_diff_obj = mpl_cm.get_cmap(cmap_diff)
    except ValueError as e:
        print(
            f"Error getting colormap: {e}. Check if the names '{cmap_mask}' and '{cmap_diff}' are valid matplotlib colormaps."
        )
        return
    except AttributeError:
        # Handle cases where get_cmap might return None or mpl_cm is not as expected
        print(f"Error: Could not retrieve colormaps '{cmap_mask}' or '{cmap_diff}'.")
        return

    # --- Data Loading ---
    try:
        first_ens = list(selected_mems_dict.keys())[0]
        first_mem = selected_mems_dict[first_ens][0]
    except (IndexError, KeyError) as e:
        print(
            f"Error accessing selected members dictionary: {e}. Check the structure of selected_mems_dict."
        )
        return

    print("Loading full time series data...")
    try:
        # Load full masks
        mask1_full = mask_data_dict[mask_name_1]["socat_mask"]
        mask2_full = mask_data_dict[mask_name_2]["socat_mask"]
        # Check if time coordinate exists
        if "time" not in mask1_full.coords or "time" not in mask2_full.coords:
            raise ValueError(
                "Mask data must have a 'time' coordinate for seasonal grouping."
            )
    except KeyError as e:
        print(
            f"Error accessing mask data: Key {e} not found in mask_data_dict. Check mask names ('{mask_name_1}', '{mask_name_2}') and dict structure."
        )
        return
    except ValueError as e:
        print(f"Error with mask data structure: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred loading mask data: {e}")
        return

    # Load full reconstructions paths
    recon_path_1 = f"{output_dir}/reconstructions/{mask_name_1}/{first_ens}/{first_mem}/recon_pCO2_{first_ens}_{first_mem}_mon_1x1_{init_date}_{fin_date}.zarr"
    recon_path_2 = f"{output_dir}/reconstructions/{mask_name_2}/{first_ens}/{first_mem}/recon_pCO2_{first_ens}_{first_mem}_mon_1x1_{init_date}_{fin_date}.zarr"

    # Load full reconstructions
    try:
        print(f"Loading Zarr: {recon_path_1}")
        recon1_full_members = xr.open_zarr(recon_path_1, consolidated=True)[
            "pCO2_recon_full"
        ]
        print(f"Loading Zarr: {recon_path_2}")
        recon2_full_members = xr.open_zarr(recon_path_2, consolidated=True)[
            "pCO2_recon_full"
        ]
        # Check if time coordinate exists
        if (
            "time" not in recon1_full_members.coords
            or "time" not in recon2_full_members.coords
        ):
            raise ValueError(
                "Reconstruction data must have a 'time' coordinate for seasonal grouping."
            )
    except FileNotFoundError as e:
        print(
            f"Error opening reconstruction file: {e}. Check paths and ensure files exist."
        )
        print(f"Path 1: {recon_path_1}")
        print(f"Path 2: {recon_path_2}")
        return
    except KeyError as e:
        print(f"Error: Variable 'pCO2_recon_full' not found in Zarr store: {e}")
        return
    except ValueError as e:
        print(f"Error with reconstruction data structure: {e}")
        return
    except Exception as e:  # Catch other potential zarr/xarray errors
        print(f"An unexpected error occurred opening Zarr stores: {e}")
        return

    # --- Data Processing ---
    try:
        # Handle potential member dimension (select first member)
        member_dim_name = "member"  # Adjust if needed

        # Process recon1
        if member_dim_name in recon1_full_members.dims:
            if recon1_full_members.dims[member_dim_name] > 1:
                print(f"Selecting first member for {mask_name_1}.")
                recon1_full = recon1_full_members.isel({member_dim_name: 0}, drop=True)
            elif recon1_full_members.dims[member_dim_name] == 1:
                print(f"Squeezing single member dim for {mask_name_1}.")
                recon1_full = recon1_full_members.squeeze(
                    dim=member_dim_name, drop=True
                )
            else:  # Dim size 0, unlikely but handle
                print(
                    f"Warning: Member dimension '{member_dim_name}' found with size 0 for {mask_name_1}. Using as is."
                )
                recon1_full = recon1_full_members
        else:
            print(
                f"No member dimension '{member_dim_name}' found or needed for {mask_name_1}."
            )
            recon1_full = recon1_full_members  # Assume it's already (time, lat, lon)

        # Process recon2
        if member_dim_name in recon2_full_members.dims:
            if recon2_full_members.dims[member_dim_name] > 1:
                print(f"Selecting first member for {mask_name_2}.")
                recon2_full = recon2_full_members.isel({member_dim_name: 0}, drop=True)
            elif recon2_full_members.dims[member_dim_name] == 1:
                print(f"Squeezing single member dim for {mask_name_2}.")
                recon2_full = recon2_full_members.squeeze(
                    dim=member_dim_name, drop=True
                )
            else:
                print(
                    f"Warning: Member dimension '{member_dim_name}' found with size 0 for {mask_name_2}. Using as is."
                )
                recon2_full = recon2_full_members
        else:
            print(
                f"No member dimension '{member_dim_name}' found or needed for {mask_name_2}."
            )
            recon2_full = recon2_full_members

        # Check final dimensions
        if len(recon1_full.dims) > 3 or "time" not in recon1_full.dims:
            raise ValueError(
                f"Processed recon1 for {mask_name_1} has unexpected dimensions: {recon1_full.dims}. Expected (time, lat, lon)."
            )
        if len(recon2_full.dims) > 3 or "time" not in recon2_full.dims:
            raise ValueError(
                f"Processed recon2 for {mask_name_2} has unexpected dimensions: {recon2_full.dims}. Expected (time, lat, lon)."
            )

        # --- Calculate Seasonal Means ---
        print("Calculating seasonal means...")
        seasons_order = ["DJF", "MAM", "JJA", "SON"]

        # Group by season and calculate mean over time for each group
        mask1_seasonal = (
            mask1_full.groupby("time.season")
            .mean(dim="time", skipna=True)
            .sel(season=seasons_order)
        )
        mask2_seasonal = (
            mask2_full.groupby("time.season")
            .mean(dim="time", skipna=True)
            .sel(season=seasons_order)
        )
        recon1_seasonal = (
            recon1_full.groupby("time.season")
            .mean(dim="time", skipna=True)
            .sel(season=seasons_order)
        )
        recon2_seasonal = (
            recon2_full.groupby("time.season")
            .mean(dim="time", skipna=True)
            .sel(season=seasons_order)
        )

        # Calculate the difference between the *seasonal means*
        diff_seasonal = recon2_seasonal - recon1_seasonal

        # --- Align Longitude (Roll at 180°) for all seasonal means ---
        print("Aligning longitudes...")
        lon_coord_name = (
            "xlon"
            if "xlon" in mask1_seasonal.dims
            else "lon"
            if "lon" in mask1_seasonal.dims
            else None
        )
        if lon_coord_name:
            lon_size = len(mask1_seasonal[lon_coord_name])

            # Define adjustment function
            def adjust_lon(ds, lon_name, size):
                ds_rolled = ds.roll(**{lon_name: size // 2}, roll_coords=True)
                # Adjust coordinate values to -180 to 180
                ds_rolled[lon_name] = (ds_rolled[lon_name] + 180) % 360 - 180
                # Re-sort by the new longitude values to avoid plotting issues
                ds_rolled = ds_rolled.sortby(lon_name)
                return ds_rolled

            mask1_seasonal = adjust_lon(mask1_seasonal, lon_coord_name, lon_size)
            mask2_seasonal = adjust_lon(mask2_seasonal, lon_coord_name, lon_size)
            # Assuming recons and diff have same lon coord
            recon1_seasonal = adjust_lon(recon1_seasonal, lon_coord_name, lon_size)
            recon2_seasonal = adjust_lon(recon2_seasonal, lon_coord_name, lon_size)
            diff_seasonal = adjust_lon(diff_seasonal, lon_coord_name, lon_size)
        else:
            print(
                "Warning: Could not find standard longitude coordinate ('xlon' or 'lon') for rolling."
            )

    except ValueError as e:
        print(
            f"Error during data processing (member selection, seasonal mean, lon align): {e}"
        )
        return
    except KeyError as e:  # e.g. if 'season' dim not created correctly by groupby
        print(
            f"Error accessing calculated seasonal data: {e}. Check time coordinate validity."
        )
        return
    except Exception as e:
        print(f"An unexpected error occurred during data processing: {e}")
        return

    # --- Plotting ---
    print("Generating seasonal plots...")
    try:
        # Check if SpatialMap2 class is available
        if "SpatialMap2" not in globals():
            raise NameError("SpatialMap2 class is not defined or imported.")

        with plt.style.context(plot_style):
            # Create a figure with 4 rows (one for each season) and 3 columns
            fig = plt.figure(figsize=(15, 16), dpi=200)  # Taller figure

            # Initialize map class - ensure it can handle multi-row axes
            worldmap = SpatialMap2(
                fig=fig,
                region="world",
                cbar_mode="each",  # One colorbar per plot
                colorbar=True,
                cbar_location="bottom",
                nrows_ncols=[4, 3],  # 4 rows, 3 columns
            )

            # Determine coordinate names for plotting
            lon_coord_plot = (
                lon_coord_name
                if lon_coord_name
                else ("xlon" if "xlon" in mask1_seasonal.coords else "lon")
            )  # Best guess if roll failed
            lat_coord_plot = "ylat" if "ylat" in mask1_seasonal.coords else "lat"

            # Loop through each season and plot the corresponding row
            for i, season in enumerate(seasons_order):
                print(f"  Plotting season: {season}")
                # Calculate axis indices for the current row
                ax_idx_mask1 = i * 3 + 0
                ax_idx_mask2 = i * 3 + 1
                ax_idx_diff = i * 3 + 2

                # Select data for the current season
                mask1_s = mask1_seasonal.sel(season=season)
                mask2_s = mask2_seasonal.sel(season=season)
                diff_s = diff_seasonal.sel(season=season)

                # --- Plotting Column 1: Mask 1 ---
                try:
                    sub0 = worldmap.add_plot(
                        lon=mask1_s[lon_coord_plot],
                        lat=mask1_s[lat_coord_plot],
                        data=mask1_s,
                        vrange=mask_vrange,
                        cmap=cmap_mask_obj,
                        ax=ax_idx_mask1,  # Use cmap object
                    )
                    worldmap.set_title(
                        f"Mask: {mask_name_1} ({season})", ax=ax_idx_mask1, fontsize=11
                    )
                    cbar0 = worldmap.add_colorbar(sub0, ax=ax_idx_mask1)
                    worldmap.set_cbar_xlabel(
                        cbar0, "Mean Sampling Presence", fontsize=9
                    )
                except Exception as plot_err:
                    print(f"Error plotting Mask 1 for {season}: {plot_err}")
                    # Try to set title even if plot fails
                    try:
                        worldmap.set_title(
                            f"Mask: {mask_name_1} ({season})\nPLOT ERROR",
                            ax=ax_idx_mask1,
                            fontsize=11,
                        )
                    except:
                        pass  # Ignore error setting error title

                # --- Plotting Column 2: Mask 2 ---
                try:
                    sub1 = worldmap.add_plot(
                        lon=mask2_s[lon_coord_plot],
                        lat=mask2_s[lat_coord_plot],
                        data=mask2_s,
                        vrange=mask_vrange,
                        cmap=cmap_mask_obj,
                        ax=ax_idx_mask2,  # Use cmap object
                    )
                    worldmap.set_title(
                        f"Mask: {mask_name_2} ({season})", ax=ax_idx_mask2, fontsize=11
                    )
                    cbar1 = worldmap.add_colorbar(sub1, ax=ax_idx_mask2)
                    worldmap.set_cbar_xlabel(
                        cbar1, "Mean Sampling Presence", fontsize=9
                    )
                except Exception as plot_err:
                    print(f"Error plotting Mask 2 for {season}: {plot_err}")
                    try:
                        worldmap.set_title(
                            f"Mask: {mask_name_2} ({season})\nPLOT ERROR",
                            ax=ax_idx_mask2,
                            fontsize=11,
                        )
                    except:
                        pass

                # --- Plotting Column 3: Difference ---
                try:
                    sub2 = worldmap.add_plot(
                        lon=diff_s[lon_coord_plot],
                        lat=diff_s[lat_coord_plot],
                        data=diff_s,
                        vrange=diff_vrange,
                        cmap=cmap_diff_obj,
                        ax=ax_idx_diff,  # Use cmap object
                    )
                    worldmap.set_title(
                        f"Mean Recon Diff ({season})\n({mask_name_2} - {mask_name_1})",
                        ax=ax_idx_diff,
                        fontsize=11,
                    )
                    cbar2 = worldmap.add_colorbar(sub2, ax=ax_idx_diff)
                    worldmap.set_cbar_xlabel(
                        cbar2, f"Mean Δ pCO₂ ({season}) (µatm)", fontsize=9
                    )
                except Exception as plot_err:
                    print(f"Error plotting Difference for {season}: {plot_err}")
                    try:
                        worldmap.set_title(
                            f"Mean Recon Diff ({season})\nPLOT ERROR",
                            ax=ax_idx_diff,
                            fontsize=11,
                        )
                    except:
                        pass

            # Adjust layout
            plt.tight_layout(
                rect=[0, 0.03, 1, 0.97]
            )  # Added rect to give slight room for bottom cbars/top titles
            # plt.subplots_adjust(hspace=0.3, wspace=0.15, bottom=0.08, top=0.95) # Alternative fine-tuning

            plt.show()
            print("Plotting complete.")

    except NameError as e:
        print(
            f"Plotting Error: {e}. Ensure the SpatialMap2 class is defined or imported and initialized correctly."
        )
        # Close the figure if it was created but SpatialMap2 failed
        if "fig" in locals() and plt.fignum_exists(fig.number):
            plt.close(fig)
    except Exception as e:
        print(f"An unexpected error occurred during plotting setup or looping: {e}")
        if "fig" in locals() and plt.fignum_exists(fig.number):
            plt.close(fig)


###
from scipy.stats import norm


def calculate_p_value_map(mean_A, std_A, mean_B, std_B):
    """Calculate p-value map between two prediction fields (mean, std). Handles NaNs."""
    epsilon = 1e-9
    var_A = np.square(std_A) + epsilon
    var_B = np.square(std_B) + epsilon
    p_values = np.full_like(mean_A, np.nan)
    valid_mask = (
        np.isfinite(mean_A)
        & np.isfinite(std_A)
        & np.isfinite(mean_B)
        & np.isfinite(std_B)
    )
    z_scores = (mean_B[valid_mask] - mean_A[valid_mask]) / np.sqrt(
        var_A[valid_mask] + var_B[valid_mask]
    )
    p_values[valid_mask] = 2 * (1 - norm.cdf(np.abs(z_scores)))
    return p_values


def plot_reconstruction_comparison_panel_full(
    mask_name_1,
    mask_name_2,
    selected_mems_dict,
    output_dir,
    init_date,
    fin_date,
    diff_vrange_recon=[-30, 30],
    diff_vrange_std=[-5, 5],
    pval_vrange=[0, 0.1],
    cmap_diff="RdBu_r",
    cmap_pval="viridis_r",
    plot_style="seaborn-v0_8-talk",
):
    """
    Plots a 3-panel comparison:
    - Mean Reconstruction Difference
    - Mean STD Difference
    - p-value map
    Using SpatialMap2.
    """
    cmap_diff = mpl_cm.get_cmap(cmap_diff)
    cmap_pval = mpl_cm.get_cmap(cmap_pval)

    try:
        first_ens = list(selected_mems_dict.keys())[0]
        first_mem = selected_mems_dict[first_ens][0]

        base_path_str = f"{output_dir}/reconstructions/{{mask_name}}/{first_ens}/{first_mem}/recon_pCO2_{first_ens}_{first_mem}_mon_1x1_{init_date}_{fin_date}.zarr"

        # --- Load mean reconstructions ---
        path1 = base_path_str.format(mask_name=mask_name_1)
        path2 = base_path_str.format(mask_name=mask_name_2)

        recon1_full = xr.open_zarr(path1, consolidated=True)
        recon2_full = xr.open_zarr(path2, consolidated=True)

        mean1 = (
            recon1_full["pCO2_recon_full_mean"]
            .mean(dim="time", skipna=True)
            .squeeze(drop=True)
        )
        mean2 = (
            recon2_full["pCO2_recon_full_mean"]
            .mean(dim="time", skipna=True)
            .squeeze(drop=True)
        )

        # --- Load mean STDs ---
        std1 = (
            recon1_full["pCO2_recon_full_std"]
            .mean(dim="time", skipna=True)
            .squeeze(drop=True)
        )
        std2 = (
            recon2_full["pCO2_recon_full_std"]
            .mean(dim="time", skipna=True)
            .squeeze(drop=True)
        )

        # --- Calculate differences ---
        mean_diff = mean2 - mean1
        std_diff = std2 - std1

        # --- Calculate p-value map ---
        print("Calculating p-value map...")
        pval_map = calculate_p_value_map(
            mean1.values, std1.values, mean2.values, std2.values
        )
        print(f"Number of valid (non-NaN) p-values: {np.isfinite(pval_map).sum()}")

        # --- Get coordinates ---
        lon_coords = mean1["xlon"] if "xlon" in mean1.coords else mean1["lon"]
        lat_coords = mean1["ylat"] if "ylat" in mean1.coords else mean1["lat"]

        # --- Plotting ---
        if "SpatialMap2" not in globals():
            raise NameError("SpatialMap2 class is not defined or imported.")

        with plt.style.context(plot_style):
            fig = plt.figure(figsize=(18, 5), dpi=200)
            worldmap = SpatialMap2(
                fig=fig,
                region="world",
                cbar_mode="each",
                colorbar=True,
                cbar_location="bottom",
                nrows_ncols=[1, 3],
            )

            # 1. Mean Reconstruction Difference
            sub0 = worldmap.add_plot(
                lon=lon_coords,
                lat=lat_coords,
                data=mean_diff,
                vrange=diff_vrange_recon,
                cmap=cmap_diff,
                ax=0,
            )
            worldmap.set_title(
                f"Mean Reconstruction Difference\n({mask_name_2} - {mask_name_1})",
                ax=0,
                fontsize=13,
            )

            # 2. Mean STD Difference
            sub1 = worldmap.add_plot(
                lon=lon_coords,
                lat=lat_coords,
                data=std_diff,
                vrange=diff_vrange_std,
                cmap=cmap_diff,
                ax=1,
            )
            worldmap.set_title(
                f"Mean STD Difference\n({mask_name_2} - {mask_name_1})",
                ax=1,
                fontsize=13,
            )

            # 3. p-value Map
            sub2 = worldmap.add_plot(
                lon=lon_coords,
                lat=lat_coords,
                data=pval_map,
                vrange=pval_vrange,
                cmap=cmap_pval,
                ax=2,
            )
            worldmap.set_title(
                f"p-value Map\n({mask_name_2} vs {mask_name_1})", ax=2, fontsize=13
            )

            # Colorbars
            # cbar0 = worldmap.add_colorbar(sub0, ax=0)
            # cbar1 = worldmap.add_colorbar(sub1, ax=1)
            # cbar2 = worldmap.add_colorbar(sub2, ax=2)
            cbar0 = worldmap.add_colorbar(
                sub0, ax=0, cmap=cmap_diff, vrange=diff_vrange_recon
            )
            cbar1 = worldmap.add_colorbar(
                sub1, ax=1, cmap=cmap_diff, vrange=diff_vrange_std
            )
            cbar2 = worldmap.add_colorbar(
                sub2, ax=2, cmap=cmap_pval, vrange=pval_vrange
            )

            worldmap.set_cbar_xlabel(cbar0, "\u0394 pCO₂ (\u03bcatm)", fontsize=11)
            worldmap.set_cbar_xlabel(cbar1, "\u0394 STD pCO₂ (\u03bcatm)", fontsize=11)
            worldmap.set_cbar_xlabel(cbar2, "p-value", fontsize=11)

            plt.tight_layout()
            plt.show()
            print("Plotting complete.")

    except FileNotFoundError as e:
        print(f"Error: File not found. {e}")
    except KeyError as e:
        print(f"Error: Key error. {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")




def plot_seasonal_difference_panel_1x4( # Renamed function
    mask_name_1,
    mask_name_2,
    selected_mems_dict,
    ensemble_dir,
    output_dir,
    init_date,
    fin_date,
    diff_vrange=[-30, 30],
    cmap_diff="RdBu_r",
    plot_style="seaborn-v0_8-talk",
):
    """
    Plots a 1x4 panel (single row) showing seasonal means (DJF, MAM, JJA, SON)
    of reconstruction differences (mask_name_2 - mask_name_1).
    """
    # Use the new alias mpl_cm here
    cmap_diff_obj = mpl_cm.get_cmap(cmap_diff)

    first_ens = list(selected_mems_dict.keys())[0]
    first_mem = selected_mems_dict[first_ens][0]

    # --- Load FULL Reconstruction Data ---
    print("Loading full time series data for reconstructions...")
    recon_path_1 = f"{output_dir}/reconstructions/{mask_name_1}/{first_ens}/{first_mem}/recon_pCO2_{first_ens}_{first_mem}_mon_1x1_{init_date}_{fin_date}.zarr"
    recon_path_2 = f"{output_dir}/reconstructions/{mask_name_2}/{first_ens}/{first_mem}/recon_pCO2_{first_ens}_{first_mem}_mon_1x1_{init_date}_{fin_date}.zarr"

    print(f"Loading Zarr: {recon_path_1}")
    recon1_full_members = xr.open_zarr(recon_path_1, consolidated=True)["pCO2_recon_full"]
    print(f"Loading Zarr: {recon_path_2}")
    recon2_full_members = xr.open_zarr(recon_path_2, consolidated=True)["pCO2_recon_full"]

    # --- Handle Member Dimension ---
    member_dim_name = 'member'
    # Process recon1
    if member_dim_name in recon1_full_members.dims and recon1_full_members.dims[member_dim_name] > 1:
        recon1_full = recon1_full_members.isel({member_dim_name: 0})
    elif member_dim_name in recon1_full_members.dims and recon1_full_members.dims[member_dim_name] == 1:
        recon1_full = recon1_full_members.squeeze(dim=member_dim_name, drop=True)
    else:
        recon1_full = recon1_full_members
    # Process recon2
    if member_dim_name in recon2_full_members.dims and recon2_full_members.dims[member_dim_name] > 1:
        recon2_full = recon2_full_members.isel({member_dim_name: 0})
    elif member_dim_name in recon2_full_members.dims and recon2_full_members.dims[member_dim_name] == 1:
        recon2_full = recon2_full_members.squeeze(dim=member_dim_name, drop=True)
    else:
        recon2_full = recon2_full_members
    print("Selected first member (if applicable) for both reconstructions.")

    # --- Calculate Seasonal Means ---
    print("Calculating seasonal means for reconstructions...")
    seasons_order = ['DJF', 'MAM', 'JJA', 'SON']
    recon1_seasonal = recon1_full.groupby('time.season').mean(dim="time").sel(season=seasons_order)
    recon2_seasonal = recon2_full.groupby('time.season').mean(dim="time").sel(season=seasons_order)
    diff_seasonal = recon2_seasonal - recon1_seasonal
    print("Seasonal means calculated.")

    # --- Align Longitude ---
    print("Aligning longitude for difference map...")
    # Use 'lon' if 'xlon' is not present, adapt as necessary
    lon_coord_name = 'xlon' if 'xlon' in diff_seasonal.dims else 'lon' if 'lon' in diff_seasonal.dims else None

    if lon_coord_name:
         diff_seasonal = diff_seasonal.roll({lon_coord_name: len(diff_seasonal[lon_coord_name]) // 2}, roll_coords=True)
         diff_seasonal[lon_coord_name] = (diff_seasonal[lon_coord_name] + 180) % 360 - 180
         print(f"Longitude coordinates ('{lon_coord_name}') adjusted to -180 to 180.")
    else:
         print("Warning: Longitude dimension ('xlon' or 'lon') not found in diff_seasonal. Skipping longitude alignment.")


    # --- Plotting ---
    print("Generating 1x4 seasonal difference plots...")
    with plt.style.context(plot_style):
        # Create a figure with 1 row and 4 columns - make it wide
        fig = plt.figure(figsize=(22, 6), dpi=200) # Adjusted figsize for 1x4

        try:
            # Check if SpatialMap2 exists before trying to use it
            if 'SpatialMap2' not in globals():
                 raise NameError("SpatialMap2 class is not defined. Please define or import it.")

            # Initialize your map class with 1x4 layout
            worldmap = SpatialMap2(
                fig=fig,
                region="world",
                cbar_mode="each",
                colorbar=True,
                cbar_location="bottom",
                nrows_ncols=[1, 4] # **** CHANGED TO 1x4 ****
            )
        except NameError as e:
             print(f"Error: {e}")
             plt.close(fig)
             return
        except Exception as e:
             print(f"Error initializing SpatialMap2: {e}")
             plt.close(fig)
             return

        # Check for latitude coordinate name
        lat_coord_name = 'ylat' if 'ylat' in diff_seasonal.dims else 'lat' if 'lat' in diff_seasonal.dims else None
        if not lat_coord_name:
            print("Error: Latitude dimension ('ylat' or 'lat') not found in diff_seasonal. Cannot plot.")
            plt.close(fig)
            return
        if not lon_coord_name: # Re-check after alignment attempt
             print("Error: Longitude dimension ('xlon' or 'lon') not found in diff_seasonal. Cannot plot.")
             plt.close(fig)
             return


        # Loop through each season and plot the difference
        # The axis index 'i' (0, 1, 2, 3) directly corresponds to the
        # flattened index of the 1x4 grid.
        for i, season in enumerate(seasons_order):
            print(f"  Plotting difference for season: {season} on axis index {i}")
            ax_idx_diff  = i # Direct mapping: 0->leftmost, ..., 3->rightmost

            diff_s = diff_seasonal.sel(season=season)

            # Plot the difference map for the current season
            try:
                sub_diff = worldmap.add_plot(
                    lon=diff_s[lon_coord_name], # Use identified lon coord
                    lat=diff_s[lat_coord_name], # Use identified lat coord
                    data=diff_s,
                    vrange=diff_vrange,
                    cmap=cmap_diff_obj, # Pass the actual colormap object
                    ax=ax_idx_diff,
                )
                worldmap.set_title(
                    f"Mean Recon Diff ({season})\n({mask_name_2} - {mask_name_1})",
                    ax=ax_idx_diff, fontsize=11, # Adjust font size if needed
                )
                cbar_diff = worldmap.add_colorbar(sub_diff, ax=ax_idx_diff)
                # Make sure colorbar label font size is appropriate
                worldmap.set_cbar_xlabel(cbar_diff, f"Mean Δ pCO₂ ({season}) (µatm)", fontsize=9)

                # Optional: Add coastlines/land if SpatialMap2 supports it
                # worldmap.add_coastlines(ax=ax_idx_diff)
                # worldmap.add_land(ax=ax_idx_diff)

            except Exception as e:
                 print(f"Error plotting Difference for {season} on axis {ax_idx_diff}: {e}")
                 # Attempt to set an error title even if plotting fails
                 try:
                    worldmap.set_title(f"Mean Recon Diff ({season})\nPLOT ERROR", ax=ax_idx_diff, fontsize=11)
                 except:
                    print(f"  Could not set error title for axis {ax_idx_diff}.")


        # Adjust layout for 1x4 grid
        # Use subplots_adjust, focusing on wspace and bottom/top margins
        plt.subplots_adjust(
            left=0.04,    # Small left margin
            right=0.98,   # Small right margin
            bottom=0.22,  # Need enough space for colorbars + labels
            top=0.85,     # Need space for titles
            wspace=0.15,  # Space between the plots (adjust as needed)
            hspace=0      # Not applicable for single row
        )
        print(f"Using subplots_adjust for 1x4 layout.")


        plt.show()
        print("Plotting complete.")
