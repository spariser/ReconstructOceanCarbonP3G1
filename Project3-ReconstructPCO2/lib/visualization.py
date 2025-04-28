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

    def add_colorbar(self, sub, ax=0, *args, **kwargs):
        """
        add_colorbar(sub, ax, **kwargs)

        Inputs:
        ==============
        sub : subplot (this is returuned from add_plot())
        ax. : axis number to add colorbar to

        """
        # Weird whitespace when you use 'extend'
        # The workaround is to make a colorbar
        # Help from : https://github.com/matplotlib/matplotlib/issues/9778

        # col = self.grid.cbar_axes[ax].colorbar(sub, *args, **kwargs)
        col = mpl.colorbar.ColorbarBase(
            self.grid.cbar_axes[ax],
            orientation=self.cbar_orientation,
            cmap=self.cmap,
            norm=mpl.colors.Normalize(vmin=self.vrange[0], vmax=self.vrange[1]),
            *args,
            **kwargs,
        )

        # cb2 = mpl.colorbar.ColorbarBase(ax, cmap=cmap,
        #                                norm=norm,
        #                                boundaries=[0] + bounds + [13],
        #                                extend='both',
        #                                ticks=bounds,
        #                                spacing='proportional',
        #                                orientation='horizontal')

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


def plot_reconstruction_vs_truth(
    mask_name,
    mask_data_dict,
    selected_mems_dict,
    ensemble_dir,
    output_dir,
    dates,
    init_date,
    fin_date,
    vrange=[280, 440],
    chosen_time="2021-01",
):
    """
    Plots the comparison between reconstructed pCO₂ data and the original
    (truth) data for a specific time period and region.

    Parameters:
        mask_name (str): Name of the mask used for SOCAT data.
        mask_data_dict (dict): Dictionary containing SOCAT mask datasets.
        selected_mems_dict (dict): Dictionary specifying selected ensemble
            members for reconstruction.
        ensemble_dir (str): Path to the directory containing ensemble data.
        output_dir (str): Path to the output directory for reconstructions.
        dates (list): List of dates defining the time range for the data.
        init_date (str): Initial date for the reconstruction period (format: YYYY-MM).
        fin_date (str): Final date for the reconstruction period (format: YYYY-MM).
        vrange (list, optional): Value range for the colorbar. Defaults to [280, 440].
        chosen_time (str, optional): Specific time (month) to plot (format: YYYY-MM).
            Defaults to "2021-01".
    """
    cmap = cm.cm.thermal

    # Select the first ensemble and member
    first_ens = list(selected_mems_dict.keys())[0]
    first_mem = selected_mems_dict[first_ens][0]

    # Load original member data from ESM output
    fs = gcsfs.GCSFileSystem()
    member_dir = f"{ensemble_dir}/{first_ens}/{first_mem}"
    member_path = fs.glob(f"{member_dir}/*.zarr")[0]
    member_data = xr.open_zarr("gs://" + member_path).sel(
        time=slice(str(dates[0]), str(dates[-1]))
    )
    print("Member path:", member_path)

    # Load reconstructed pCO₂ data
    recon_output_dir = f"{output_dir}/reconstructions/{mask_name}"
    recon_dir = f"{recon_output_dir}/{first_ens}/{first_mem}"
    recon_path = f"{recon_dir}/recon_pCO2_{first_ens}_{first_mem}_mon_1x1_{init_date}_{fin_date}.zarr"
    print("Recon path:", recon_path)
    full = xr.open_zarr(recon_path, consolidated=True)["pCO2_recon_full"]

    # Extract specific month
    raw_data = member_data["spco2"].sel(time=chosen_time).squeeze()
    recon_data = full.sel(time=chosen_time)[0, ...]

    # Shift longitudes for global plotting
    raw_data = raw_data.roll(xlon=len(raw_data.xlon) // 2, roll_coords=True)
    recon_data = recon_data.roll(xlon=len(recon_data.xlon) // 2, roll_coords=True)

    # Load SOCAT mask and align
    mask_data = mask_data_dict[mask_name]
    mask = mask_data.sel(time=chosen_time)["socat_mask"].squeeze()
    mask = mask.roll(xlon=len(mask.xlon) // 2, roll_coords=True)

    # Mask the raw data
    masked_raw = np.ma.masked_array(raw_data, mask=(mask == 0))
    # masked_raw = raw_data

    # Start plotting
    with plt.style.context("seaborn-v0_8-talk"):
        fig = plt.figure(figsize=(8, 3), dpi=200)
        worldmap = SpatialMap2(
            fig=fig,
            region="world",
            cbar_mode="single",
            colorbar=True,
            cbar_location="bottom",
            nrows_ncols=[1, 2],
        )

        sub0 = worldmap.add_plot(
            lon=raw_data["xlon"],
            lat=raw_data["ylat"],
            data=masked_raw,
            vrange=vrange,
            cmap=cmap,
            ax=0,
        )

        sub1 = worldmap.add_plot(
            lon=recon_data["xlon"],
            lat=recon_data["ylat"],
            data=recon_data,
            vrange=vrange,
            cmap=cmap,
            ax=1,
        )

        worldmap.set_title(f"{mask_name} pCO₂ ({chosen_time})", ax=0, fontsize=13)
        worldmap.set_title(f"pCO₂ Reconstruction  ({chosen_time})", ax=1, fontsize=13)

        colorbar = worldmap.add_colorbar(sub0, ax=0)
        worldmap.set_cbar_xlabel(colorbar, "pCO₂ (µatm)", fontsize=12)

        plt.show()


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


def plot_reconstruction_std_side_by_side(
    mask_name_1,
    mask_name_2,
    selected_mems_dict,
    output_dir,
    init_date,
    fin_date,
    chosen_time="2021-01",
    vrange=[280, 440],
):
    """
    Plot the STD of reconstructed pCO₂ side-by-side for two masking strategies.

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
    chosen_time : str, optional
        Specific time (month) to extract and plot, in the format 'YYYY-MM'. Default is "2021-01".
    vrange : list, optional
        Value range for the color scale in the plot. Default is [280, 440].
    """

    cmap = cm.cm.thermal

    # Select the first ensemble and member
    first_ens = list(selected_mems_dict.keys())[0]
    first_mem = selected_mems_dict[first_ens][0]

    # Load reconstructed pCO₂ STD for both masks
    recon_path_1 = f"{output_dir}/reconstructions/{mask_name_1}/{first_ens}/{first_mem}/recon_pCO2_{first_ens}_{first_mem}_mon_1x1_{init_date}_{fin_date}.zarr"
    recon_path_2 = f"{output_dir}/reconstructions/{mask_name_2}/{first_ens}/{first_mem}/recon_pCO2_{first_ens}_{first_mem}_mon_1x1_{init_date}_{fin_date}.zarr"

    full_std_1 = xr.open_zarr(recon_path_1, consolidated=True)["pCO2_recon_full_std"]
    full_std_2 = xr.open_zarr(recon_path_2, consolidated=True)["pCO2_recon_full_std"]

    # Extract specific month
    std_1 = full_std_1.sel(time=chosen_time)[0, ...]
    std_2 = full_std_2.sel(time=chosen_time)[0, ...]

    # Align longitudes
    std_1 = std_1.roll(xlon=len(std_1.xlon) // 2, roll_coords=True)
    std_2 = std_2.roll(xlon=len(std_2.xlon) // 2, roll_coords=True)

    # Start plotting
    with plt.style.context("seaborn-v0_8-talk"):
        fig = plt.figure(figsize=(12, 4), dpi=200)
        worldmap = SpatialMap2(
            fig=fig,
            region="world",
            cbar_mode="single",
            colorbar=True,
            cbar_location="bottom",
            nrows_ncols=[1, 2],
        )

        sub0 = worldmap.add_plot(
            lon=std_1["xlon"],
            lat=std_1["ylat"],
            data=std_1,
            vrange=vrange,
            cmap=cmap,
            ax=0,
        )

        sub1 = worldmap.add_plot(
            lon=std_2["xlon"],
            lat=std_2["ylat"],
            data=std_2,
            vrange=vrange,
            cmap=cmap,
            ax=1,
        )

        worldmap.set_title(f"STD {mask_name_1} ({chosen_time})", ax=0, fontsize=13)
        worldmap.set_title(f"STD {mask_name_2} ({chosen_time})", ax=1, fontsize=13)

        colorbar = worldmap.add_colorbar(sub0, ax=0)
        worldmap.set_cbar_xlabel(colorbar, "STD pCO₂ (µatm)", fontsize=12)

        plt.show()


def plot_reconstruction_std_difference(
    mask_name_1,
    mask_name_2,
    selected_mems_dict,
    output_dir,
    init_date,
    fin_date,
    chosen_time="2021-01",
    diff_vrange=[-20, 20],
):
    """
    Visualize the difference between the standard deviation (STD) maps of two reconstructed
    pCO₂ datasets derived from different masking strategies.

    This function computes the difference between the STDs of two reconstructions
    (calculated as Mask2 - Mask1) and displays the result on a world map with a
    specified color range.

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
    chosen_time : str, optional
        Specific time (month) to extract and plot, in the format 'YYYY-MM'. Default is "2021-01".
    diff_vrange : list, optional
        Value range for the color scale in the plot. Default is [-20, 20].
    """
    cmap_diff = cm.thermal

    # Select the first ensemble and member
    first_ens = list(selected_mems_dict.keys())[0]
    first_mem = selected_mems_dict[first_ens][0]

    # Load reconstructed pCO₂ std for both masks
    recon_path_1 = f"{output_dir}/reconstructions/{mask_name_1}/{first_ens}/{first_mem}/recon_pCO2_{first_ens}_{first_mem}_mon_1x1_{init_date}_{fin_date}.zarr"
    recon_path_2 = f"{output_dir}/reconstructions/{mask_name_2}/{first_ens}/{first_mem}/recon_pCO2_{first_ens}_{first_mem}_mon_1x1_{init_date}_{fin_date}.zarr"

    full_std_1 = xr.open_zarr(recon_path_1, consolidated=True)["pCO2_recon_full_std"]
    full_std_2 = xr.open_zarr(recon_path_2, consolidated=True)["pCO2_recon_full_std"]

    # Extract specific month
    std_1 = full_std_1.sel(time=chosen_time)[0, ...]
    std_2 = full_std_2.sel(time=chosen_time)[0, ...]

    # Align longitudes
    std_1 = std_1.roll(xlon=len(std_1.xlon) // 2, roll_coords=True)
    std_2 = std_2.roll(xlon=len(std_2.xlon) // 2, roll_coords=True)

    # Compute the difference (Mask2 - Mask1)
    std_diff = std_2 - std_1

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

        sub0 = worldmap.add_plot(
            lon=std_diff["xlon"],
            lat=std_diff["ylat"],
            data=std_diff,
            vrange=diff_vrange,
            cmap=cmap_diff,
            ax=0,
        )

        worldmap.set_title(
            f"STD Difference ({mask_name_2} - {mask_name_1}) ({chosen_time})",
            ax=0,
            fontsize=13,
        )

        colorbar = worldmap.add_colorbar(sub0, ax=0)
        worldmap.set_cbar_xlabel(colorbar, "Δ STD pCO₂ (µatm)", fontsize=12)

        plt.show()


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
            cbar0 = worldmap.add_colorbar(sub0, ax=0)
            cbar1 = worldmap.add_colorbar(sub1, ax=1)
            cbar2 = worldmap.add_colorbar(sub2, ax=2)

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
