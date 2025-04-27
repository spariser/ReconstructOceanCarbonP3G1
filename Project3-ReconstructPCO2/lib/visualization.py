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
    cmap = cm.thermal

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
    cmap = cm.thermal

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

    cmap = cm.thermal

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
    cmap = cm.thermal

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
