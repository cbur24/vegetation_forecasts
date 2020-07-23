import numpy as np
import xarray as xr
import geopandas as gpd
import sys
from datacube.storage import masking
from datacube.utils import geometry
from datacube.helpers import ga_pq_fuser

from pyproj import Proj, transform




def calculate_anomalies(shp_fpath, collection, year, season, query_box, dask_chunks):

    """
    This function will load three months worth of satellite 
    images using the specified season, calculate the seasonal NDVI mean
    of the input timeseries, then calculate a standardised seasonal
    NDVI anomaly by comparing the NDVI seasonal mean with
    pre-calculate NDVI climatology means and standard deviations.
    
    Parameters
    ----------
    shp_fpath`: string 
        Provide a filepath to a shapefile that defines your AOI
    query_box; tuple
        A tuple of the form (lat,lon,buffer) to delineate an AOI if not
        providing a shapefile
    collection`: string
        The landsat collection to load data from. either 'c3' or 'c2'
    year : string
        The year of interest to e.g. '2018'. This will be combined with
        the 'season' to generate a time period to load data from.
    season : string
        The season of interest, e.g `DJF','JFM','FMA' etc
    dask_chunks : Dict 
        Dictionary of values to chunk the data using dask e.g. `{'x':3000, 'y':3000}`
    
    Returns
    -------
    xarr : xarray.DataArray
        A data array showing the seasonl NDVI standardised anomalies.
    
    """

    # dict of all seasons for indexing datacube
    all_seasons = {
        "JFM": [1, 2, 3],
        "FMA": [2, 3, 4],
        "MAM": [3, 4, 5],
        "AMJ": [4, 5, 6],
        "MJJ": [5, 6, 7],
        "JJA": [6, 7, 8],
        "JAS": [7, 8, 9],
        "ASO": [8, 9, 10],
        "SON": [9, 10, 11],
        "OND": [10, 11, 12],
        "NDJ": [11, 12, 1],
        "DJF": [12, 1, 2],
    }

    if season not in all_seasons:
        raise ValueError(
            "Not a valid season, " "must be one of: " + str(all_seasons.keys())
        )

    # Depending on the season, grab the time for the dc.load
    months = all_seasons.get(season)

    if (season == "DJF") or (season == "NDJ"):
        time = (year + "-" + str(months[0]), str(int(year) + 1) + "-" + str(months[2]))

    else:
        time = (year + "-" + str(months[0]), year + "-" + str(months[2]))

    # connect to datacube
    try:
        if collection == "c3":
            dc = datacube.Datacube(app="calculate_anomalies", env="c3-samples")
        if collection == "c2":
            dc = datacube.Datacube(app="calculate_anomalies")
    except:
        raise ValueError("collection must be either 'c3' or 'c2'")

    # get data from shapefile extent and mask
    if shp_fpath is not None:
        # open shapefile with geopandas
        gdf = gpd.read_file(shp_fpath).to_crs(crs={"init": str("epsg:4326")})

        if len(gdf) > 1:
            warnings.warn(
                "This script can only accept shapefiles with a single polygon feature; "
                "seasonal anomalies will be calculated for the extent of the "
                "first geometry in the shapefile only."
            )

        print("extracting data based on shapefile extent")

        # set up query based on polygon (convert to WGS84)
        geom = geometry.Geometry(
            gdf.geometry.values[0].__geo_interface__, geometry.CRS("epsg:4326")
        )

        query = {"geopolygon": geom, "time": time}

        if collection == "c3":

            ds = load_ard(
                dc=dc,
                products=["ga_ls5t_ard_3", "ga_ls7e_ard_3", "ga_ls8c_ard_3"],
                measurements=["nbart_nir", "nbart_red"],
                ls7_slc_off=False,
                # align = (15,15),
                resolution=(-30, 30),
                output_crs="epsg:3577",
                dask_chunks=dask_chunks,
                group_by="solar_day",
                **query,
            )

        if collection == "c2":
            print("loading Landsat C2")
            ds = dc.load(
                product="ls8_nbart_albers",
                group_by="solar_day",
                measurements=["nir", "red"],
                resolution=(-30, 30),
                output_crs="epsg:3577",
                dask_chunks=dask_chunks,
                **query,
            )

            # Load PQ data
            print("loading Landsat C2 pq data")
            pq = dc.load(
                product="ls8_pq_albers",
                group_by="solar_day",
                fuse_func=ga_pq_fuser,
                resolution=(-30, 30),
                output_crs="epsg:3577",
                dask_chunks=dask_chunks,
                **query,
            )

            print("making pq mask")
            good_quality = masking.make_mask(
                pq.pixelquality,
                cloud_acca="no_cloud",
                cloud_shadow_acca="no_cloud_shadow",
                cloud_shadow_fmask="no_cloud_shadow",
                cloud_fmask="no_cloud",
                blue_saturated=False,
                green_saturated=False,
                red_saturated=False,
                nir_saturated=False,
                swir1_saturated=False,
                swir2_saturated=False,
                contiguous=True,
            )

            attrs = ds.attrs
            ds = ds.astype(np.float32)
            ds = ds.where(good_quality)
            ds = masking.mask_invalid_data(ds)
            ds.attrs.update(attrs)

        # create polygon mask
        with HiddenPrints():
            mask = xr_rasterize(gdf.iloc[[0]], ds)
        # mask dataset
        ds = ds.where(mask)

    else:
        print("Extracting data based on lat, lon coords")
        query = {
            "lon": (query_box[1] - query_box[2], query_box[1] + query_box[2]),
            "lat": (query_box[0] - query_box[2], query_box[0] + query_box[2]),
            "time": time,
        }

        if collection == "c3":

            ds = load_ard(
                dc=dc,
                products=["ga_ls5t_ard_3", "ga_ls7e_ard_3", "ga_ls8c_ard_3"],
                measurements=["nbart_nir", "nbart_red"],
                ls7_slc_off=False,
                # align = (15,15),
                resolution=(-30, 30),
                output_crs="epsg:3577",
                dask_chunks=dask_chunks,
                group_by="solar_day",
                **query,
            )

        if collection == "c2":

            print("loading Landsat collection 2")
            ds = dc.load(
                product="ls8_nbart_albers",
                group_by="solar_day",
                measurements=["nir", "red"],
                resolution=(-30, 30),
                output_crs="epsg:3577",
                dask_chunks=dask_chunks,
                **query,
            )

            # Load PQ data
            pq = dc.load(
                product="ls8_pq_albers",
                group_by="solar_day",
                fuse_func=ga_pq_fuser,
                resolution=(-30, 30),
                output_crs="epsg:3577",
                dask_chunks=dask_chunks,
                **query,
            )

            good_quality = masking.make_mask(
                pq.pixelquality,
                cloud_acca="no_cloud",
                cloud_shadow_acca="no_cloud_shadow",
                cloud_shadow_fmask="no_cloud_shadow",
                cloud_fmask="no_cloud",
                blue_saturated=False,
                green_saturated=False,
                red_saturated=False,
                nir_saturated=False,
                swir1_saturated=False,
                swir2_saturated=False,
                contiguous=True,
            )

            attrs = ds.attrs
            ds = ds.astype(np.float32)
            ds = ds.where(good_quality)
            ds = masking.mask_invalid_data(ds)
            ds.attrs.update(attrs)

    print(
        "start: "
        + str(ds.time.values[0])
        + ", end: "
        + str(ds.time.values[-1])
        + ", time dim length: "
        + str(len(ds.time.values))
    )
    print("calculating vegetation indice")
    if collection == "c3":
        vegIndex = (ds.nbart_nir - ds.nbart_red) / (ds.nbart_nir + ds.nbart_red)
    if collection == "c2":
        vegIndex = (ds.nir - ds.red) / (ds.nir + ds.red)

    vegIndex = vegIndex.mean("time").rename("ndvi_mean")

    # get the bounding coords of the input ds to help with indexing the climatology
    xmin, xmax = vegIndex.x.values[0], vegIndex.x.values[-1]
    ymin, ymax = vegIndex.y.values[0], vegIndex.y.values[-1]
    x_slice = [i for i in range(int(xmin), int(xmax + 30), 30)]
    y_slice = [i for i in range(int(ymin), int(ymax - 30), -30)]

    # index the climatology dataset to the location of our AOI
    climatology_mean = (
        xr.open_rasterio(
            "/g/data/r78/cb3058/dea-notebooks/vegetation_anomalies/results/NSW_NDVI_Climatologies_mean/mosaics/ndvi_clim_mean_"
            + season
            + "_nsw.tif"
        )
        .sel(x=x_slice, y=y_slice, method="nearest")
        .chunk(chunks=dask_chunks)
        .squeeze()
    )

    climatology_std = (
        xr.open_rasterio(
            "/g/data1a/r78/cb3058/dea-notebooks/vegetation_anomalies/results/NSW_NDVI_Climatologies_std/mosaics/ndvi_clim_std_"
            + season
            + "_nsw.tif"
        )
        .sel(x=x_slice, y=y_slice, method="nearest")
        .chunk(chunks=dask_chunks)
        .squeeze()
    )

    # test if the arrays match before we calculate the anomalies
    np.testing.assert_allclose(
        vegIndex.x.values,
        climatology_mean.x.values,
        err_msg="The X coordinates on the AOI dataset do not match "
        "the X coordinates on the climatology mean dataset. "
        "You're AOI may be beyond the extent of the pre-computed "
        "climatology dataset.",
    )

    np.testing.assert_allclose(
        vegIndex.y.values,
        climatology_mean.y.values,
        err_msg="The Y coordinates on the AOI dataset do not match "
        "the Y coordinates on the climatology mean dataset. "
        "You're AOI may be beyond the extent of the pre-computed "
        "climatology dataset.",
    )

    np.testing.assert_allclose(
        vegIndex.x.values,
        climatology_std.x.values,
        err_msg="The X coordinates on the AOI dataset do not match "
        "the X coordinates on the climatology std dev dataset. "
        "You're AOI may be beyond the extent of the pre-computed "
        "climatology dataset.",
    )

    np.testing.assert_allclose(
        vegIndex.y.values,
        climatology_std.y.values,
        err_msg="The Y coordinates on the AOI dataset do not match "
        "the Y coordinates on the climatology std dev dataset. "
        "You're AOI may be beyond the extent of the pre-computed "
        "climatology dataset.",
    )

    print("calculating anomalies")
    # calculate standardised anomalies
    anomalies = xr.apply_ufunc(
        lambda x, m, s: (x - m) / s,
        vegIndex,
        climatology_mean,
        climatology_std,
        output_dtypes=[np.float32],
        dask="parallelized",
    )

    # add back metadata
    anomalies = anomalies.rename("std_anomalies").to_dataset()
    anomalies.attrs = ds.attrs
    anomalies.attrs["units"] = 1

    return anomalies