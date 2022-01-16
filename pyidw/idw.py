import numpy as np
from math import sqrt, floor, ceil
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.plot import show
from rasterio.crs import CRS
from rasterio.transform import from_bounds
import fiona
from rasterio.enums import Resampling
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut
from matplotlib import colors


def show_map(input_raster='', colormap='coolwarm', image_size=1.5):
    image_data = rasterio.open(input_raster)
    my_matrix = image_data.read(1)
    my_matrix = np.ma.masked_where(my_matrix == 32767, my_matrix)
    fig, ax = plt.subplots()
    image_hidden = ax.imshow(my_matrix, cmap=colormap)
    plt.close()

    fig, ax = plt.subplots()
    width = fig.get_size_inches()[0] * image_size
    height = fig.get_size_inches()[1] * image_size
    fig.set_size_inches(w=width, h=height)
    image = show(image_data, cmap=colormap, ax=ax)
    fig.colorbar(image_hidden, ax=ax, pad=0.02)
    plt.show()


#################################################


def crop_resize(input_raster_filename='',
                extent_shapefile_name='',
                max_height_or_width=250):
    # Here, co-variable raster file (elevation in this case) is croped and resized using rasterio.
    BD = gpd.read_file(extent_shapefile_name)
    elevation = rasterio.open(input_raster_filename)

    # Using mask method from rasterio.mask to clip study area from larger elevation file.
    croped_data, croped_transform = mask(dataset=elevation,
                                         shapes=BD.geometry,
                                         crop=True,
                                         all_touched=True)
    croped_meta = elevation.meta
    croped_meta.update({
        'height': croped_data.shape[-2],
        'width': croped_data.shape[-1],
        'transform': croped_transform
    })

    croped_filename = input_raster_filename.rsplit('.', 1)[0] + '_croped.tif'
    with rasterio.open(croped_filename, 'w', **croped_meta) as croped_file:
        croped_file.write(croped_data)  # Save the croped file as croped_elevation.tif to working directory.

    # Calculate resampling factor for resizing the elevation file, this is done to reduce calculation time.
    # Here 250 is choosed for optimal result, it can be more or less depending on users desire.
    # max_height_or_width = 250
    resampling_factor = max_height_or_width / max(rasterio.open(croped_filename).shape)

    # Reshape/resize the croped elevation file and save it to working directory.
    with rasterio.open(croped_filename, 'r') as croped_elevation:

        resampled_elevation = croped_elevation.read(
            out_shape=(croped_elevation.count,
                       int(croped_elevation.height * resampling_factor),
                       int(croped_elevation.width * resampling_factor)),
            resampling=Resampling.bilinear)

        resampled_transform = croped_elevation.transform * croped_elevation.transform.scale(
            croped_elevation.width / resampled_elevation.shape[-1],
            croped_elevation.height / resampled_elevation.shape[-2])

        resampled_meta = croped_elevation.meta
        resampled_meta.update({
            'height': resampled_elevation.shape[-2],
            'width': resampled_elevation.shape[-1],
            'dtype': np.float64,
            'transform': resampled_transform
        })

        resampled_filename = input_raster_filename.rsplit(
            '.', 1)[0] + '_resized.tif'
        with rasterio.open(resampled_filename, 'w', **resampled_meta) as resampled_file:
            resampled_file.write(resampled_elevation )  # Save the resized file as resampled_elevation.tif in working directory.


#################################################


def blank_raster(extent_shapefile=''):
    calculationExtent = gpd.read_file(extent_shapefile)

    minX = floor(calculationExtent.bounds.minx)
    minY = floor(calculationExtent.bounds.miny)
    maxX = ceil(calculationExtent.bounds.maxx)
    maxY = ceil(calculationExtent.bounds.maxy)
    longRange = sqrt((minX - maxX)**2)
    latRange = sqrt((minY - maxY)**2)

    gridWidth = 400
    pixelPD = (gridWidth / longRange)  # Pixel Per Degree
    gridHeight = floor(pixelPD * latRange)
    BlankGrid = np.ones([gridHeight, gridWidth])

    blank_filename = extent_shapefile.rsplit('.', 1)[0] + '_blank.tif'

    with rasterio.open(
            blank_filename,
            "w",
            driver='GTiff',
            height=BlankGrid.shape[0],
            width=BlankGrid.shape[1],
            count=1,
            dtype=BlankGrid.dtype,  #BlankGrid.dtype, np.float32, np.int16
            crs=CRS.from_string(calculationExtent.crs.srs),
            transform=from_bounds(minX, minY, maxX, maxY, BlankGrid.shape[1], BlankGrid.shape[0]),
            nodata=32767) as dst:
        dst.write(BlankGrid, 1)


#################################################


# def standard_idw(lon, lat, elev, longs, lats, elevs, d_values, id_power, p_degree, s_radious):
def standard_idw(lon, lat, longs, lats, d_values, id_power, s_radious):
    """regression_idw is responsible for mathmatic calculation of idw interpolation with regression as a covariable."""
    calc_arr = np.zeros(shape=( len(longs), 6))  # create an empty array shape of (total no. of observation * 6)
    calc_arr[:, 0] = longs  # First column will be Longitude of known data points.
    calc_arr[:, 1] = lats  # Second column will be Latitude of known data points.
    #     calc_arr[:, 2] = elevs    # Third column will be Elevation of known data points.
    calc_arr[:, 3] = d_values  # Fourth column will be Observed data value of known data points.

    # Fifth column is weight value from idw formula " w = 1 / (d(x, x_i)^power + 1)"
    # >> constant 1 is to prevent int divide by zero when distance is zero.
    calc_arr[:, 4] = 1 / (np.sqrt((calc_arr[:, 0] - lon)**2 + (calc_arr[:, 1] - lat)**2)**id_power + 1)

    # Sort the array in ascendin order based on column_5 (weight) "np.argsort(calc_arr[:,4])"
    # and exclude all the rows outside of search radious "[ - s_radious :, :]"
    calc_arr = calc_arr[np.argsort(calc_arr[:, 4])][-s_radious:, :]

    # Sixth column is multiplicative product of inverse distant weight and actual value.
    calc_arr[:, 5] = calc_arr[:, 3] * calc_arr[:, 4]
    # Divide sum of weighted value vy sum of weights to get IDW interpolation.
    idw = calc_arr[:, 5].sum() / calc_arr[:, 4].sum()
    return idw


#################################################


def idw_interpolation(input_point_shapefile='',
                      extent_shapefile='',
                      column_name='',
                      power=2,
                      search_radious=4,
                      output_resolution=250):
    blank_raster(extent_shapefile)
    
    blank_filename = extent_shapefile.rsplit('.', 1)[0] + '_blank.tif'
    crop_resize(input_raster_filename=blank_filename,
                extent_shapefile_name=extent_shapefile,
                max_height_or_width=output_resolution)
    
    resized_raster_name = blank_filename.rsplit('.', 1)[0] + '_resized.tif'
    #     baseRasterFile = rasterio.open(resized_raster_name) # baseRasterFile stands for resampled elevation.

    with rasterio.open(resized_raster_name) as baseRasterFile:
        inputPoints = gpd.read_file(input_point_shapefile)
        # obser_df stands for observation_dataframe, lat, lon, data_value for each station will be stored here.
        obser_df = pd.DataFrame()
        obser_df['station_name'] = inputPoints.iloc[:, 0]

        # create two list of indexes of station longitude, latitude in elevation raster file.
        lons, lats = baseRasterFile.index(
            [lon for lon in inputPoints.geometry.x],
            [lat for lat in inputPoints.geometry.y])
        obser_df['lon_index'] = lons
        obser_df['lat_index'] = lats
        obser_df['data_value'] = inputPoints[column_name]

        idw_array = baseRasterFile.read(1)
        for x in range(baseRasterFile.height):
            for y in range(baseRasterFile.width):
                if baseRasterFile.read(1)[x][y] == 32767:
                    continue
                else:
                    idw_array[x][y] = standard_idw(
                        lon=x,
                        lat=y,
                        longs=obser_df.lon_index,
                        lats=obser_df.lat_index,
                        d_values=obser_df.data_value,
                        id_power=power,
                        s_radious=search_radious)

        output_filename = input_point_shapefile.rsplit('.', 1)[0] + '_idw.tif'
        with rasterio.open(output_filename, 'w', **baseRasterFile.meta) as std_idw:
            std_idw.write(idw_array, 1)

        show_map(output_filename)


#################################################


def accuracy_standard_idw(input_point_shapefile='',
                          extent_shapefile='',
                          column_name='',
                          power=2,
                          search_radious=4,
                          output_resolution=250):
    blank_raster(extent_shapefile)
    
    blank_filename = extent_shapefile.rsplit('.', 1)[0] + '_blank.tif'
    crop_resize(input_raster_filename=blank_filename,
                extent_shapefile_name=extent_shapefile,
                max_height_or_width=output_resolution)
    
    resized_raster_name = blank_filename.rsplit('.', 1)[0] + '_resized.tif'

    with rasterio.open(resized_raster_name) as baseRasterFile:
        inputPoints = gpd.read_file(input_point_shapefile)
        # obser_df stands for observation_dataframe, lat, lon, data_value for each station will be stored here.
        obser_df = pd.DataFrame()
        obser_df['station_name'] = inputPoints.iloc[:, 0]

        # create two list of indexes of station longitude, latitude in elevation raster file.
        lons, lats = baseRasterFile.index(
            [lon for lon in inputPoints.geometry.x],
            [lat for lat in inputPoints.geometry.y])
        obser_df['lon_index'] = lons
        obser_df['lat_index'] = lats
        obser_df['data_value'] = inputPoints[column_name]
        obser_df['predicted'] = 0.0

        cv = LeaveOneOut()
        for train_ix, test_ix in cv.split(obser_df):
            test_point = obser_df.iloc[test_ix[0]]
            train_df = obser_df.iloc[train_ix]

            obser_df.loc[test_ix[0], 'predicted'] = standard_idw(
                lon=test_point.lon_index,
                lat=test_point.lon_index,
                longs=train_df.lon_index,
                lats=train_df.lat_index,
                d_values=train_df.data_value,
                id_power=power,
                s_radious=search_radious)
        return obser_df.data_value.to_list(), obser_df.predicted.to_list()


#################################################


def regression_idw(lon, lat, elev, longs, lats, elevs, d_values, id_power,
                   p_degree, s_radious, x_max, x_min):
    """regression_idw is responsible for mathmatic calculation of idw interpolation with regression as a covariable."""
    calc_arr = np.zeros(shape=(len(longs), 6))  # create an empty array shape of (total no. of observation * 6)
    calc_arr[:, 0] = longs  # First column will be Longitude of known data points.
    calc_arr[:, 1] = lats  # Second column will be Latitude of known data points.
    calc_arr[:, 2] = elevs  # Third column will be Elevation of known data points.
    calc_arr[:, 3] = d_values  # Fourth column will be Observed data value of known data points.

    # Fifth column is weight value from idw formula " w = 1 / (d(x, x_i)^power + 1)"
    # >> constant 1 is to prevent int divide by zero when distance is zero.
    calc_arr[:, 4] = 1 / (np.sqrt((calc_arr[:, 0] - lon)**2 + (calc_arr[:, 1] - lat)**2)**id_power + 1)

    # Sort the array in ascendin order based on column_5 (weight) "np.argsort(calc_arr[:,4])"
    # and exclude all the rows outside of search radious "[ - s_radious :, :]"
    calc_arr = calc_arr[np.argsort(calc_arr[:, 4])][-s_radious:, :]

    # Sixth column is multiplicative product of inverse distant weight and actual value.
    calc_arr[:, 5] = calc_arr[:, 3] * calc_arr[:, 4]
    # Divide sum of weighted value vy sum of weights to get IDW interpolation.
    idw = calc_arr[:, 5].sum() / calc_arr[:, 4].sum()

    # Create polynomial regression equation where independent variable is elevation and dependent variable is data_value.
    # Then, calculate R_squared value for just fitted polynomial equation.
    poly_reg = np.poly1d(np.polyfit(x=calc_arr[:, 2], y=calc_arr[:, 3], deg=p_degree))
    r_squared = r2_score(calc_arr[:, 3], poly_reg(calc_arr[:, 2]))

    regression_idw_combined = (1 - r_squared) * idw + r_squared * poly_reg(elev)
    if regression_idw_combined >= x_min and regression_idw_combined <= x_max:
        return regression_idw_combined
    elif regression_idw_combined < x_min:
        return x_min
    elif regression_idw_combined > x_max:
        return x_max


#################################################


def regression_idw_interpolation(input_point_shapefile='',
                                 input_raster_file='',
                                 extent_shapefile='',
                                 column_name='',
                                 power=2,
                                 polynomial_degree=1,
                                 search_radious=4,
                                 output_resolution=250):

    crop_resize(input_raster_filename=input_raster_file,
                extent_shapefile_name=extent_shapefile,
                max_height_or_width=output_resolution)

    metStat = gpd.read_file(input_point_shapefile)  # metStat stands for meteoriological stations.

    resampled_filename = input_raster_file.rsplit('.', 1)[0] + '_resized.tif'

    with rasterio.open(resampled_filename) as re_elevation:
        # obser_df stands for observation_dataframe, lat, lon, elevation, data_value for each station will be stored here.
        obser_df = pd.DataFrame()
        obser_df['station_name'] = metStat.iloc[:, 0]

        # create two list of indexes of station longitude, latitude in elevation raster file.
        lons, lats = re_elevation.index([lon for lon in metStat.geometry.x],
                                        [lat for lat in metStat.geometry.y])
        obser_df['lon_index'] = lons
        obser_df['lat_index'] = lats
        obser_df['elevation'] = re_elevation.read(1)[lons, lats]  # read elevation data for each station.
        obser_df['data_value'] = metStat[column_name]
        obser_df['predicted'] = 0.0
        upper_range = obser_df["data_value"].max() + obser_df["data_value"].std()
        lower_range = obser_df["data_value"].min() - obser_df["data_value"].std()

        regression_idw_array = re_elevation.read(1)
        for x in range(re_elevation.height):
            for y in range(re_elevation.width):
                if re_elevation.read(1)[x][y] == 32767:
                    continue
                else:
                    regression_idw_array[x][y] = regression_idw(
                        lon=x,
                        lat=y,
                        elev=re_elevation.read(1)[x][y],
                        longs=obser_df.lon_index,
                        lats=obser_df.lat_index,
                        elevs=obser_df.elevation,
                        d_values=obser_df.data_value,
                        id_power=power,
                        p_degree=polynomial_degree,
                        s_radious=search_radious,
                        x_max=upper_range,
                        x_min=lower_range)

        output_filename = input_point_shapefile.rsplit('.', 1)[0] + '_regression_idw.tif'
        with rasterio.open(output_filename, 'w', **re_elevation.meta) as reg_idw:
            reg_idw.write(regression_idw_array, 1)

        show_map(output_filename)


#################################################


def accuracy_regression_idw(input_point_shapefile='',
                            input_raster_file='',
                            extent_shapefile='',
                            column_name='',
                            power=2,
                            polynomial_degree=1,
                            search_radious=4,
                            output_resolution=250):

    crop_resize(input_raster_filename=input_raster_file,
                extent_shapefile_name=extent_shapefile,
                max_height_or_width=output_resolution)

    metStat = gpd.read_file(input_point_shapefile)  # metStat stands for meteoriological stations.

    resampled_filename = input_raster_file.rsplit('.', 1)[0] + '_resized.tif'

    with rasterio.open(resampled_filename) as re_elevation:
        # obser_df stands for observation_dataframe, lat, lon, elevation, data_value for each station will be stored here.
        obser_df = pd.DataFrame()
        obser_df['station_name'] = metStat.iloc[:, 0]

        # create two list of indexes of station longitude, latitude in elevation raster file.
        lons, lats = re_elevation.index([lon for lon in metStat.geometry.x],
                                        [lat for lat in metStat.geometry.y])
        obser_df['lon_index'] = lons
        obser_df['lat_index'] = lats
        obser_df['elevation'] = re_elevation.read(1)[
            lons, lats]  # read elevation data for each station.
        obser_df['data_value'] = metStat[column_name]
        obser_df['predicted'] = 0.0
        upper_range = obser_df["data_value"].max() + obser_df["data_value"].std()
        lower_range = obser_df["data_value"].min() - obser_df["data_value"].std()

        cv = LeaveOneOut()
        for train_ix, test_ix in cv.split(obser_df):
            test_point = obser_df.iloc[test_ix[0]]
            train_df = obser_df.iloc[train_ix]

            obser_df.loc[test_ix[0], 'predicted'] = regression_idw(
                lon=test_point.lon_index,
                lat=test_point.lon_index,
                elev=test_point.elevation,
                longs=train_df.lon_index,
                lats=train_df.lat_index,
                elevs=train_df.elevation,
                d_values=train_df.data_value,
                id_power=power,
                p_degree=polynomial_degree,
                s_radious=search_radious,
                x_max=upper_range,
                x_min=lower_range)

        return obser_df.data_value.to_list(), obser_df.predicted.to_list()
