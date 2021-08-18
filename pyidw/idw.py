import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.plot import show
import fiona
from rasterio.enums import Resampling
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from matplotlib import colors



#########################################



def regression_idw(lon, lat, elev, longs, lats, elevs, d_values, id_power, p_degree, s_radious):
    """regression_idw is responsible for mathmatic calculation of idw interpolation with regression as a covariable."""
    calc_arr = np.zeros(shape=(len(longs), 6)) # create an empty array shape of (total no. of observation * 6)
    calc_arr[:, 0] = longs    # First column will be Longitude of known data points.
    calc_arr[:, 1] = lats     # Second column will be Latitude of known data points.
    calc_arr[:, 2] = elevs    # Third column will be Elevation of known data points.
    calc_arr[:, 3] = d_values # Fourth column will be Observed data value of known data points.
    
    # Fifth column is weight value from idw formula " w = 1 / (d(x, x_i)^power + 1)" 
    # >> constant 1 is to prevent int divide by zero when distance is zero.
    calc_arr[:, 4] = 1 / (np.sqrt((calc_arr[:,0] - lon)**2 + (calc_arr[:,1] - lat)**2)**id_power + 1)

    # Sort the array in ascendin order based on column_5 (weight) "np.argsort(calc_arr[:,4])"
    # and exclude all the rows outside of search radious "[ - s_radious :, :]"
    calc_arr = calc_arr[np.argsort(calc_arr[:,4])][ - s_radious :, :]
    
    # Sixth column is multiplicative product of inverse distant weight and actual value. 
    calc_arr[:, 5] = calc_arr[:,3] * calc_arr[:,4]
    # Divide sum of weighted value vy sum of weights to get IDW interpolation.
    idw = calc_arr[:, 5].sum() / calc_arr[:, 4].sum() 
    
    # Create polynomial regression equation where independent variable is elevation and dependent variable is data_value.
    # Then, calculate R_squared value for just fitted polynomial equation.
    poly_reg = np.poly1d(np.polyfit(x=calc_arr[:, 2], y=calc_arr[:, 3], deg=p_degree))
    r_squared = r2_score(calc_arr[:, 3], poly_reg(calc_arr[:, 2]))
    
    regression_idw_combined = (1 - r_squared)*idw + r_squared*poly_reg(elev)
    return regression_idw_combined




########################################################



def crop_resize(input_raster_filename='', extent_shapefile_name='', max_height_or_width = 250):
    # Here, co-variable raster file (elevation in this case) is croped and resized using rasterio.
    BD = gpd.read_file(extent_shapefile_name)
    elevation = rasterio.open(input_raster_filename)

    # Using mask method from rasterio.mask to clip study area from larger elevation file.
    croped_data, croped_transform = mask(dataset=elevation, shapes=BD.geometry, crop=True)
    croped_meta = elevation.meta
    croped_meta.update({'height': croped_data.shape[-2],
                        'width': croped_data.shape[-1],
                        'transform': croped_transform})

    croped_filename = input_raster_filename.rsplit('.', 1)[0] + '_croped.tif'
    with rasterio.open(croped_filename, 'w', **croped_meta) as croped_file:
        croped_file.write(croped_data) # Save the croped file as croped_elevation.tif to working directory.

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
            resampling=Resampling.bilinear
        )

        resampled_transform = croped_elevation.transform * croped_elevation.transform.scale(
            croped_elevation.width / resampled_elevation.shape[-1],
            croped_elevation.height / resampled_elevation.shape[-2]
        )

        resampled_meta = croped_elevation.meta
        resampled_meta.update({'height': resampled_elevation.shape[-2],
                               'width': resampled_elevation.shape[-1],
                              'transform': resampled_transform})

        resampled_filename = input_raster_filename.rsplit('.', 1)[0] + '_resized.tif'
        with rasterio.open(resampled_filename, 'w', **resampled_meta) as resampled_file:
            resampled_file.write(resampled_elevation) # Save the resized file as resampled_elevation.tif in working directory.



######################################



def accuracy_score(obser_df, power, polynomial_degree, search_radious):
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
            s_radious=search_radious)




######################




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
    
    metStat = gpd.read_file(input_point_shapefile) # metStat stands for meteoriological stations.
    
    resampled_filename = input_raster_file.rsplit('.', 1)[0] + '_resized.tif'
    re_elevation = rasterio.open(resampled_filename) # re_elevation stands for resampled elevation.

    # obser_df stands for observation_dataframe, lat, lon, elevation, data_value for each station will be stored here.
    obser_df = pd.DataFrame()
    obser_df['station_name'] = metStat.StationNam

    # create two list of indexes of station longitude, latitude in elevation raster file.
    lons, lats = re_elevation.index([lon for lon in metStat.geometry.x], [lat for lat in metStat.geometry.y])
    obser_df['lon_index'] = lons
    obser_df['lat_index'] = lats
    obser_df['elevation'] = re_elevation.read(1)[lons, lats] # read elevation data for each station.
    obser_df['data_value'] = metStat[column_name]
    obser_df['predicted'] = 0.0

    regression_idw_array = re_elevation.read(1)

    for x in range(re_elevation.height):
        for y in range(re_elevation.width):
            if re_elevation.read(1)[x][y] == 32767:
                continue
            else:
                regression_idw_array[x][y] = regression_idw(
                    lon = x,
                    lat = y,
                    elev = re_elevation.read(1)[x][y],
                    longs = obser_df.lon_index,
                    lats = obser_df.lat_index,
                    elevs = obser_df.elevation,
                    d_values = obser_df.data_value,
                    id_power = power,
                    p_degree = polynomial_degree,
                    s_radious = search_radious
                )

    output_filename = input_point_shapefile.rsplit('.', 1)[0] + '_regression_idw.tif'
    with rasterio.open(output_filename, 'w', **re_elevation.meta) as reg_idw:
        reg_idw.write(regression_idw_array, 1)
        
    fig, ax = plt.subplots(figsize=(6,8))
    show(rasterio.open(output_filename), cmap='nipy_spectral', ax=ax)
    metStat.plot(ax=ax, marker='D')
    plt.show()
    
    accuracy_score(obser_df, power, polynomial_degree, search_radious)
    print('RMSE:', mean_squared_error(obser_df.data_value.to_list(), obser_df.predicted.to_list(), squared=False))
    print('MAPE:', mean_absolute_error(obser_df.data_value.to_list(), obser_df.predicted.to_list())*100)
