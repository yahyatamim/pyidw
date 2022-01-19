# All in one IDW package for python
> **This is an example map created using pyidw library.**![idw interpolated map using pyidw](https://github.com/yahyatamim/pyidw/blob/master/images/output_map.png?raw=true)

## Features
 1. Simple IDW
 2. IDW with external raster (eg, elevation raster) covariable.
 3. Accuracy Score.
 4. Builtin raster visualisation with co-ordinate and colour bar.

## Why pyidw ?

Inverse distance weighted interpolation is one of the simplest geospatial interpolations available in GIS. Although it is easy to produce an idw raster using conventional desktop GIS software (eg. ArcGIS, QGIS). It was never straightforward to create such a beautiful map image using python. This is why I have created the pyidw library where you can create beautiful idw maps of your desired location using your favourite programming language 🐍

pyidw also incorporates a clever technique to use additional raster data as a covariable using simple linear regression. For example, if you are working with temperature data, it is widely known that temperature is inversely proportional to elevation, the higher the elevation, the lower the temperature is. With pyidw, you can easily add elevation data with traditional idw calculation and improve your interpolation accuracy.

## Installation
**pyidw** library can be installed using simple `pip install pyidw`. However, if you are facing trouble installing **pyidw** on your windows machine, please try the commands below on the windows command line. 

    pip install wheel
    pip install pipwin
    pipwin refresh
    pipwin install numpy
    pipwin install pandas
    pipwin install shapely
    pipwin install gdal
    pipwin install fiona
    pipwin install pyproj
    pipwin install six
    pipwin install rtree
    pipwin install geopandas
    pipwin install rasterio
    pip install pyidw
    pipwin refresh

## Example
If you are convinced enough to give **pyidw** a try, here is a simple tutorial for you. You should first download the pyidw_example.zip file. This zip file contains four files, 
- ***pyidw_tutorial.ipynb***
- ***Bangladesh_Temperature.shp***
- ***Bangladesh_Border.shp***
- ***Bangladesh_Elevation.tif***

The ***pyidw_tutorial.ipynb*** file is a jupyter notebook file which you could try to run and then modify with your data. ***Bangladesh_Temperature.shp***  file is an ESRI point shapefile which contains maximum and minimum temperture value for 34 weather stations all over Bangladesh. It's attribute table looks something like this.

Station\_Name|Station\_ID|Latitude|Longitude|Max\_Temp|Min\_Temp
-------------|-----------|--------|---------|---------|---------
BARISAL      |BGM00041950|22.75   |90.37    |36.75    |9.60     
BHOLA        |41951099999|22.68   |90.65    |35.62    |10.19    
BOGRA        |BGM00041883|24.85   |89.37    |38.62    |8.29     
CHANDPUR     |41941099999|23.27   |90.70    |35.87    |11.28    
CHITTAGONG   |BGM00041978|22.25   |91.81    |36.92    |11.24    
CHUADANGA    |41926099999|23.65   |88.82    |37.84    |8.59     
COMILLA      |41933099999|23.43   |91.18    |35.41    |10.35    
COXS BAZAR   |BGM00041992|21.45   |91.96    |37.11    |11.51    

For those who are not familiar wih shapefile, Every shapefile consists of seven different file with same name but seven different file extensions. Namely **.cpg .dbf .prj .sbn .sbx .shp** and **.shx**. If any of these file is missing then shapefile system won't work properly. Note that **Max_Temp** and **Min_Temp** column, we will use this value latter when creating IDW interpolated maps.

The ***Bangladesh_Border.shp*** is an ESRI polygon shapefile which covers all the area of country Bangladesh. We will use this shapefile to define the calculation extent for IDW interpolation. And finally the ***Bangladesh_Elevation.tif*** file which is a raster file containing elevation information in meter, We don't need this file for standard IDW interpolation but with regression_idw, we will use this file as an external covariable. All the files and their spatial dimension is shown below.

![Images of input files with their spatial dimensions.](https://github.com/yahyatamim/pyidw/raw/master/images/point_extent_elevation.png)

If you have any questions or problems, feel free to contact me at: yahyatamim0@gmail.com
