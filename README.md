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

If you have any questions or facing any problems, feel free to contact me at: yahyatamim0@gmail.com
