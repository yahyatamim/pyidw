import unittest

class MultiplicationTestCase(unittest.TestCase):

    def test_zero(self):
        
        self.assertEqual(1, 1) # dummy test class


from pyidw.idw import regression_idw_interpolation

regression_idw_interpolation(
    input_point_shapefile='Lowest_temperature.shp',
    input_raster_file='Elevation.tif',
    extent_shapefile='BGD_adm0.shp',
    column_name='Tnn_median',
    power=2, 
    polynomial_degree=0,
    search_radious=4)
