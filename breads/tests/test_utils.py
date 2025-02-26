import breads.utils
import astropy.coordinates

def test_propagate_coordinates_at_epoch():
    result = breads.utils.propagate_coordinates_at_epoch('HD 19467', '2025-01-01')
    assert isinstance(result, astropy.coordinates.SkyCoord), "Function should return a valid SkyCoord object"
