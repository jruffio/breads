import breads.utils
import astropy.coordinates, astropy.units as u

def test_propagate_coordinates_at_epoch():
    result = breads.utils.propagate_coordinates_at_epoch('HD 19467', '2025-01-01')
    assert isinstance(result, astropy.coordinates.SkyCoord), "Function should return a valid SkyCoord object"

def test_relative_to_absolute_position():
    coord = breads.utils.companion_relative_to_absolute_position('HD 19467',
                                                     comp_name = 'HD 19467 B',
                                                     comp_sep = 1.6*u.arcsec,
                                                     comp_pa = 236*u.deg,
                                                     obs_date = '2023-01-20')
    expected = astropy.coordinates.SkyCoord(46.82695938, -13.76370023, unit='deg')
    assert coord.separation(expected) < 1*u.milliarcsecond, "Coordinates do not match expected absolute position"
