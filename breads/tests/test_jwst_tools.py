import breads.jwst_tools

def test_visualize_fov():
    import astropy.units as u
    breads.jwst_tools.visualize_nrs_ifu_fov('GJ 504 b', 2.5*u.arcsec, 315*u.degree, v3pa=272, center_on='comp')

