Content of utility folder
==============================

\*_roughbadpix.fits
--------------------
Utility file from compute_med_filt_badpix().

===  ===========  ===  ==========  =====  ============  =======
No.  Name         Ver  Type        Cards  Dimensions    Format
===  ===========  ===  ==========  =====  ============  =======
0    PRIMARY        1  PrimaryHDU    268  ()
1    BADPIXEL       1  ImageHDU       73  (2048, 2048)  float64
2    BREADS         1  ImageHDU       13  ()
===  ===========  ===  ==========  =====  ============  =======

\*_relcoords.fits
------------------
Utility file from compute_coordinates_arrays().

This file contains the relative coordinates of each pixel in the field of view with respect to the star.
The file is in FITS format and contains the following extensions:

Format of the utils file:

===  ============  ===  ==========  =====  ============  =======
No.  Name          Ver  Type        Cards  Dimensions    Format
===  ============  ===  ==========  =====  ============  =======
0    PRIMARY        1   PrimaryHDU    264  ()
1    WAVE           1   ImageHDU       73  (2048, 2048)  float32
2    X              1   ImageHDU        9  (2048, 2048)  float64
3    Y              1   ImageHDU        9  (2048, 2048)  float64
4    AREA2D         1   ImageHDU        9  (2048, 2048)  float32
5    TRACE_ID_MAP   1   ImageHDU        8  (2048, 2048)  float64
6    BREADS         1   ImageHDU       11  ()
===  ============  ===  ==========  =====  ============  =======

\*_starspec_contnorm.fits or \*_starspec_contnorm_3Dspline.fits
-----------------------------
Utility file from compute_starspectrum_contnorm().

===  =============  ===  ==========  =====  ============  =======
No.  Name           Ver  Type        Cards  Dimensions    Format
===  =============  ===  ==========  =====  ============  =======
0    PRIMARY          1  PrimaryHDU    268  ()
1    WAVE             1  ImageHDU       72  (2629,)       float64
2    COM_FLUXES       1  ImageHDU        7  (2629,)       float64
3    COM_ERRORS       1  ImageHDU        7  (2629,)       float64
4    SPLINE_CONT0     1  ImageHDU        8  (2048, 2048)  float32
5    SPLINE_PARAS0    1  ImageHDU        8  (40, 2048)    float64
6    X_NODES          1  ImageHDU        7  (40,)         float64
7    CONT_NORM_IM     1  ImageHDU        8  (2048, 2048)  float32
8    BREADS           1  ImageHDU       18  ()
===  =============  ===  ==========  =====  ============  =======

compute_starspectrum_contnorm_3dspline():

===  =============  ===  ==========  =====  ==============  =======
No.  Name           Ver  Type        Cards  Dimensions      Format
===  =============  ===  ==========  =====  ==============  =======
0    PRIMARY          1  PrimaryHDU    268  ()
1    WAVE             1  ImageHDU       72  (2614,)         float64
2    COM_FLUXES       1  ImageHDU        7  (2614,)         float64
3    COM_ERRORS       1  ImageHDU        7  (2614,)         float64
4    SPLINE_CONT0     1  ImageHDU        8  (2048, 32768)   float32
5    SPLINE_PARAS0    1  ImageHDU        9  (10, 10, 5)     float32
6    WV_NODES         1  ImageHDU        7  (5,)            float64
7    X_NODES          1  ImageHDU        7  (10,)           float64
8    Y_NODES          1  ImageHDU        7  (10,)           float64
9    CONT_NORM_IM     1  ImageHDU        8  (2048, 32768)   float32
10   BREADS           1  ImageHDU       43  ()
===  =============  ===  ==========  =====  ==============  =======

\*_starspec_contnorm_combined_1dspline.fits
-----------------------------

\*_starsub.fits
-----------------------------
Utility file from compute_starsubtraction().

===  =============  ===  ==========  =====  ============  =======
No.  Name           Ver  Type        Cards  Dimensions    Format
===  =============  ===  ==========  =====  ============  =======
0    PRIMARY          1  PrimaryHDU    268  ()
1    IM_SUB           1  ImageHDU       73  (2048, 2048)  float64
2    IM               1  ImageHDU        8  (2048, 2048)  float32
3    STARMODEL        1  ImageHDU        8  (2048, 2048)  float32
4    BADPIX           1  ImageHDU        8  (2048, 2048)  float32
5    SPLINE_PARAS0    1  ImageHDU        8  (40, 2048)    float64
6    X_NODES          1  ImageHDU        7  (40,)         float64
7    BREADS           1  ImageHDU       21  ()
===  =============  ===  ==========  =====  ============  =======

\*_regwvs.fits and \*_regwvs_starsub.fits
-----------------------------
Utility file from compute_interpdata_regwvs().

===  ====================  ===  ==========  =====  ===============  =======
No.  Name                  Ver  Type        Cards  Dimensions       Format
===  ====================  ===  ==========  =====  ===============  =======
0    PRIMARY                 1  PrimaryHDU    268  ()
1    INTERP_DATA             1  ImageHDU       73  (2197, 2048)     float64
2    INTERP_ERR              1  ImageHDU        8  (2197, 2048)     float64
3    INTERP_X                1  ImageHDU        8  (2197, 2048)     float64
4    INTERP_Y                1  ImageHDU        8  (2197, 2048)     float64
5    INTERP_WAVE             1  ImageHDU        8  (2197, 2048)     float64
6    INTERP_BADPIX           1  ImageHDU        8  (2197, 2048)     float64
7    INTERP_AREA2D           1  ImageHDU        8  (2197, 2048)     float64
8    INTERP_LEFTNRIGHT       1  ImageHDU        9  (2197, 2048, 2)  float64
9    BREADS                  1  ImageHDU       21  ()
===  ====================  ===  ==========  =====  ===============  =======

\*_webbpsf.fits
-----------------------------

===  ========  ===  ==========  =====  =================  =======
No.  Name      Ver  Type        Cards  Dimensions         Format
===  ========  ===  ==========  =====  =================  =======
0    OVERSAMP    1  PrimaryHDU    109  (600, 600, 2197)   float64
1    PSFS        1  ImageHDU        9  (600, 600, 2197)   float64
2    EPSFS       1  ImageHDU        9  (600, 600, 2197)   float64
3    WAVE        1  ImageHDU        7  (2197,)            float64
4    X           1  ImageHDU        8  (600, 600)         float64
5    Y           1  ImageHDU        8  (600, 600)         float64
6    BREADS      1  ImageHDU       20  ()
===  ========  ===  ==========  =====  =================  =======



\*_fitspsf.fits
-----------------------------


\*_fitspsf_poly_centroid_IWA????_OWA????_stpsf.txt
-----------------------------


\*_fitspsf_poly_fluxcal_IWA????_OWA????_stpsf.txt
-----------------------------

