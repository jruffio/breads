import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import curve_fit
import os
from astropy.stats import sigma_clipped_stats as sigclip
from breads.utils import propagate_coordinates_at_epoch

def gauss1d(x, A, mu, sigma, baseline):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma) ** 2) + baseline

def check_band(band, channel_file, band_file):
    if len(band)!=2:
        return False

    num_band = band[0]
    letter_band = band[1]
    
    if band_file=='SHORT':
        letter_band_file = 'A'
    elif band_file=='MEDIUM':
        letter_band_file = 'B'
    elif band_file=='LONG':
        letter_band_file = 'C'
    else:
        print(f"ERROR : band file {band_file} not supported for psf fitting")
        return False

    if num_band in channel_file and letter_band == letter_band_file:
        return True
    else:
        return False


def trace_slice(thisslice, data, snum, basex, basey, nmed, verbose, everyn, ch4C=False):
    ysize, xsize = data.shape
    verbose = True
    everyn = 5
    y_start = 100
    y_end = 1000
    plot = False

    # Zero out everything outside the peak slice
    indx = np.where(snum == thisslice)
    data_slice = data * 0.
    data_slice[indx] = data[indx]
    xmin, xmax = np.min(basex[indx]), np.max(basex[indx])

    ###################
    # First pass for x locations in this slice;
    if verbose:
        print('First pass trace fitting')
    xcen_pass1 = np.zeros(ysize)
    for ii in range(0, ysize):
        ystart = max(0, int(ii - nmed / 2))
        ystop = min(ysize, ystart + nmed)
        cut = np.nanmedian(data_slice[ystart:ystop, :], axis=0)
        xcen_pass1[ii] = np.nanargmax(cut)

    # Clean up any bad values by looking for 3sigma outliers
    # and replacing them with the median value
    rms, med = np.nanstd(xcen_pass1), np.nanmedian(xcen_pass1)
    indx = np.where((xcen_pass1 < med - 3 * rms) | (xcen_pass1 > med + 3 * rms))
    xcen_pass1[indx] = med
    xwid_pass1 = np.ones(ysize)  # First pass width is 1 pixel

    ###################
    # Second pass for x locations along the trace within this slice
    if verbose:
        print('Second pass trace fitting')

    xcen_pass2 = np.empty(ysize)
    xcen_pass2[:] = np.nan
    xwid_pass2 = np.empty(ysize)
    xwid_pass2[:] = np.nan

    for ii in range(y_start, y_end, everyn):
        xtemp = np.arange(xmin, xmax, 1)
        ftemp = data_slice[ii, xtemp]

        xtemp = xtemp[np.isfinite(ftemp)]
        ftemp = ftemp[np.isfinite(ftemp)]

        # Initial guess at fit parameters
        p0 = [ftemp.max(), xcen_pass1[ii], xwid_pass1[ii], 0.]
        # Bounds for fit parameters
        bound_low = [0., xcen_pass1[ii] - 3 * rms, 0, -ftemp.max()]
        bound_hi = [10 * np.max(ftemp), xcen_pass1[ii] + 3 * rms, 10, ftemp.max()]
        # Do the fit
        try:
            popt, _ = curve_fit(gauss1d, xtemp, ftemp, p0=p0, bounds=(bound_low, bound_hi), method='trf')
        except:
            popt = p0
        xcen_pass2[ii] = popt[1]
        xwid_pass2[ii] = popt[2]

        if plot:
            plt.title("fit second pass")
            plt.scatter(xtemp, ftemp)
            xtemp_sur = np.linspace(np.min(xtemp), np.max(xtemp), 1000)
            plt.plot(xtemp_sur, gauss1d(xtemp_sur, *popt))
            plt.show()

    ###################
    # Third pass for x location; use a fixed profile width
    twidth = np.nanmedian(xwid_pass2)

    xamp_pass3 = np.empty(ysize)
    xamp_pass3[:] = np.nan
    xbase_pass3 = np.empty(ysize)
    xbase_pass3[:] = np.nan
    xcen_pass3 = np.empty(ysize)
    xcen_pass3[:] = np.nan

    if verbose:
        print('Third pass trace fitting, median trace width ', twidth, ' pixels')

    for ii in range(y_start, y_end, everyn):
        xtemp = np.arange(xmin, xmax, 1)
        ftemp = data_slice[ii, xtemp]

        xtemp = xtemp[np.isfinite(ftemp)]
        ftemp = ftemp[np.isfinite(ftemp)]

        # Initial guess at fit parameters
        p0 = [ftemp.max(), xcen_pass2[ii], twidth, 0.]
        # Bounds for fit parameters
        bound_low = [0., xcen_pass2[ii] - 3 * rms, twidth * 0.999, -ftemp.max()]
        bound_hi = [10 * np.max(ftemp), xcen_pass2[ii] + 3 * rms, twidth * 1.001, ftemp.max()]
        # Do the fit
        try:
            popt, _ = curve_fit(gauss1d, xtemp, ftemp, p0=p0, bounds=(bound_low, bound_hi), method='trf')
        except:
            popt = p0
        xamp_pass3[ii] = popt[0]
        xcen_pass3[ii] = popt[1]
        xbase_pass3[ii] = popt[3]
        if plot:
            plt.title("fit")
            plt.scatter(xtemp, ftemp)
            xtemp_sur = np.linspace(np.min(xtemp), np.max(xtemp), 1000)
            plt.plot(xtemp_sur, gauss1d(xtemp_sur, *popt))
            plt.show()

    # Clean up the fit to remove outliers
    thisbasey = basey[:, 0]
    qual = np.ones(ysize)

    # Low order polynomial fit to replace nans
    good = np.where(np.isfinite(xcen_pass3) == True)
    fit = np.polyfit(thisbasey[good], xcen_pass3[good], 2)
    model = np.polyval(fit, thisbasey)
    indx = np.where(np.isfinite(xcen_pass3) == False)
    qual[indx] = 0
    xcen_pass3[indx] = model[indx]

    # Standard deviation clipping to replace outliers
    indx = np.where(np.abs(xcen_pass3 - model) > 3 * np.nanstd(xcen_pass3[good] - model[good]))
    qual[indx] = 0
    good = (np.where(qual == 1))[0]

    # Another fit to find lesser outliers using sigma-clipped RMS
    fit = np.polyfit(thisbasey[good], xcen_pass3[good], 2)
    model = np.polyval(fit, thisbasey)
    indx = np.where(np.abs(xcen_pass3 - model) > 3 * (sigclip(xcen_pass3[good] - model[good])[2]))
    qual[indx] = 0

    # Find any nan values and set them to bad quality
    indx = np.where(np.isfinite(xcen_pass3) != True)
    qual[indx] = 0

    # If we're in Ch4C, ignore everything with y< 500
    # but set a point at y=0 to the median of all good points
    # to help ensure the polynomial doesn't go crazy
    if ch4C:
        indx = np.where(thisbasey < 500)
        qual[indx] = 0
        good = (np.where(qual == 1))[0]
        xcen_pass3[0] = np.median(xcen_pass3[good])
        qual[0] = 1

    # Final model fit
    good = (np.where(qual == 1))[0]
    fit = np.polyfit(thisbasey[good], xcen_pass3[good], 2)
    model = np.polyval(fit, thisbasey)

    return xcen_pass2, xcen_pass3, model


def fit_trace(hdu, band, crds_dir, nmed=10, everyn=5, verbose=False):
    # Set a special flag if we're in Ch4C
    ch4C = False
    plot = False

    # Compute an array of x,y pixel locations for the entire detector
    basex, basey = np.meshgrid(np.arange(1032), np.arange(1024))

    ablgrid = xytoabl(basex.ravel(), basey.ravel(), crds_dir, band)
    snum = np.reshape(ablgrid['slicenum'], basex.shape)

    # Which channel & band is this data?
    chan_file = hdu[0].header['CHANNEL']
    band_file = hdu[0].header['BAND']

    if not check_band(band, chan_file, band_file):
        return 0

    data = hdu[1].data

    ysize, xsize = data.shape

    # Determine how many slices and their numbers
    slices = np.unique(snum)[1:]  # Cut out 0th entry (null slice value)
    nslices = len(slices)

    # Zero out all data not in one of these slices (i.e., other half of detector)
    indx = np.where(snum < 0)
    data[indx] = 0.

    # Identify the peak slice using a cut along the central row of the detector
    # Median combine down nmed rows to add robustness against noise
    ystart = int(ysize / 2 - nmed / 2)
    ystop = ystart + nmed
    cut = np.nanmedian(data[ystart:ystop, :], axis=0)

    # Define sum in each slice
    slicecut = snum[int(ysize / 2), :]
    slicesum = np.zeros(nslices)

    # Need to sum the fluxes in each slice to avoid issues where centroid is on pixel boundary vs not
    # in different slices
    for ii in range(0, nslices):
        indx = np.where(slicecut == slices[ii])
        slicesum[ii] = np.nansum(cut[indx])

    # Which slice is the peak in?
    peakslice = slices[np.argmax(slicesum)]
    print(f"Peakslice ID is {peakslice}")

    # Define central beta of the slices
    slice_beta = np.zeros(nslices)
    for ii in range(0, nslices):
        # Use any pixel in the slice to get the beta
        indx = np.where(snum == slices[ii])
        slice_beta[ii] = (xytoabl(basex[indx][0], basey[indx][0], crds_dir, band))['beta']
    # Define absolute beta ranges (top of top slice to bottom of bottom slice)
    dbeta = np.abs(slice_beta[0] - slice_beta[1])
    minbeta = np.min(slice_beta) - dbeta / 2.
    maxbeta = np.max(slice_beta) + dbeta / 2.

    # Get the x trace in the central slice

    xtrace_mid_pass2, xtrace_mid_pass3, xtrace_mid_poly = trace_slice(peakslice, data, snum, basex, basey, nmed,
                                                                      verbose=verbose, everyn=everyn, ch4C=ch4C)
    alpha_mid = (xytoabl(xtrace_mid_poly, basey[:, 0], crds_dir, band))['alpha']
    # Final alpha value is the median alpha along the central trace
    # For most bands use entire Y range; for 4C use only rows > 700
    # Ensure we don't use anything that centroided between slices

    good = np.where(alpha_mid > -100)
    alpha = np.nanmedian(alpha_mid[good])

    # We can't simply compare fluxes across slices at a given Y, because
    # there are wavelength offsets and we'd thus see spectral changes
    # not spatial changes in the flux.  Therefore sample at a grid
    # of wavelengths instead

    ftemp = np.zeros(nslices)
    bcen_vec = np.zeros(ysize)
    bwid_vec = np.zeros(ysize)

    # Set upper bound on PSF width (sigma) for beta fit based on the band
    psfmax = 0.5  # default
    if ((band == '1A') or (band == '1B') or (band == '1C')):
        psfmax = 0.3
    if ((band == '2A') or (band == '2B') or (band == '2C')):
        psfmax = 0.4
    if ((band == '3A') or (band == '3B') or (band == '3C')):
        psfmax = 0.5
    if ((band == '4A') or (band == '4B') or (band == '4C')):
        psfmax = 0.6

    # NEW: ignore wavelength shifts and just do this up rows instead so we can median
    for ii in range(0, ysize, everyn):
        ftemp[:] = 0.
        ystart = ii
        ystop = ystart + nmed
        if (ystop > ysize):
            ystop = ysize
        cut = np.nanmedian(data[ystart:ystop, :], axis=0)
        for jj in range(0, nslices):
            # We can't trivially fit the trace in each slice either, because it's too faint to
            # see in most, and the width changes from slice to slice.
            # Therefore just sum in a wavelength box.
            indx = np.where(snum[ii, :] == slices[jj])
            ftemp[jj] = np.nansum(cut[indx])
        # Initial guess at fit parameters
        p0 = [ftemp.max(), slice_beta[np.argmax(ftemp)], psfmax / 2., 0.]
        # Bounds for fit parameters.  Assume center MUST be somewhere in field.
        bound_low = [0., minbeta, 0, -ftemp.max()]
        bound_hi = [10 * np.max(ftemp), maxbeta, psfmax, ftemp.max()]
        # Do the fit

        try:
            popt, _ = curve_fit(gauss1d, slice_beta, ftemp, p0=p0, bounds=(bound_low, bound_hi), method='trf')
            if plot:
                plt.title("BETA FIT")
                plt.scatter(slice_beta, ftemp)
                x_sur = np.linspace(np.nanmin(slice_beta), np.nanmax(slice_beta), 100)
                plt.plot(x_sur, gauss1d(x_sur, *popt))
                plt.show()
        except Exception as e:
            print("EXCEPTION BETA")
            print(e)
            popt = p0
        bcen_vec[ii] = popt[1]
        bwid_vec[ii] = popt[2]

    # Final beta is the median of the values measured at various wavelengths
    # Median all non-zero values (b/c of failure cases on IFU edge or skipping above)
    # In band 4C only use rows > 700

    good = np.where(bcen_vec != 0)
    beta = np.nanmedian(bcen_vec[good])

    # Convert final alpha,beta coordinates to v2,v3
    v2, v3 = abtov2v3(alpha, beta, crds_dir, band)
    # And convert to RA, DECabtov2v3
    ra, dec = jwst_v2v3toradec([v2], [v3], hdr=hdu[1].header)

    return alpha, beta, ra, dec  # xtrace_mid_pass2, xtrace_mid_pass3, xtrace_mid_poly


def abtov2v3(alin, bein, crds_dir, channel):
    # Split input channel into components, e.g.
    # if channel='1A' then ch=1 and sband=A

    alpha = np.array(alin) * 1.
    beta = np.array(bein) * 1.

    # Open relevant distortion file
    distfile = fits.open(get_fitsreffile(channel, crds_dir))

    # Read the distortion table
    convtable = distfile['albe_to_V2V3'].data
    # Determine which rows we need
    v2index = (np.where(convtable['Label'] == 'T_CH' + channel + '_V2'))[0][0]
    v3index = (np.where(convtable['Label'] == 'T_CH' + channel + '_V3'))[0][0]

    if (np.logical_or(v2index < 0, v3index < 0)):
        print('Bad channel specification!')

    conv_v2 = convtable[v2index]
    conv_v3 = convtable[v3index]

    # Apply transform to V2,V3
    v2 = conv_v2[1] + conv_v2[2] * alpha + conv_v2[3] * alpha * alpha + \
         conv_v2[4] * beta + conv_v2[5] * beta * alpha + conv_v2[6] * beta * alpha * alpha + \
         conv_v2[7] * beta * beta + conv_v2[8] * beta * beta * alpha + conv_v2[9] * beta * beta * alpha * alpha
    v3 = conv_v3[1] + conv_v3[2] * alpha + conv_v3[3] * alpha * alpha + \
         conv_v3[4] * beta + conv_v3[5] * beta * alpha + conv_v3[6] * beta * alpha * alpha + \
         conv_v3[7] * beta * beta + conv_v3[8] * beta * beta * alpha + conv_v3[9] * beta * beta * alpha * alpha

    distfile.close()

    return v2, v3


def xytoabl(xin, yin, crds_dir, channel, **kwargs):
    # Split input channel into components, e.g.
    # if channel='1A' then ch=1 and sband=A
    ch = channel[0]
    sband = channel[1]


    trimx = np.array(xin)
    trimy = np.array(yin)

    # Ensure we're not using integer inputs
    # Also handle possible 1-element or multi-element input
    try:
        numpoints = len(xin)
        x = np.array(xin) * 1.0
        y = np.array(yin) * 1.0
    except:
        numpoints = 1
        x = np.array([xin]) * 1.0
        y = np.array([yin]) * 1.0

    # Open relevant distortion file
    distfile = fits.open(get_fitsreffile(channel, crds_dir))

    # Read global header
    hdr = distfile[0].header

    # Get beta zeropoint and spacing from header
    beta0 = hdr['B_ZERO' + ch]
    dbeta = hdr['B_DEL' + ch]

    # Alpha matrix
    d2c_alpha = distfile['Alpha_CH' + ch].data
    # Lambda matrix
    d2c_lambda = distfile['Lambda_CH' + ch].data
    # Slice map
    d2c_slice_all = distfile['Slice_Number'].data
    # Unless otherwise specified, use the 80% throughput slice map
    if 'mapplane' in kwargs:
        d2c_slice = d2c_slice_all[kwargs['mapplane'], :, :]
    else:
        d2c_slice = d2c_slice_all[7, :, :]

    # Define slice for these pixels
    slicenum = np.zeros(x.size, int)
    slicename = np.array(['JUNK' for i in range(0, x.size)])

    for i in range(0, x.size):
        slicenum[i] = int(d2c_slice[int(round(y[i])), int(round(x[i]))]) - int(ch) * 100
        slicename[i] = str(int(d2c_slice[int(round(y[i])), int(round(x[i]))])) + sband

    # Eliminate slice numbers on the wrong half of the detector
    bad = np.where((slicenum < 0) | (slicenum > 50))
    slicenum[bad] = -100
    slicename[bad] = 'NA'

    # Define index0 where the slice number is physical
    # (i.e., not between slices).  The [0] seems necessary to get
    # actual values rather than a single list object
    index0 = (np.where((slicenum > 0) & (slicenum < 50)))[0]
    nindex0 = len(index0)

    # Initialize a,b,l to -999.
    # (they will be set to something else if the pixel lands on a valid slice)
    al = np.zeros(x.size) - 999.
    be = np.zeros(x.size) - 999.
    lam = np.zeros(x.size) - 999.

    # Define beta for these pixels
    if (nindex0 > 0):
        be[index0] = beta0 + (slicenum[index0] - 1.) * dbeta

    # Get the alpha,lambda coefficients for all of the valid pixels
    alphacoeff = d2c_alpha[slicenum[index0] - 1]
    lamcoeff = d2c_lambda[slicenum[index0] - 1]
    # Build big matrices of the x,y inputs combined with the corresponding coefficients
    thealphamatrix = np.zeros([nindex0, 26])
    thelammatrix = np.zeros([nindex0, 26])
    # Python hates loops, so instead of looping over individual entries
    # loop over columns in the big matrices instead
    for i in range(0, 5):
        for j in range(0, 5):
            coind = 1 + (i * 5) + j
            thealphamatrix[:, coind] = alphacoeff.field(coind) * (
                    ((x[index0] - alphacoeff.field(0)) ** j) * (y[index0] ** i))
            thelammatrix[:, coind] = lamcoeff.field(coind) * (((x[index0] - lamcoeff.field(0)) ** j) * (y[index0] ** i))

    # Sum the contributions from each column in the big matrices
    al[index0] = np.sum(thealphamatrix, axis=1)
    lam[index0] = np.sum(thelammatrix, axis=1)

    distfile.close()

    # Return a dictionary of results
    values = dict()
    values['x'] = trimx
    values['y'] = trimy
    values['alpha'] = al
    values['beta'] = be
    values['lam'] = lam
    values['slicenum'] = slicenum
    values['slicename'] = slicename

    return values


def jwst_v2v3toradec(v2in, v3in, **kwargs):
    if ('hdr' in kwargs):
        hdr = kwargs['hdr']
        v2ref = hdr['V2_REF']
        v3ref = hdr['V3_REF']
        raref = hdr['RA_REF']
        decref = hdr['DEC_REF']
        rollref = hdr['ROLL_REF']
    elif ('v2ref' in kwargs):
        v2ref = kwargs['v2ref']
        v3ref = kwargs['v3ref']
        raref = kwargs['raref']
        decref = kwargs['decref']
        rollref = kwargs['rollref']
    else:
        print('Error: no reference values provided!')

    # Convert reference values to units of radians
    v2ref = v2ref / 3600. * np.pi / 180.
    v3ref = v3ref / 3600. * np.pi / 180.
    raref = raref * np.pi / 180.
    decref = decref * np.pi / 180.
    rollref = rollref * np.pi / 180.

    # Compute the JWST attitude matrix from the 5 attitude keywords
    attmat = jwst_attmatrix(v2ref, v3ref, raref, decref, rollref)

    # Number of input points
    v2 = np.array(v2in)
    v3 = np.array(v3in)
    npoints = len(v2)

    # Make empty vectors to hold the output ra,dec,NEWROLL
    ra = np.zeros(npoints)
    dec = np.zeros(npoints)

    # Loop over input points in the simplest way
    for i in range(0, npoints):
        # Compute the vector describing the input location
        invector = [np.cos(v2[i] / 3600. * np.pi / 180.) * np.cos(v3[i] / 3600. * np.pi / 180.),
                    np.sin(v2[i] / 3600. * np.pi / 180.) * np.cos(v3[i] / 3600. * np.pi / 180.),
                    np.sin(v3[i] / 3600. * np.pi / 180.)]

        # Compute the output vector (cos(RA)cos(dec),sin(RA)cos(dec),sin(dec))
        # by applying the attitude matrix
        outvector = np.matmul(invector, attmat)

        # Split the output vector into RA and DEC components and convert
        # back to degrees
        ra[i] = np.arctan2(outvector[1], outvector[0]) * 180. / np.pi

        # Ensure 0-360 degrees
        if (ra[i] < 0.):
            ra[i] = ra[i] + 360.

        dec[i] = np.arcsin(outvector[2]) * 180. / np.pi
    ra *= 3600 #degrees to arcsec
    dec *= 3600

    return ra, dec


def jwst_attmatrix(V2REF, V3REF, RAREF, DECREF, ROLLREF):
    thematrix = np.matmul(jwst_att1(V2REF, V3REF), jwst_att2(RAREF, DECREF, ROLLREF))

    return thematrix


# Construct the JWST M1 attitude matrix (V2 and V3 rotations)
# V2REF and V3REF should be in radians
def jwst_att1(V2REF, V3REF):
    # M1=  a00  a01  a02
    #      a10  a11  a12
    #      a20  a21  a22

    thematrix = np.zeros((3, 3))
    thematrix[0, 0] = np.cos(V2REF) * np.cos(V3REF)
    thematrix[1, 0] = np.sin(V2REF) * np.cos(V3REF)
    thematrix[2, 0] = np.sin(V3REF)
    thematrix[0, 1] = -np.sin(V2REF)
    thematrix[1, 1] = np.cos(V2REF)
    thematrix[2, 1] = 0.
    thematrix[0, 2] = -np.cos(V2REF) * np.sin(V3REF)
    thematrix[1, 2] = -np.sin(V2REF) * np.sin(V3REF)
    thematrix[2, 2] = np.cos(V3REF)

    return thematrix


#############################

# Construct the JWST M2 attitude matrix (RA,DEC,ROLL rotations)
# RAREF, DECREF, ROLLREF should be in radians
def jwst_att2(RAREF, DECREF, ROLLREF):
    # M2=  a00  a01  a02
    #      a10  a11  a12
    #      a20  a21  a22

    thematrix = np.zeros((3, 3))
    thematrix[0, 0] = np.cos(RAREF) * np.cos(DECREF)
    thematrix[1, 0] = -np.sin(RAREF) * np.cos(ROLLREF) + np.cos(RAREF) * np.sin(DECREF) * np.sin(ROLLREF)
    thematrix[2, 0] = -np.sin(RAREF) * np.sin(ROLLREF) - np.cos(RAREF) * np.sin(DECREF) * np.cos(ROLLREF)
    thematrix[0, 1] = np.sin(RAREF) * np.cos(DECREF)
    thematrix[1, 1] = np.cos(RAREF) * np.cos(ROLLREF) + np.sin(RAREF) * np.sin(DECREF) * np.sin(ROLLREF)
    thematrix[2, 1] = np.cos(RAREF) * np.sin(ROLLREF) - np.sin(RAREF) * np.sin(DECREF) * np.cos(ROLLREF)
    thematrix[0, 2] = np.sin(DECREF)
    thematrix[1, 2] = -np.cos(DECREF) * np.sin(ROLLREF)
    thematrix[2, 2] = np.cos(DECREF) * np.cos(ROLLREF)

    return thematrix


def get_fitsreffile(channel, crds_dir):
    if (channel == '1A'):
        file = 'MIRI_FM_MIRIFUSHORT_12SHORT_DISTORTION_flt8.fits'
    elif (channel == '1B'):
        file = 'MIRI_FM_MIRIFUSHORT_12MEDIUM_DISTORTION_flt8.fits'
    elif (channel == '1C'):
        file = 'MIRI_FM_MIRIFUSHORT_12LONG_DISTORTION_flt8.fits'
    elif (channel == '2A'):
        file = 'MIRI_FM_MIRIFUSHORT_12SHORT_DISTORTION_flt8.fits'
    elif (channel == '2B'):
        file = 'MIRI_FM_MIRIFUSHORT_12MEDIUM_DISTORTION_flt8.fits'
    elif (channel == '2C'):
        file = 'MIRI_FM_MIRIFUSHORT_12LONG_DISTORTION_flt8.fits'
    elif (channel == '3A'):
        file = 'MIRI_FM_MIRIFULONG_34SHORT_DISTORTION_flt8.fits'
    elif (channel == '3B'):
        file = 'MIRI_FM_MIRIFULONG_34MEDIUM_DISTORTION_flt8.fits'
    elif (channel == '3C'):
        file = 'MIRI_FM_MIRIFULONG_34LONG_DISTORTION_flt8.fits'
    elif (channel == '4A'):
        file = 'MIRI_FM_MIRIFULONG_34SHORT_DISTORTION_flt8.fits'
    elif (channel == '4B'):
        file = 'MIRI_FM_MIRIFULONG_34MEDIUM_DISTORTION_flt8.fits'
    elif (channel == '4C'):
        file = 'MIRI_FM_MIRIFULONG_34LONG_DISTORTION_flt8.fits'
    else:
        raise ValueError(f'Unknown Channel {channel}')

    # Try looking for the file in the expected location
    rootdir = os.path.join(crds_dir, "miri_coord")
    reffile = os.path.join(rootdir, file)

    if os.path.exists(reffile):
        #fits.open(reffile)
        return reffile
    else:
        raise ValueError(f'File {reffile} does not exist')

def run_centroid_fit(util_path, band='2A', targname=None):
    files = os.listdir(util_path)

    ra = []
    dec = []
    alpha, beta = [], []
    dra, ddec = [], []
    file_list = []

    for file in files:
        if file.endswith("cal.fits"):
            print(file)
            hdu = fits.open(os.path.join(util_path, file))
            chan_file = hdu[0].header['CHANNEL']
            band_file = hdu[0].header['BAND']
            if check_band(band, chan_file, band_file):
                if targname is None:
                    targname = hdu[0].header['TARGNAME']
                date = hdu[0].header["DATE-OBS"]
                alpha_k, beta_k, ra_k, dec_k = fit_trace(hdu, band, everyn=10)
                print("alpha", alpha)
                print("beta", beta)
                print("ra_centroid", ra_k)
                print("dec_centroid", dec_k)
                alpha.append(alpha_k)
                beta.append(beta_k)
                ra.append(ra_k)
                dec.append(dec_k)
                file_list.append(file)
                try:
                    host_coord = propagate_coordinates_at_epoch(targname, date, verbose=True)
                    host_ra = host_coord.ra.deg * 3600
                    host_dec = host_coord.dec.deg * 3600
                    dra.append(ra - host_ra)
                    ddec.append(dec - host_dec)
                except Exception as e:
                    print(e)
                    dra.append(np.nan)
                    ddec.append(np.nan)

    return np.nanmedian(np.array(ra)), np.nanmedian(np.array(dec)), dra, ddec, alpha, beta, file_list
