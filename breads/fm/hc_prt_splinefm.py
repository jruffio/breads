import numpy as np
from copy import copy
import yaml
import pandas as pd
from astropy import constants as const
from  scipy.interpolate import interp1d
from PyAstronomy import pyasl
import scipy.interpolate as interp
from breads.utils import broaden
from multiprocessing import Pool
from petitRADTRANS.physical_constants import r_jup
from petitRADTRANS.radtrans import Radtrans
# from breads.utils import LPFvsHPF

from breads.utils import get_spline_model

def pixgauss2d(p, shape, hdfactor=10, xhdgrid=None, yhdgrid=None):
    """
    2d gaussian model. Documentation to be completed. Also faint of t
    """
    A, xA, yA, w, bkg = p
    ny, nx = shape
    if xhdgrid is None or yhdgrid is None:
        xhdgrid, yhdgrid = np.meshgrid(np.arange(hdfactor * nx).astype(float) / hdfactor,
                                       np.arange(hdfactor * ny).astype(float) / hdfactor)
    else:
        hdfactor = xhdgrid.shape[0] // ny
    gaussA_hd = A / (2 * np.pi * w ** 2) * np.exp(
        -0.5 * ((xA - xhdgrid) ** 2 + (yA - yhdgrid) ** 2) / w ** 2)
    gaussA = np.nanmean(np.reshape(gaussA_hd, (ny, hdfactor, nx, hdfactor)), axis=(1, 3))
    return gaussA + bkg

def convolve_and_sample(wv_channels, sigmas, model_wvs, model_fluxes, channel_width=None, num_sigma=3):
    """
    Simulate the observations of a model. Convolves the model with a variable Gaussian LSF, sampled at each desired spectral channel.

    Args:
        wv_channels: the wavelengths desired (length of N_output)
        sigmas: the LSF gaussian stddev of each wv_channels in units of channels (length of N_output)
        model_wvs: the wavelengths of the model (length of N_model)
        model_fluxes: the fluxes of the model (length of N_model)
        channel_width: (optional) the full width of each wavelength channel in units of wavelengths (length of N_output)
        num_sigma (float): number of +/- sigmas to evaluate the LSF to. 

    Returns:
        output_model: the fluxes in each of the wavelength channels (length of N_output)
    """
    # create the wavelength grid for variable LSF convolution
    dwv = np.abs(wv_channels - np.roll(wv_channels, 1))
    dwv[0] = dwv[1]  # edge case
    sigmas_wvs = sigmas * dwv
    #filter_size_wv = int(np.ceil(np.max(sigmas_wvs))) * 6 # wavelength range to filter

    model_in_range = np.where((model_wvs >= np.min(wv_channels)) & (model_wvs < np.max(wv_channels)))
    dwv_model = np.abs(model_wvs[model_in_range] - np.roll(model_wvs[model_in_range], 1))
    dwv_model[0] = dwv_model[1]

    filter_size = int(np.ceil(np.max((2 * num_sigma * sigmas_wvs)/np.min(dwv_model)) ))
    filter_coords = np.linspace(-num_sigma, num_sigma, filter_size)
    filter_coords = np.tile(filter_coords, [wv_channels.shape[0], 1]) #  shape of (N_output, filter_size)
    filter_wv_coords = filter_coords * sigmas_wvs[:,None] + wv_channels[:,None] # model wavelengths we want
    lsf = np.exp(-filter_coords**2/2)/np.sqrt(2*np.pi)
    
    model_interp = interp.interp1d(model_wvs, model_fluxes, kind='cubic', bounds_error=False)
    filter_model = model_interp(filter_wv_coords)

    output_model = np.nansum(filter_model * lsf, axis=1)/np.sum(lsf, axis=1)
    
    return output_model

# pos: (x,y) or fiber, position of the companion
def hc_prt_splinefm(nonlin_paras, cubeobj, prt_base_atm=None, prt_atm_params=None, transmission=None, star_spectrum=None,boxw=1, psfw=1.2, nodes=20,
            badpixfraction=0.75, loc=None, fix_parameters=None, return_where_finite=False):
    """
    For high-contrast companions (planet + speckles).
    Generate forward model fitting the continuum with a spline.

    Args:
        nonlin_paras: Non-linear parameters of the model, which are first the parameters defining the atmopsheric grid
            (atm_grid). The following parameters are the spin (vsini), the radial velocity, and the position (if loc is
            not defined) of the planet in the FOV.
                [atm paras ....,vsini,rv,y,x] for 3d cubes (e.g. OSIRIS)
                [atm paras ....,vsini,rv,y] for 2d (e.g. KPIC, y being fiber)
                [atm paras ....,vsini,rv] for 1d spectra
        cubeobj: Data object.
            Must inherit breads.instruments.instrument.Instrument.
        prt_base_atm: petitRADTRANS atmospheric model object.
            prt.radtrans object with the base atmospheric model.
        prt_atm_params: Dictionary of parameters to be passed to the petitRADTRANS atmospheric model.
            dict with the following
        star_spectrum: Stellar spectrum to be continuum renormalized to fit the speckle noise at each location. It is
            (for now) assumed to be the same everywhere which is not compatible with a field dependent wavelength solution.
            np.ndarray of size the number of wavelength bins.
        boxw: size of the stamp to be extracted and modeled around the (x,y) location of the planet.
            Must be odd. Default is 1.
        psfw: Width (sigma) of the 2d gaussian used to model the planet PSF. This won't matter if boxw=1 however.
        nodes: If int, number of nodes equally distributed. If list, custom locations of nodes [x1,x2,..].
            To model discontinous functions, use a list of list [[x1,...],[xn,...]].
        badpixfraction: Max fraction of bad pixels in data.
        loc: Deprecated, Use fix_parameters.
            (x,y) position of the planet for spectral cubes, or fiber position (y position) for 2d data.
            When loc is not None, the x,y non-linear parameters should not be given.
        fix_parameters: List. Use to fix the value of some non-linear parameters. The values equal to None are being
                    fitted for, other elements will be fixed to the value specified.

    Returns:
        d: Data as a 1d vector with bad pixels removed (no nans)
        M: Linear model as a matrix of shape (Nd,Np) with bad pixels removed (no nans). Nd is the size of the data
            vector and Np = N_nodes*boxw^2+1 is the number of linear parameters.
        s: Noise vector (standard deviation) as a 1d vector matching d.
    """
    # Define the petitRADTRANS atm
    atmosphere = prt_base_atm
    pressures_bar = prt_base_atm.pressures*1e-6 # cgs to bar
    planet_radius = nonlin_paras[0]*r_jup
    ref_grav = 10**nonlin_paras[1]
    ref_press = prt_atm_params['reference_pressure']

    kappa_IR = prt_atm_params['kappa_IR'] # (cm2.s-1) infrared mean opacity for a solar metallicity (Z = 0) atmosphere
    gamma = prt_atm_params['gamma']
    delta = prt_atm_params['delta'] = 0.1 # The delta parameter of Guillot et al. 2010, used for retrieval guillot global only (instead of kappa and gamma)
    T_int = nonlin_paras[2]
    #T_equ = prt_atm_params['equilibrium_temperature']
    T_equ = T_int
    
    from petitRADTRANS.physics import temperature_profile_function_guillot, temperature_profile_function_guillot_global, temperature_profile_function_guillot_dayside, temperature_profile_function_guillot_metallic, temperature_profile_function_guillot_modif, temperature_profile_function_ret_model, temperature_profile_function_isothermal

    if prt_atm_params['atm_profile'] == 'guillot':
        redistribution_coeff = prt_atm_params['redistribution_coefficient'] # The redistribution coefficient of the irradiance. A value of 1 corresponds to the substellar point, 1/2 for the day-side average and 1/4 for the global average.
        temperatures = temperature_profile_function_guillot(pressures = pressures_bar,
                                                            infrared_mean_opacity = kappa_IR,
                                                            gamma = gamma, 
                                                            gravities = ref_grav, 
                                                            intrinsic_temperature = T_int, 
                                                            equilibrium_temperature = T_equ, 
                                                            redistribution_coefficient=redistribution_coeff if redistribution_coeff is not None else 0.25)
    elif prt_atm_params['atm_profile'] == 'guillot_global':
        temperatures = temperature_profile_function_guillot_global(pressures = pressures_bar, 
                                                                   infrared_mean_opacity = kappa_IR, 
                                                                   gamma = gamma, 
                                                                   gravities = ref_grav, 
                                                                   intrinsic_temperature = T_int, 
                                                                   equilibrium_temperature = T_equ)
    elif prt_atm_params['atm_profile'] == 'guillot_dayside':
        temperatures = temperature_profile_function_guillot_dayside(pressures = pressures_bar, 
                                                           infrared_mean_opacity = kappa_IR, 
                                                           gamma = gamma, 
                                                           gravities = ref_grav, 
                                                           intrinsic_temperature = T_int, 
                                                           equilibrium_temperature = T_equ)
    elif prt_atm_params['atm_profile'] == 'guillot_metallic':
        metallicity = prt_atm_params['metallicity']  # The metallicity in log10 relative to solar, used for guillot_metallic only
        infrared_mean_opacity_solar = prt_atm_params['infrared_mean_opacity_solar'] #(cm2.s-1) infrared mean opacity for a solar metallicity (Z = 0) atmosphere
        temperatures = temperature_profile_function_guillot_metallic(pressures = pressures_bar, 
                                                            gamma = gamma, 
                                                            reference_gravity = ref_grav, 
                                                            intrinsic_temperature = T_int, 
                                                            equilibrium_temperature = T_equ, 
                                                            infrared_mean_opacity_solar_matallicity = infrared_mean_opacity_solar, 
                                                            metallicity=metallicity)
    elif prt_atm_params['atm_profile'] == 'guillot_modif':
        temperatures = temperature_profile_function_guillot_modif(pressures = pressures_bar, 
                                                         delta = delta, 
                                                         gamma = gamma, 
                                                         intrinsic_temperature = T_int, 
                                                         equilibrium_temperature = T_equ, 
                                                         ptrans = None, 
                                                         alpha = None)
    elif prt_atm_params['atm_profile'] == 'ret_model':
        temperatures = temperature_profile_function_ret_model(pressures_bar,
        )
    elif prt_atm_params['atm_profile'] == 'isothermal':
        temperatures = temperature_profile_function_isothermal(pressures_bar, temperature = T_equ)
    else:
        print('Unknown atmospheric profile. Please choose one of the following:',
              'guillot_global \n'
              'guillot_global_ret \n',
              'guillot_dayside \n',
              'guillot \n',
              'guillot_metallic \n',
              'guillot_modif \n',
              'ret_model \n',
              'isothermal')
    
    
    mean_molar_masses = prt_atm_params['mean_molar_masses']
    C1 = nonlin_paras[3] #H2(16)O
    C2 = nonlin_paras[4] #C(16)O
    C3 = nonlin_paras[5] #(13)CO
    C4 = nonlin_paras[6] #(12)CH4
    C5 = nonlin_paras[7] #H2(18)O
    
    mass_fractions = {
    'H2': 0.74 * np.ones_like(temperatures, dtype=float), 
    'He': 0.24 * np.ones_like(temperatures, dtype=float),
    '1H2-16O': 10**C1 * np.ones_like(temperatures, dtype=float),
    '1H2-18O': 10**C5 * np.ones_like(temperatures, dtype=float), 
    '12C-16O': 10**C2 * np.ones_like(temperatures, dtype=float),
    '13C-16O': 10**C3 * np.ones_like(temperatures, dtype=float),
    '12C-1H4': 10**C4 * np.ones_like(temperatures, dtype=float)
}
    
            #  2.33 is a typical value for H2-He dominated atmospheres
    # mean_molar_masses = 2.33 * np.ones_like(temperatures)
    print("Generating Atmosphere . . . ")
    
    waves, flux = generate_prt_atm(base_atmosphere=atmosphere,temperatures=temperatures,mass_fractions=mass_fractions, mean_molar_masses=mean_molar_masses, reference_gravity=ref_grav, planet_radius=planet_radius, frequencies_to_wavelengths=True)
    
    wv = waves*10**4
    # print(wv)
    if loc is None:
        print('Please specify a location for which to broaden the model')
    else:
        if __name__=="__main__":
            # with Pool(process=16) as p:
                flux = cubeobj.broaden(wv,flux,loc=loc,mppool=None)
    if transmission is None:
        transmission = np.ones(cubeobj.data.shape[0])

    if fix_parameters is not None:
        _nonlin_paras = np.array(fix_parameters)
        _nonlin_paras[np.where(np.array(fix_parameters)==None)] = nonlin_paras
    else:
        _nonlin_paras = nonlin_paras

    Natmparas = len(nonlin_paras)-2
    atm_paras = [p for p in _nonlin_paras[0:Natmparas]]
    other_nonlin_paras = _nonlin_paras[Natmparas::]

    # Handle the different data dimensions
    # Convert everything to 3D cubes (wv,y,x) for the followying
    if len(cubeobj.data.shape)==1:
        data = cubeobj.data[:,None,None]
        noise = cubeobj.noise[:,None,None]
        bad_pixels = cubeobj.bad_pixels[:,None,None]
    elif len(cubeobj.data.shape)==2:
        data = cubeobj.data[:,:,None]
        noise = cubeobj.noise[:,:,None]
        bad_pixels = cubeobj.bad_pixels[:,:,None]
    elif len(cubeobj.data.shape)==3:
        data = cubeobj.data
        noise = cubeobj.noise
        bad_pixels = cubeobj.bad_pixels
    if cubeobj.refpos is None:
        refpos = [0,0]
    else:
        refpos = cubeobj.refpos

    vsini,rv = other_nonlin_paras[0:2]
    # Defining the position of companion
    # If loc is not defined, then the x,y position is assume to be a non linear parameter.
    if np.size(loc) ==2:
        x,y = loc
    elif np.size(loc) ==1 and loc is not None:
        x,y = 0,loc
    elif loc is None:
        if len(cubeobj.data.shape)==1:
            x,y = 0,0
        elif len(cubeobj.data.shape)==2:
            x,y = 0,other_nonlin_paras[2]
        elif len(cubeobj.data.shape)==3:
            x,y = other_nonlin_paras[3],other_nonlin_paras[2]

    nz, ny, nx = data.shape

    # Handle the different dimensions for the wavelength
    # Only 2 cases are acceptable, anything else is undefined:
    # -> 1d wavelength and it is assumed to be position independent
    # -> The same shape as the data in which case the wavelength at each position is specified and can bary.
    if len(cubeobj.wavelengths.shape)==1:
        wvs = cubeobj.wavelengths[:,None,None]
    elif len(cubeobj.wavelengths.shape)==2:
        wvs = cubeobj.wavelengths[:,:,None]
    elif len(cubeobj.wavelengths.shape)==3:
        wvs = cubeobj.wavelengths
    _, nywv, nxwv = wvs.shape

    if boxw % 2 == 0:
        raise ValueError("boxw, the width of stamp around the planet, must be odd in splinefm().")
    if boxw > ny or boxw > nx:
        raise ValueError("boxw cannot be bigger than the data in splinefm().")

    # remove pixels that are bad in the transmission or the star spectrum
    bad_pixels[np.where(np.isnan(star_spectrum*transmission))[0],:,:] = np.nan

    # Extract stamp data cube cropping at the edges
    w = int((boxw - 1) // 2)

    _paddata =np.pad(data,[(0,0),(w,w),(w,w)],mode="constant",constant_values = np.nan)
    _padnoise =np.pad(noise,[(0,0),(w,w),(w,w)],mode="constant",constant_values = np.nan)
    _padbad_pixels =np.pad(bad_pixels,[(0,0),(w,w),(w,w)],mode="constant",constant_values = np.nan)
    k, l = int(np.round(refpos[1] + y)), int(np.round(refpos[0] + x))
    dx,dy = x-l+refpos[0],y-k+refpos[1]
    padk,padl = k+w,l+w

    # high pass filter the data
    cube_stamp = _paddata[:, padk-w:padk+w+1, padl-w:padl+w+1]
    badpix_stamp = _padbad_pixels[:, padk-w:padk+w+1, padl-w:padl+w+1]
    badpixs = np.ravel(badpix_stamp)
    d = np.ravel(cube_stamp)
    s = np.ravel(_padnoise[:, padk-w:padk+w+1, padl-w:padl+w+1])
    badpixs[np.where(s==0)] = np.nan

    # manage all the different cases to define the position of the spline nodes
    if type(nodes) is int:
        N_nodes = nodes
        x_knots = np.linspace(np.min(wvs), np.max(wvs), N_nodes, endpoint=True).tolist()
    elif type(nodes) is list  or type(nodes) is np.ndarray :
        x_knots = nodes
        if type(nodes[0]) is list or type(nodes[0]) is np.ndarray :
            N_nodes = np.sum([np.size(n) for n in nodes])
        else:
            N_nodes = np.size(nodes)
    else:
        raise ValueError("Unknown format for nodes.")

    # Number of linear parameters
    fitback = False
    if fitback:
        N_linpara = boxw * boxw * N_nodes +1 + 3*boxw**2
    else:
        N_linpara = boxw * boxw * N_nodes +1



    where_finite = np.where(np.isfinite(badpixs))
    if np.size(where_finite[0]) <= (1-badpixfraction) * np.size(badpixs) or vsini < 0 or \
            padk > ny+2*w-1 or padk < 0 or padl > nx+2*w-1 or padl < 0:
        # don't bother to do a fit if there are too many bad pixels
        return np.array([]), np.array([]).reshape(0,N_linpara), np.array([])
    else:
        # Get the linear model (ie the matrix) for the spline
        M_speckles = np.zeros((nz, boxw, boxw, boxw, boxw, N_nodes))
        for _k in range(boxw):
            for _l in range(boxw):
                lwvs = wvs[:,np.clip(k-w+_k,0,nywv-1),np.clip(l-w+_l,0,nxwv-1)]
                M_spline = get_spline_model(x_knots, lwvs, spline_degree=3)
                M_speckles[:, _k, _l, _k, _l, :] = M_spline * star_spectrum[:, None]
        M_speckles = np.reshape(M_speckles, (nz, boxw, boxw, boxw * boxw * N_nodes))

        if fitback:
            M_background = np.zeros((nz, boxw, boxw, boxw, boxw,3))
            for _k in range(boxw):
                for _l in range(boxw):
                    lwvs = wvs[:,np.clip(k-w+_k,0,nywv-1),np.clip(l-w+_l,0,nxwv-1)]
                    M_background[:, _k, _l, _k, _l, 0] = 1
                    M_background[:, _k, _l, _k, _l, 1] = lwvs
                    M_background[:, _k, _l, _k, _l, 2] = lwvs**2
            M_background = np.reshape(M_background, (nz, boxw, boxw, 3*boxw**2))

        # w1,planet_model = pyasl.equidistantInterpolation(wv,flux,"mean")
        planet_model = flux
        # print(cubeobj.wavelengths[:,loc].shape)
        # print(planet_model.shape)

        #cubeobj.wavelengths[:,loc]     
        if np.sum(np.isnan(planet_model)) >= 1 or np.sum(planet_model)==0 or np.size(wv) != np.size(planet_model):
            return np.array([]), np.array([]).reshape(0,N_linpara), np.array([])
        else:
            if vsini != 0:
                # print(w1.shape)
                # print(planet_model.shape)
                spinbroad_model = pyasl.fastRotBroad(wv, planet_model, 0.1, vsini)
            else:
                spinbroad_model = planet_model
            planet_f = interp1d(wv,spinbroad_model, bounds_error=False, fill_value=0)

        psfs = np.zeros((nz, boxw, boxw))
        # Technically allows super sampled PSF to account for a true 2d gaussian integration of the area of a pixel.
        # But this is disabled for now with hdfactor=1.
        hdfactor = 1#5
        xhdgrid, yhdgrid = np.meshgrid(np.arange(hdfactor * (boxw)).astype(float) / hdfactor,
                                    np.arange(hdfactor * (boxw)).astype(float) / hdfactor)
        psfs += pixgauss2d([1., w+dx, w+dy, psfw, 0.], (boxw, boxw), xhdgrid=xhdgrid, yhdgrid=yhdgrid)[None, :, :]
        psfs = psfs / np.nansum(psfs, axis=(1, 2))[:, None, None]

        # flux ratio normalization
        star_flux = np.nanmean(star_spectrum) * np.size(star_spectrum)

        scaled_psfs = np.zeros((nz,boxw,boxw))+np.nan
        for _k in range(boxw):
            for _l in range(boxw):
                lwvs = wvs[:,np.clip(k-w+_k,0,nywv-1),np.clip(l-w+_l,0,nxwv-1)]
                # The planet spectrum model is RV shifted and multiplied by the tranmission
                planet_spec = transmission * planet_f(lwvs * (1 - (rv - cubeobj.bary_RV) / const.c.to('km/s').value))
                scaled_psfs[:,_k,_l] = psfs[:, _k,_l] * planet_spec

        planet_flux = np.size(scaled_psfs) * np.nanmean(scaled_psfs)
        scaled_psfs = scaled_psfs / planet_flux * star_flux
        # print(np.nansum(scaled_psfs))

        # combine planet model with speckle model
        if fitback:
            M = np.concatenate([scaled_psfs[:, :, :, None], M_speckles,M_background], axis=3)
        else:
            M = np.concatenate([scaled_psfs[:, :, :, None], M_speckles], axis=3)
        # Ravel data dimension
        M = np.reshape(M, (nz * boxw * boxw, N_linpara))
        # Get rid of bad pixels
        sr = s[where_finite]
        dr = d[where_finite]
        Mr = M[where_finite[0], :]

        if return_where_finite:
            return dr, Mr, sr, where_finite
        else:
            return dr, Mr, sr
        
def generate_prt_atm(base_atmosphere,temperatures,mass_fractions, mean_molar_masses, reference_gravity, planet_radius, frequencies_to_wavelengths=True):
    atmosphere = base_atmosphere
    
    
    waves, flux, _ = atmosphere.calculate_flux(temperatures=temperatures,
                                                mass_fractions=mass_fractions,
                                                mean_molar_masses = mean_molar_masses,
                                                reference_gravity = reference_gravity,
                                                planet_radius = planet_radius,
                                                frequencies_to_wavelengths=True)
    return waves, flux

def set_up_atm_from_yaml(config_path):
    """
    Set up the petitRADTRANS atmosphere from a YAML configuration file.
    """
    with open(config_path, 'r') as stream:
        config = yaml.safe_load(stream=stream)
        print(config)
    pressures_dict = config['pressures'][0]
    # print(config['wavelength_boundaries'][0]['start'])
    pressures = np.logspace(float(pressures_dict['start']), float(pressures_dict['stop']), pressures_dict['num'])
    wavelength_boundaries = [float(config['wavelength_boundaries'][0]['start']), float(config['wavelength_boundaries'][0]['stop'])]
    # print(wavelength_boundaries)
    line_species = config['line_species']
    # print(line_species)
    rayleigh_species = config['rayleigh_species']
    # print(rayleigh_species)
    continuum_opacities = config['continuum_opacities']
    # print(continuum_opacities)
    # Set up the Radtrans object
    atmosphere_from_yaml = Radtrans(
        pressures=pressures,
        line_species=line_species,
        rayleigh_species=rayleigh_species,
        gas_continuum_contributors=continuum_opacities,
        wavelength_boundaries=wavelength_boundaries,
        line_opacity_mode='lbl')
    
    return atmosphere_from_yaml

def get_wl_bounds(dataobj, hostobj, A0obj1):
    """
    Get the wavelength bounds for the data object.
    """
    wl_bounds = {}
    wl_bounds['min'] = np.min(dataobj.wavelengths)
    wl_bounds['max'] = np.max(dataobj.wavelengths)
    wl_bounds['host_min'] = np.min(hostobj.wavelengths)
    wl_bounds['host_max'] = np.max(hostobj.wavelengths)
    wl_bounds['A0_min'] = np.min(A0obj1.wavelengths)
    wl_bounds['A0_max'] = np.max(A0obj1.wavelengths)
    wl_min = np.min([wl_bounds['min'], wl_bounds['host_min'], wl_bounds['A0_min']])
    wl_max = np.max([wl_bounds['max'], wl_bounds['host_max'], wl_bounds['A0_max']])
    return wl_bounds, wl_min, wl_max