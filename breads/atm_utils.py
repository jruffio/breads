from breads.utils import broaden

import species
from species.data.database import Database
from species.read.read_model import ReadModel

species.SpeciesInit()
database = Database()

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from species.phot.syn_phot import SyntheticPhotometry
import astropy.units as u
from astropy import constants as const

from contextlib import redirect_stdout
from io import StringIO
import sys
import os

def rprint(string):
    sys.stdout.write('\r'+str(string))
    sys.stdout.flush()

class NullIO(StringIO):
    def write(self, txt):
        pass

def silent(fn):
    """Decorator to silence functions."""
    def silent_fn(*args, **kwargs):
        with redirect_stdout(NullIO()):
            return fn(*args, **kwargs)
    return silent_fn

class miniRGI():
    def __init__(self,
                 model_name,
                 wavelength_bounds = None,
                 R = None,
                 filter_name = None,
                 MJy_flag = False,
                 parallel_broadening_flag = True,
                 save_flag = True,
                 load_flag = True):
        
        self.model_name = model_name
        self.wavelength_bounds = wavelength_bounds
        self.R = R
        self.filter_name = filter_name
        self.MJy_flag = MJy_flag
        self.parallel_broadening_flag = parallel_broadening_flag
        self.save_flag = save_flag
        self.load_flag = load_flag
        
        wbstr = '_'+str(wavelength_bounds[0])+'-'+str(wavelength_bounds[1])+'um' if not wavelength_bounds is None else ''
        Rstr = '_R'+str(R) if not R is None else ''
        self.identity = 'miniRGI_'+model_name+wbstr+Rstr
        
        self.savedir = database.data_folder[:-6]+'/miniRGI/'
        self.fname = self.identity+'.npy'
        
        RM = ReadModel(model=self.model_name)
        self.points = RM.get_points()
        self.bounds = RM.get_bounds()
        print('miniRGI model: {},\nparameter bounds: {}'.format(self.identity,self.bounds))
        with h5py.File(database.database, "r") as hdf_file:
            self.wavelength = np.copy(hdf_file['models'][model_name]['wavelength'])
        if self.wavelength_bounds is not None:
            print('restricting wavelengths: {} to {} um'.format(self.wavelength_bounds[0],self.wavelength_bounds[1]))
            self.wavelength_bool = np.logical_and(self.wavelength>=self.wavelength_bounds[0],
                                                  self.wavelength<=self.wavelength_bounds[1])
            self.wavelength = self.wavelength[self.wavelength_bool]
        print('Spectral Resolution: {}'.format(self.R)) if R is not None else None
        self.values = self.fakeNDarray([len(v) for v in self.points.values()]+[len(self.wavelength)])
        
        if self.MJy_flag:
            print('Normalizing to 1 MJy in filter : {}'.format(self.filter_name))
            self.synphot = SyntheticPhotometry(self.filter_name)

        if self.load_flag:
            self.load()
        else:
            self.miniRGI_dictionary = {}
        
    def __call__(self,atm_paras):
        
        slicing, mini_param_list = self.define_subgrid_slice(atm_paras)
        slice_string = str(slicing)
        if not (slice_string in self.miniRGI_dictionary.keys()):
            self.load_and_broaden_mini_hypercube(slicing, mini_param_list)
            
        mRGI = self.miniRGI_dictionary[slice_string]
            
        if self.MJy_flag:
            return self.normalize(mRGI(atm_paras))
        else:
            return mRGI(atm_paras)
    
    def define_subgrid_slice(self, atm_paras):
        
        with h5py.File(database.database, "r") as hdf_file:
    
            model = hdf_file['models'][self.model_name]    

            n_param = model.attrs['n_param']
            param_list = []
            for n in range(n_param):
                key = model.attrs['parameter{}'.format(n)]
                point_vector = np.copy(model[key])
                param_list.append(point_vector)
        
            slicing = []
            mini_param_list = []
            for ind,p in enumerate(atm_paras):
                plist = param_list[ind]
                w1 = np.where(plist<=p)[0][-1]
                w2 = np.where(plist>=p)[0][0]
                slicing.append(slice(w1,w2+1))
                if w2 == w1:
                    mini_param_list.append([param_list[ind][w1]])
                else:
                    mini_param_list.append([param_list[ind][w1],param_list[ind][w2]])
            if self.wavelength_bounds is None:
                slicing.append(slice(None))
            else:
                where_wavelength_bool = np.where(self.wavelength_bool==True)[0]
                slicing.append(slice(where_wavelength_bool[0],where_wavelength_bool[-1]+1))

        return slicing, mini_param_list
    
    def load_and_broaden_mini_hypercube(self, slicing, mini_param_list):
        
        with h5py.File(database.database, "r") as hdf_file: 
            model = hdf_file['models'][self.model_name]  
            model_hypercube = model['flux']
            mini_hypercube = np.copy(model_hypercube[tuple(slicing)])
            
        if self.R is not None:
            parameter_shape = mini_hypercube.shape[:-1]
            n_grid_points = np.prod(parameter_shape)
            reshaped_cube = mini_hypercube.reshape(n_grid_points,-1)
            broad_cube = np.empty(reshaped_cube.shape)
            
            def _miniRGI_broadening_task(i):
                rprint('broadening... {}/{} '.format(i+1,n_grid_points))
                return broaden(self.wavelength,reshaped_cube[i,:],R=self.R)
                
            if self.parallel_broadening_flag:
                from multiprocess import Pool
                with Pool(n_grid_points) as pool:
                    outputs = pool.map(_miniRGI_broadening_task,range(n_grid_points))
                for i,o in enumerate(outputs):
                    broad_cube[i,:] = o
            else:
                for i in range(n_grid_points):
                    broad_cube[i,:] = _miniRGI_broadening_task(i)
            print()
            re_reshaped_cube = broad_cube.reshape(parameter_shape+(-1,))
            mini_hypercube = re_reshaped_cube

        self.miniRGI_dictionary[str(slicing)] = RegularGridInterpolator(tuple(mini_param_list), mini_hypercube, method='linear', fill_value=np.nan)
        if self.save_flag:
            self.save()
            
    def normalize(self,flux):

        flam = flux*u.W/u.m**2/u.micron
        w0 = np.mean(self.synphot.wavel_range)
        fnu = flam * (w0 * u.um)**2 / const.c
        fMJy = fnu.to(u.MJy).value
        bandflux, _ = self.synphot.spectrum_to_flux(self.wavelength,fMJy.reshape(-1))
        normflux = fMJy/bandflux
        
        return normflux
        
    def load(self):
        if os.path.exists(self.savedir+self.fname):
            print('loading... {}'.format(self.savedir+self.fname))
            self.miniRGI_dictionary = np.load(self.savedir+self.fname,allow_pickle=True).item()
        else:
            print('nothing to load... empty memory.')
            self.miniRGI_dictionary = {}
    
    def save(self):
        if not os.path.exists(self.savedir):
            os.mkdir(self.savedir)
        print('saving... {}'.format(self.savedir+self.fname))
        np.save(self.savedir+self.fname,self.miniRGI_dictionary)

    class fakeNDarray():
        def __init__(self,shape):
            self.shape = tuple(shape)
    
def realRGI(model_name):
    
    with h5py.File(database.database, "r") as hdf_file:
        model = hdf_file['models'][model_name]    
        n_param = model.attrs['n_param']
        param_list = []
        for n in range(n_param):
            key = model.attrs['parameter{}'.format(n)]
            point_vector = np.copy(model[key])
            param_list.append(point_vector)
        model_hypercube = np.copy(model['flux'])
        w = np.copy(model['wavelength'])

    return w, RegularGridInterpolator(tuple(param_list), model_hypercube, method='linear', fill_value=np.nan)

def download_all_models(skip=[], verbose=True, clobber=False):
    
    def download_all_task(skip, clobber):
        downloaded_model_keys, all_available_keys = viable_models()
        downloaded_model_keys = [] if clobber else downloaded_model_keys
        for k in all_available_keys:
            if (k not in downloaded_model_keys) and (k not in skip):
                    database.add_model(model=k)
            else:
                print('skipping... {}'.format(k))
                
    if verbose:
        download_all_task(skip,clobber)
    else:
        silent(download_all_task)(skip,clobber)

def viable_models():

    with h5py.File(database.database, "r") as hdf_file:
        downloaded_model_keys = list(hdf_file['models'].keys())
    
    dbdict = silent(database.available_models)()
    all_available_keys = list(dbdict.keys())
    
    return downloaded_model_keys, all_available_keys

def valid_models(keylist,search_bounds,plotflag=False):
    valid = []
    for k in keylist:
        points = ReadModel(model=k).get_points()
        v = {}
        for bk in search_bounds.keys():
            bs = search_bounds[bk]
            if bs is not None:
                try:
                    ps = points[bk]
                    if np.any(ps <= bs[0]) and np.any(ps >= bs[1]):
                        v[bk] = True
                    else:
                        v[bk] = False
                except KeyError:
                    pass
        if np.all(list(v.values())):
            valid.append(k)  
    valid = np.unique(valid)
    
    if plotflag:
        valid_plot(keylist,search_bounds,valid)

    return valid

def valid_plot(keylist,search_bounds,valid):
    
    grid_dict = {}
    mean_Ts = []
    for k in keylist:
        read_model = ReadModel(model=k)
        points = read_model.get_points()
        grid_dict[k] = points
        mean_T = np.mean(points['teff'])
        mean_Ts.append(mean_T)
    mean_Ts_sort,keys_sort = zip(*sorted(zip(mean_Ts,list(grid_dict))))
    
    fig,ax = plt.subplots(1,7,figsize=(50,15),dpi=150,sharey=True)

    params = ['teff','logg','feh','c_o_ratio','log_kzz','fsed','ad_index']
    keys = grid_dict.keys()
    colors = plt.cm.plasma(np.linspace(0,1,len(keys)))

    for j,k in enumerate(keys_sort):
        param_dict = grid_dict[k]
        for i,p in enumerate(params):
            try:
                pvec = param_dict[p]
            except:
                pvec = np.nan*np.ones(2)
            ax[i].plot(pvec,j*np.ones(len(pvec)),color=colors[j],marker='.')

    for i,p in enumerate(params):
        bs = search_bounds[p]
        if bs is not None:
            ax[i].axvline(x=bs[0],color='k')
            ax[i].axvline(x=bs[1],color='k')
        ax[i].set_xlabel(p)

    tick_labels = []
    for k in keys_sort:
        if k in valid:
            tick_labels.append(r'$\mathbf{'+k+r'}$')
        else:
            tick_labels.append(k)

    ax[0].set_yticks(range(len(keys)))
    ax[0].set_yticklabels(tick_labels)
    ax[0].invert_yaxis()
    ax[0].set_xscale('log')
    ax[0].set_xlim([100,10000])
    ax[1].set_xlim([2.5,6])
    ax[2].set_xlim([-2,2])
    plt.show()
    
def object_memory_profiler(obj,g            ,level = 0, verbose=True):   
    #                       !! g = globals()
    if level == 0:
        for k in g.keys():
            if g[k] is obj:
                print()
                print('!----! profiling: {} !----!'.format(k))
                print('type: {}'.format(obj.__class__))
                print()
                break
    
    total_bytes = 0

    if hasattr(obj,'__dict__'):
        iterable = obj.__dict__.keys()
        container = obj.__dict__
    elif obj.__class__ is list:
        iterable = range(len(obj))
        container = obj
    elif obj.__class__ is dict:
        iterable = obj.keys()
        container = obj
    else:
        iterable = None
        container = None

    if iterable is not None:
        for k in iterable:
            item = container[k]
            size = object_memory_profiler(item,g,level=level+1)
            print('   '*level,k,size) if verbose else None
            total_bytes += size
        return total_bytes
    else:
        if obj.__class__ is type(np.array([])):
            return obj.nbytes
        else:
            return sys.getsizeof(obj)