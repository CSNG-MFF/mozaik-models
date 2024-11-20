from mozaik.storage.datastore import PickledDataStore
from parameters import ParameterSet
from mozaik.storage.queries import *
import sys
import numpy as np
import scipy
import pickle
from mozaik.analysis.technical import NeuronAnnotationsToPerNeuronValues
import time
from scipy import signal
from mozaik.analysis.analysis import Analysis
from mozaik.analysis.data_structures import SingleValue
from som import SOM
import quantities as pq
import copy
from skimage.transform import resize
from skimage.draw import disk
from mozaik.storage.datastore import DataStoreView

import matplotlib.gridspec as gridspec
import pylab
from mozaik.tools.mozaik_parametrized import varying_parameters
from mozaik.tools.distribution_parametrization import load_parameters
from scipy.optimize import curve_fit

def get_experimental_data():
    return {
        "Smith 2018": {
            "similarity": [0.382,0.437,0.562,0.457,0.368,0.382,0.288,0.475],
            "spatial scale of correlation": [0.83419,0.72982,1.08257,0.96128,1.09259],
            "dimensionality": [4,7,15.7,15.9,21.4],
            "mean eccentricity": [0.7508,0.68167,0.69373,0.60048,0.59405],
            "local correlation eccentricity_hist": {
                "x": [0.02380952, 0.07142857, 0.11904762, 0.16666667, 0.21428571, 0.26190476, 0.30952381, 0.35714286, 0.4047619 , 0.45238095, 0.5, 0.54761905, 0.5952381 , 0.64285714, 0.69047619,0.73809524, 0.78571429, 0.83333333, 0.88095238, 0.92857143,0.97619048],
                "y": [0, 0, 0, 0.00267881, 0.00401822,0.00750067, 0.01392982, 0.01607286, 0.02518082, 0.04125368,0.05464774, 0.07232789, 0.08143584, 0.10393785, 0.11197428,0.11251005, 0.12643986, 0.09375837, 0.06804179, 0.04553978,0.01875167],
            },
        },
        "Mulholland 2021": {
            "exc inh similarity": [0.533,0.207,0.419],
            "corr above 2 mm": {
                "exc": [0.26698,0.3316,0.19743,0.30415,0.27338,0.33148,0.26845],
                "inh": [0.28544,0.20432,0.26624,0.22759,0.37394,0.32446,0.28704],
            },
            "corr wavelength": {
                "exc": [0.77539,0.76498,0.74042,0.85129,0.81111,0.95075,1.01449],
                "inh": [0.67875,0.92874,0.74371,1.16734,0.96917,0.83597,0.67894],
            },
            "mean eccentricity": {
                "exc": [0.7323,0.7117,0.7022,0.6933,0.6839,0.6579,0.6219],
                "inh": [0.6455,0.6567,0.6579,0.6697,0.7211,0.7347,0.7754],
            },
            "dimensionality": {
                # Dimensionality (mean of 100x random sample of 30 events) different from Smith 2018 (all events 1x)
                "exc": [9.6303,11.3543,6.6423,10.6972,8.3869,8.9095,9.6195],
                "inh": [12.6281,10.702,11.636,8.9744,8.7557,12.129,10.5414],
            },
        },
        "Mulholland 2024": {
            "exc opt similarity": [0.50349,0.56358,0.6047,0.55066,0.45385,0.37765],
            "spontaneous": {
                "wavelength": {
                    "hist": {
                        "x": [0.55, 0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35, 1.45, 1.55],
                        "y": [0.00661,0.10761,0.41724,0.273,0.10933,0.03516,0.02023,0.01135,0,0.01048,0.00524],
                    },
                    "mean": [0.752101,0.702946,0.765684,0.829392,0.832626,0.836509,0.896983],
                    "std": [0.0965972,0.138543,0.0851731,0.0878505,0.147288,0.274197,0.174062],
                },

            },
            "fullfield opto": {
                "wavelength": {
                    "hist": {
                        "x": [0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35, 1.45, 1.55, 1.65, 1.75],
                        "y": [0.005945,0.008341,0.025811,0.194454,0.308177,0.251508,0.119363,0.04852,0.019643,0.01318,0.005614,0.008601,0.002685,0.003032,0.007663], 
                    },
                    "mean": [0.81685,0.816851,0.717572,0.763818,0.871507,0.874743,0.847903],
                    "std": [0.075143,0.187204,0.193154,0.211712,0.0649419,0.257613,0.133793],
                },
            },
        }
    }

def get_insides_outsides(resp, stims):
    insides = []
    outsides = []

    for i in range(resp.shape[0]):
        for j in range(resp.shape[1]):
            stim = stims[i,...].flatten()
            orr = resp[i,j,:,:,:].reshape([-1,resp.shape[-1]])
            inside = orr[stim > 0,:].mean(axis=0)
            outside = orr[stim == 0,:].mean(axis=0)
            insides.append(inside)
            outsides.append(outside)
            
    return np.stack(insides),np.stack(outsides)

def get_A_multitrial(dsv_in,s_res,t_res,array_width,electrode_radius=50):
    
    def get_As(dsv_in,s_res,t_res,array_width,electrode_radius):
        As = []
        trials_list = [load_parameters(s.replace("MozaikExtended",""))["trial"] for s in dsv_in.get_stimuli()]
        assert len(trials_list) == max(trials_list) + 1 # all trials are in dsv_in
        trials = len(trials_list)
        for trial in range(trials):
            dsv = param_filter_query(dsv_in,st_trial=trial)
            A = gen_st_array(dsv, s_res=s_res, t_res=t_res, array_width=array_width, electrode_radius=50)
            As.append(A)

        As = np.array(As)
        dsv_in.full_datastore.purge_segments()
        return As
    if type(dsv_in) == list:
        return np.stack([get_As(dsv,s_res,t_res,array_width,electrode_radius) for dsv in dsv_in])
    else:
        return get_As(dsv_in,s_res,t_res,array_width,electrode_radius)

def retrieve_ds_param_values(dsv, param_name):
    # Hacky function because of DataStore limitations
    # Retrieves all direct stimulation parameter values from dsv
    l=[]
    for s in dsv.get_stimuli():
        if MozaikParametrized.idd(s).direct_stimulation_parameters != None:
            l.append(MozaikParametrized.idd(s).direct_stimulation_parameters.stimulating_signal_parameters[param_name])
    return sorted(list(set(l)))

def fetch_ds_dsv(dsv,params,remove_direct_stim_params=False):
    # Hacky function because of DataStore limitations
    # Retrieves dsv based on direct stimulation parameter
    stims = dsv.get_stimuli()
    segs = dsv.get_segments()
    dsv_out = DataStoreView(ParameterSet({}), dsv.full_datastore)
    for i in range(len(stims)): 
        s = ParameterSet(str(stims[i]).replace("MozaikExtended",""))
        s = ParameterSet(str(s))["direct_stimulation_parameters"]
        if s == None:
            continue
        s = ParameterSet(s)["stimulating_signal_parameters"]
        if params.items() <= s.items():
            dsv_out.block.segments.append(segs[i])
            if remove_direct_stim_params:
                p = load_parameters(segs[i].annotations['stimulus'].replace("MozaikExtended",""))
                p.direct_stimulation_parameters = None
                segs[i].annotations['stimulus'] = str(p)
    return dsv_out

def load_ds(p, sheet="V1_Exc_L2/3"):
    dss = []
    if type(p) == str:
        dss.append(param_filter_query(
            PickledDataStore(
            load=True,
            parameters=ParameterSet({"root_directory": p, "store_stimuli": False}),
            replace=False,
            ), sheet_name=sheet))
    elif type(p) == list:
        dss.extend([ param_filter_query(
                PickledDataStore(
                load=True,
                parameters=ParameterSet({"root_directory": path, "store_stimuli": False}),
                replace=False,
            ), sheet_name=sheet)
            for path in p
        ])
    dss_ret = []
    # Split datastores based on image_path or radius, if exists
    for ds in dss:
        if "image_path" not in ds.get_stimuli()[0] and "radius" not in ds.get_stimuli()[0]:
            dss_ret.append(ds)
            continue
        if "image_path" in ds.get_stimuli()[0]:    
            im_paths = retrieve_ds_param_values(ds, "image_path")
            dss_ret.extend([fetch_ds_dsv(ds,{"image_path":im_path}) for im_path in im_paths])
        if "radius" in ds.get_stimuli()[0]:    
            radii = retrieve_ds_param_values(ds, "radius")
            dss_ret.extend([fetch_ds_dsv(ds,{"radius":radius}) for radius in radii])
    return dss_ret if len(dss_ret) > 1 else dss_ret[0]

def retrieve_ds_param_values(dsv, param_name):
    l=[]
    for s in dsv.get_stimuli():
        if MozaikParametrized.idd(s).direct_stimulation_parameters != None:
            l.append(MozaikParametrized.idd(s).direct_stimulation_parameters.stimulating_signal_parameters[param_name])
    return sorted(list(set(l)))

def get_stim_patterns(dsv_list, array_width, s_res):
    patterns = []
    for dsv in dsv_list:
        basepath = "/".join(dsv.full_datastore.parameters.root_directory.split('/')[:-3]) + "/"
        im_paths = retrieve_ds_param_values(dsv, "image_path")
        assert len(im_paths) == 1
        im_path = im_paths[0]
        pattern = np.load(basepath[1:]+im_path).T
        pattern = resize(pattern, (array_width // s_res, array_width // s_res),anti_aliasing=False)
        patterns.append(pattern)
    return np.stack(patterns)

def calc_dsv_correlations(A,stims,t_res,stim_start,stim_end):
    corrs = np.zeros((A.shape[0],A.shape[1]))
    assert A.shape[0] == len(stims)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            A_im = A[i,j,:,:,stim_start//t_res:(stim_end+500)//t_res].mean(axis=2)
            corrs[i][j], _ = scipy.stats.pearsonr(A_im.flatten(),stims[i,:,:].flatten())
    return corrs

def normalize(x,nmin=None,nmax=None):
    if nmin==None:
        nmin=x.min()
    if nmax==None:
        nmax=x.max()
    return (x-nmin) / (nmax-nmin)

def calcium_light_spread_kernel(s_res=40):
    x,y,I = np.load("calcium_light_spread_kernel.npy")
    x_t, y_t = np.meshgrid(np.arange(-2000,2001,s_res),np.arange(-2000,2001,s_res))
    return scipy.interpolate.griddata((x.flatten(), y.flatten()), I.flatten(), (x_t, y_t), method='cubic')
    
def get_calcium_signal(A_in, s_res, t_res, f0_2d=None, time_convolve=True, return_f0=False, calc_f0=True):

    A = A_in.copy()
    t_ker = t_kernel(t_res)
    # Padding is mean of first 1/4th of activity, hopefully without any stimulation
    cval = A[..., :(A.shape[-1] // 4)].mean()
    if time_convolve:
        A_t = scipy.ndimage.convolve1d(A, t_ker, axis=-1, mode='constant', cval=cval, origin=-len(t_kernel(t_res))//2)
    else:
        A_t = A
    
    s_ker = calcium_light_spread_kernel(s_res)
    if len(A.shape) == 3:
        A = np.dstack([scipy.signal.convolve(A_t[:,:,i],s_ker,mode='same') for i in range(A.shape[-1])])
    elif len(A.shape) == 4:
        A = np.stack([np.dstack([scipy.signal.convolve(A_t[j,:,:,i],s_ker,mode='same') for i in range(A.shape[-1])]) for j in range(A.shape[0])])
    elif len(A.shape) == 5:
        A = np.stack([np.stack([np.dstack([scipy.signal.convolve(A_t[j,k,:,:,i],s_ker,mode='same') for i in range(A.shape[-1])]) for k in range(A.shape[1])]) for j in range(A.shape[0])])
    
    if not calc_f0:
        return A
    tiling = np.ones_like(A.shape)
    tiling[-1] = A.shape[-1]
    if f0_2d is None:
        f0_2d = A[..., :(A.shape[-1] // 4)].mean(axis=-1)
    f0 = np.tile(f0_2d[...,np.newaxis],tiling)
    fbyf0 = (A - f0) / f0
    if return_f0:
        return fbyf0, f0_2d
    else:
        return fbyf0

def bandpass_filter(A_in, pixel_width_um):
    high_pass_sigma = 200 * (1/pixel_width_um)
    low_pass_sigma = 26 * (1/pixel_width_um)

    # Band-pass filtering
    filt = np.zeros(len(A_in.shape))
    filt[-3:-1] = low_pass_sigma
    flp = scipy.ndimage.gaussian_filter(A_in, filt)
    filt[-3:-1] = high_pass_sigma
    fhp = scipy.ndimage.gaussian_filter(flp, filt)
    fbp = flp - fhp
    
    return fbp

def t_kernel(t_res,length_ms=5000):
    # Based on https://doi.org/10.3389/fncir.2013.00201
    tau_on = 10 # ms rise time of calcium response
    tau_off = 1000 # ms fall time of calcium response
    if length_ms < 10 * t_res:
        length_ms = 10*t_res
        
    # We ignore the rise time for the moment
    return np.exp(-np.linspace(0,length_ms,length_ms//t_res)/tau_off)

# size: https://www.sciencedirect.com/science/article/pii/S0165027018301274?via%3Dihub
def s_kernel(sp_res):
    neuron_diameter = 20 # um
    neuron_diameter = max(1,neuron_diameter // sp_res)
    
    if neuron_diameter < 4:
        s_kernel = np.ones((neuron_diameter,neuron_diameter))
    else:  
        s_kernel_1d = signal.tukey(int(neuron_diameter*1.1),0.3)
        klen = len(s_kernel_1d)//2
        s_kernel_1d = s_kernel_1d[klen:]
        s_kernel = np.zeros((2*klen,2*klen))
        for x in range(2*klen):
            for y in range(2*klen):
                r = int(round(np.sqrt((x-klen)**2 + (y-klen)**2)))
                if r < klen:
                    s_kernel[x,y] = s_kernel_1d[r]
                else:
                    s_kernel[x,y] = 0
    return s_kernel

def s_kernel_smoothed(sp_res,sigma_cort=100):
    # sigma_cort = sigma in cortical coordinates (micrometres)
    sigma = sigma_cort / sp_res
    n_sigmas = 3 # To how many sigmas we sample
    sm_ker = s_kernel(sp_res)
    sm_ker = np.pad(sm_ker,int(sigma*n_sigmas))
    return scipy.ndimage.gaussian_filter(sm_ker, sigma,mode='constant')

def get_st_ids(dsv):
    assert len(dsv.sheets()) == 1
    return [s for s in dsv.get_segments() if len(s.spiketrains) > 0][0].get_stored_spike_train_ids()

def get_s(dsv, s_res=None, neuron_ids=None):
    if s_res == None:
        s_res = 1
    if neuron_ids is None:
        neuron_ids = get_st_ids(dsv)
    sheet = dsv.sheets()[0]
    pos = dsv.get_neuron_positions()[sheet]
    posx = np.round((pos[0, dsv.get_sheet_indexes(sheet, neuron_ids)] / s_res * 1000)).astype(int)
    posy = np.round((pos[1, dsv.get_sheet_indexes(sheet, neuron_ids)] / s_res * 1000)).astype(int)
    #posx -= min(posx)
    #posy -= min(posy)

    return posx, posy

def get_t(dsv, t_res=None):
    if t_res == None:
        t_res = 1
    st_ids = get_st_ids(dsv)
    segs = [s for s in dsv.get_segments() if len(s.spiketrains) > 0]
    t = [[] for i in range(len(st_ids))]
    time_passed = 0
    for i in range(len(segs)):
        if len(segs[i].spiketrains) == 0:
            continue
        sts = segs[i].get_spiketrains()
        for j in range(len(sts)):
            t[j].extend(list((sts[j].magnitude / t_res).astype(int) + time_passed))
        time_passed += int((sts[0].t_stop.magnitude - sts[0].t_start.magnitude) / t_res)
    return t

def electrode_positions(array_width, electrode_dist):
    assert array_width % electrode_dist == 0
    row_electrodes = int(array_width / electrode_dist)

    electrode_pos = np.linspace(
        electrode_dist / 2, array_width - electrode_dist / 2, row_electrodes
    )
    electrode_x, electrode_y = np.meshgrid(electrode_pos, electrode_pos)
    electrode_x, electrode_y = electrode_x.flatten(), electrode_y.flatten()
    return electrode_x - array_width / 2, electrode_y - array_width / 2

def neuron_electrode_dists(x, y, electrode_x, electrode_y):
    # Returns distance matrix (neurons x electrodes)
    neuron_x, neuron_y = (
        np.tile(x, (len(electrode_x), 1)).T,
        np.tile(y, (len(electrode_y), 1)).T,
    )
    electrode_x, electrode_y = np.tile(electrode_x, (len(x), 1)), np.tile(
        electrode_y, (len(y), 1)
    )
    return np.sqrt((electrode_x - neuron_x) ** 2 + (electrode_y - neuron_y) ** 2)


def neuron_spike_array(t, stim_len):
    s = np.zeros((len(t), stim_len))
    for i in range(len(t)):
        for j in range(len(t[i])):
            if t[i][j] < stim_len:
                s[i, t[i][j]] += 1
    return s

def subsample_recordings(rec, electrode_dists):
    min_d = min(electrode_dists)
    ret = []
    for d in electrode_dists:
        if d == min_d:
            ret.append(rec)
            continue
        assert d % min_d == 0
        step = d // min_d
        ret.append(rec[0:-1:step, 0:-1:step, :])
    return ret

def get_electrode_recordings(s, d, radius):
    # The recordings are a sum of all activity in the radius
    rec = np.zeros((d.shape[1], s.shape[1]))
    for i in range(d.shape[1]):
        #rec[i, :] += s[d[:, i] < radius, :].sum(axis=0) previous
        rec[i, :] += s[d[:, i] < radius, :].mean(axis=0)
    rec = rec.reshape(int(np.sqrt(d.shape[1])), int(np.sqrt(d.shape[1])), -1)
    return rec

def get_maximum_recording_array_width(dsv, sheet_name):
    s = dsv.full_datastore.block.annotations['sheet_parameters']
    # PARAMETER FILES SHOULD BE JSONS!!!! 
    # It is ridiculous that I have to do this sort of thing to work with them
    import re
    s = re.sub(r'PyNNDistribution\(.*\)', "''", s)
    s = load_parameters(load_parameters(s)[sheet_name]['recorders'])
    sizes = [load_parameters(s[r])['params']["size"] for r in s if "size" in load_parameters(s[r])['params'].keys()]
    return max(sizes)

def gen_st_array(dsv, s_res=None, t_res=None, array_width=3000, electrode_radius=50):
    x, y = get_s(dsv)
    t = get_t(dsv, t_res=t_res)
    stim_len = ParameterSet(str(dsv.get_stimuli()[0]).replace("MozaikExtended",""))["duration"] // t_res
    electrode_x, electrode_y = electrode_positions(array_width, s_res)
    d = neuron_electrode_dists(x, y, electrode_x, electrode_y)
    s = neuron_spike_array(t, stim_len)
    #electrode_recordings = get_electrode_recordings(s, d, electrode_radius) previous
    electrode_recordings = np.nan_to_num(get_electrode_recordings(s / t_res * 1000, d, electrode_radius))
    return electrode_recordings

def percentile_thresh(A, percentile):
    A_sorted = copy.deepcopy(A)
    A_sorted.sort()
    thresh_idx = int(np.round((A.shape[-1] - 1) * percentile))
    if len(A_sorted.shape) == 1:
        return A_sorted[thresh_idx]
    elif len(A_sorted.shape) == 3:
        return A_sorted[:, :, thresh_idx]
    else:
        return None    

def gen_or_map(dsv, sheet_name, s_res, array_width):
    x, y = get_s(dsv)
    electrode_x, electrode_y = electrode_positions(array_width, s_res)
    d = neuron_electrode_dists(x, y, electrode_x, electrode_y)
    analysis_result = dsv.full_datastore.get_analysis_result(
        identifier="PerNeuronValue",
        value_name="LGNAfferentOrientation",
        sheet_name=sheet_name,
    )
    if len(analysis_result) == 0:
        NeuronAnnotationsToPerNeuronValues(dsv, ParameterSet({})).analyse()
    result = dsv.full_datastore.get_analysis_result(
        identifier="PerNeuronValue",
        value_name="LGNAfferentOrientation",
        sheet_name=sheet_name,
    )[0]
    st_ids = [s for s in dsv.get_segments() if len(s.spiketrains) > 0][
        0
    ].get_stored_spike_train_ids()
    orientations = np.array(result.get_value_by_id(st_ids))
    
    closest_neuron_idx = np.argmin(d,axis=0)#.astype(int)
    or_map_orientations = orientations[closest_neuron_idx]
    square_side = int(np.sqrt(len(or_map_orientations)))
    return or_map_orientations.reshape(square_side,square_side)

def correlation_maps(A,coords):
    Av = (A.transpose((2,0,1)) - A.mean(axis=2)).transpose((1,2,0))
    Avss = (Av * Av).sum(axis=2)    
    results = []
    for i in range(len(coords)):
        x,y = coords[i]
        #print(x,y)
        result = np.matmul(Av,Av[x,y,:])/ np.sqrt(Avss[x,y] * Avss)
        result = np.nan_to_num(result)
        results.append(result)
    # bound the values to -1 to 1 in the event of precision issues
    return np.array(results)

def correlation_or_map_similarity(Cmaps,coords,or_map,size=None):  
    s_map = np.zeros(Cmaps[0].shape)
    or_map_s = np.sin(or_map).flatten()
    or_map_c = np.cos(or_map).flatten()
    
    results = []
    for i in range(len(coords)):
        x,y = coords[i]
        C = Cmaps[i].flatten()
        r_s, _ = scipy.stats.pearsonr(C,or_map_s)
        r_c, _ = scipy.stats.pearsonr(C,or_map_c)
        r_s = np.nan_to_num(r_s)
        r_c = np.nan_to_num(r_c)
        s_map[x,y] = np.sqrt(r_s*r_s + r_c*r_c)
    #t2=time.time()
    #print("Entire similarity map calc time: %.2f s" % (t2-t0))
    return s_map

def resize_arr(A, new_width, new_height):
    A = np.asarray(A)
    shape = list(A.shape)
    shape[0] = new_width
    shape[1] = new_height
    ind = np.indices(shape, dtype=float)
    ind[0] *= (A.shape[0] - 1) / float(new_width - 1)
    ind[1] *= (A.shape[1] - 1) / float(new_height - 1)
    return scipy.ndimage.interpolation.map_coordinates(A, ind, order=1)

def dimensionality(A):
    A = A.reshape((-1,A.shape[2]))
    try:
        cov_mat = numpy.cov(A)
        e = np.linalg.eigvalsh(cov_mat)
    except:
        return -1
    return e.sum()**2 / (e*e).sum() 

def cart_to_pol(coeffs):
    """

    Convert the cartesian conic coefficients, (a, b, c, d, e, f), to the
    ellipse parameters, where F(x, y) = ax^2 + bxy + cy^2 + dx + ey + f = 0.
    The returned parameters are x0, y0, ap, bp, e, phi, where (x0, y0) is the
    ellipse centre; (ap, bp) are the semi-major and semi-minor axes,
    respectively; e is the eccentricity; and phi is the rotation of the semi-
    major axis from the x-axis.

    """

    # We use the formulas from https://mathworld.wolfram.com/Ellipse.html
    # which assumes a cartesian form ax^2 + 2bxy + cy^2 + 2dx + 2fy + g = 0.
    # Therefore, rename and scale b, d and f appropriately.
    a = coeffs[0]
    b = coeffs[1] / 2
    c = coeffs[2]
    d = coeffs[3] / 2
    f = coeffs[4] / 2
    g = coeffs[5]

    den = b**2 - a*c
    if den > 0:
        raise ValueError('coeffs do not represent an ellipse: b^2 - 4ac must'
                         ' be negative!')

    # The location of the ellipse centre.
    x0, y0 = (c*d - b*f) / den, (a*f - b*d) / den

    num = 2 * (a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g)
    fac = np.sqrt((a - c)**2 + 4*b**2)
    # The semi-major and semi-minor axis lengths (these are not sorted).
    ap = np.sqrt(num / den / (fac - a - c))
    bp = np.sqrt(num / den / (-fac - a - c))

    # Sort the semi-major and semi-minor axis lengths but keep track of
    # the original relative magnitudes of width and height.
    width_gt_height = True
    if ap < bp:
        width_gt_height = False
        ap, bp = bp, ap

    # The eccentricity.
    r = (bp/ap)**2
    if r > 1:
        r = 1/r
    e = np.sqrt(1 - r)

    # The angle of anticlockwise rotation of the major-axis from x-axis.
    if b == 0:
        phi = 0 if a < c else np.pi/2
    else:
        phi = np.arctan((2.*b) / (a - c)) / 2
        if a > c:
            phi += np.pi/2
    if not width_gt_height:
        # Ensure that phi is the angle to rotate to the semi-major axis.
        phi += np.pi/2
    phi = np.real(phi) % np.pi

    return x0, y0, ap, bp, e, phi

def fit_ellipse(x, y):
    """

    Fit the coefficients a,b,c,d,e,f, representing an ellipse described by
    the formula F(x,y) = ax^2 + bxy + cy^2 + dx + ey + f = 0 to the provided
    arrays of data points x=[x1, x2, ..., xn] and y=[y1, y2, ..., yn].

    Based on the algorithm of Halir and Flusser, "Numerically stable direct
    least squares fitting of ellipses'.


    """

    D1 = np.vstack([x**2, x*y, y**2]).T
    D2 = np.vstack([x, y, np.ones(len(x))]).T
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    T = -np.linalg.inv(S3) @ S2.T
    M = S1 + S2 @ T
    C = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
    M = np.linalg.inv(C) @ M
    eigval, eigvec = np.linalg.eig(M)
    con = 4 * eigvec[0]* eigvec[2] - eigvec[1]**2
    ak = eigvec[:, np.nonzero(con > 0)[0]]
    return np.concatenate((ak, T @ ak)).ravel()

def get_ellipse_pts(params, npts=100, tmin=0, tmax=2*np.pi):
    """
    Return npts points on the ellipse described by the params = x0, y0, ap,
    bp, e, phi for values of the parametric variable t between tmin and tmax.

    """

    x0, y0, ap, bp, e, phi = params
    # A grid of the parametric variable, t.
    t = np.linspace(tmin, tmax, npts)
    x = x0 + ap * np.cos(t) * np.cos(phi) - bp * np.sin(t) * np.sin(phi)
    y = y0 + ap * np.cos(t) * np.sin(phi) + bp * np.sin(t) * np.cos(phi)
    return x, y

def local_correlation_eccentricity(Cmaps,coords,margin=0):
    eccentricities = []
    for i in range(len(coords)):
        x,y = coords[i]
        if x < margin or y < margin or x > Cmaps.shape[1] - margin or y > Cmaps.shape[2] - margin:
            continue
        C = Cmaps[i]

        # Crop the image to just the ellipse to make it faster!
        lw, num = scipy.ndimage.label(C>0.7)
        lw -= scipy.ndimage.binary_erosion(lw)
        X, Y = np.where(lw)

        try:
            coeffs = fit_ellipse(X, Y)
        except:
            break
        if len(coeffs) == 6:
            x0, y0, ap, bp, e, phi = cart_to_pol(coeffs)
            eccentricities.append(e)

    return np.array(eccentricities)

def dist_or_map(or_map):
    O = or_map.copy()
    O[O>np.pi/2] *= -1
    O[O<0] += np.pi
    return scipy.ndimage.gaussian_filter(O,2)

def rotate_or_map(or_map,angle):
    return (or_map.copy() - angle + np.pi) % np.pi 

def find_local_maxima(arr,min_dist):
    xmax0, ymax0 = scipy.signal.argrelextrema(arr,np.greater_equal,order=min_dist,axis=0)
    xmax1, ymax1 = scipy.signal.argrelextrema(arr,np.greater_equal,order=min_dist,axis=1)
    s1 = {(xmax0[i],ymax0[i],arr[xmax0[i],ymax0[i]]) for i in range(len(xmax0))}
    s2 = {(xmax1[i],ymax1[i],arr[xmax1[i],ymax1[i]]) for i in range(len(xmax1))}
    s = sorted(list(s1 & s2),key=lambda el : el[2],reverse=True)
    i = 0
    while i < len(s):
        j=i+1
        while j < len(s):
            if (s[i][0] - s[j][0])**2 + (s[i][1] - s[j][1])**2 < min_dist**2:
                s.pop(j)
            else:
                j+=1
        i+=1
    return s

def chance_similarity(or_map, s_res, t_res, coords):
    # Calculate similarity chance level
    # Generate 120 s of white noise activity, run through calcium imaging pipeline
    # Calculate correlation maps and their similarity to orientation map
    random_act = np.dstack([np.random.rand(len(or_map.flatten())).reshape((or_map.shape[0],or_map.shape[1])) for i in range(2400)])
    random_act = bandpass_filter(get_calcium_signal(random_act,s_res,t_res),s_res)
    random_corr = correlation_maps(random_act, coords)
    return correlation_or_map_similarity(random_corr, coords, or_map).flatten()

def kohonen_map(Cmaps,or_map):
    som = SOM(1,40,sigma_start=40)
    data = np.array([C.flatten() for C in Cmaps])
    som.fit(data,epochs=1000,verbose=False)
    return vector_readout(som,or_map)

def find_ideal_rotation(ref,rot,rot_min,rot_max,n_steps=1000):
    ref = (ref-ref.min()) / (ref.max() - ref.min())
    rot = (rot-rot_min) / (rot_max - rot_min)
    steps = np.linspace(0,1,n_steps)
    best_step = 0
    best_err = np.inf
    best_rot = None
    for step in steps:
        err = ((ref-np.fmod(rot+step,1))**2).sum()
        if err < best_err:
            best_step = step
            best_err = err
            best_rot = np.fmod(rot+step,1)
    for step in steps:
        err = ((ref-np.fmod(1-rot+step,1))**2).sum()
        if err < best_err:
            best_step = step
            best_err = err
            best_rot = np.fmod(1-rot+step,1)
    return best_rot*(rot_max-rot_min) + rot_min

def vector_readout(som,or_map):
    nodes = som.map.squeeze()
    angles = np.linspace(0,np.pi * 2,nodes.shape[0],endpoint=False)
    v = nodes.T * np.exp(1j * angles)
    v = (np.angle(v.sum(axis=1)).reshape((100,100)) + np.pi) / 2
    return find_ideal_rotation(or_map,v,0,np.pi)

def circ_dist(a, b):
    return np.pi/2 - abs(np.pi/2 - abs(a-b))

def extract_event_indices(
    A, t_res, px_active_p=0.995, event_activity_p=0.8, min_segment_duration=100
):

    thresh = percentile_thresh(A, px_active_p)
    A_active = A.copy().transpose((2, 0, 1))
    A_active[A_active < thresh] = 0
    A_active[A_active >= thresh] = 1
    A_active = A_active.transpose((1, 2, 0))
    A_active_sum = A_active.sum(axis=(0, 1))

    thresh = percentile_thresh(A_active_sum, event_activity_p)
    A_active_zeroed = A_active_sum.copy()
    A_active_zeroed[A_active_zeroed < thresh] = 0

    segment_indices = []
    i = 0
    while i < A.shape[2]:
        if A_active_zeroed[i] > 0:
            segment_max = 0
            segment_max_idx = 0
            segment_start = i
            while A_active_zeroed[i] != 0:
                if A_active_zeroed[i] > segment_max:
                    segment_max_idx = i
                    segment_max = A_active_zeroed[i]
                i += 1
                if i >= A.shape[2] - 1:
                    break
            if i - segment_start > min_segment_duration // t_res:
                segment_indices.append(i)
        i += 1
    
    return segment_indices

def local_maxima_distance_correlations(Cmaps, coords, s_res):
    cs, ds = [], []
    for i in range(Cmaps.shape[0]):
        min_distance_between_maxima = 800 #um
        maxima = np.array(find_local_maxima(Cmaps[i,:,:],min_distance_between_maxima//s_res))[:,:2].astype(int)
        d = np.sqrt(np.sum((maxima - coords[i])**2,axis=1)) * s_res / 1000
        c = Cmaps[i][maxima[:,0],maxima[:,1]]
        order = np.argsort(d)
        ds.append(d[order])
        cs.append(c[order])
        if 0:
            print(d)
            print(c)
            plt.imshow(Cmaps[i],cmap='bwr')
            plt.plot(maxima[:,1],maxima[:,0],'go')
            plt.plot(coords[i][1],coords[i][0],'o',c='lime')
            plt.show()
    return np.array([np.hstack(ds),np.hstack(cs)])

def fit_spatial_scale_correlation(distances, correlations):
    decay_func = lambda x,xi,c0 : np.exp(-x/xi) * (1-c0) + c0
    (xi, c0), _ = curve_fit(
        f=decay_func,
        xdata=distances,
        ydata=correlations,
        p0=[1,0],
    )
    return xi

def single_metric_plot(ax,exp,model,ylim,ylabel,has_legend=False):
    exp = np.array(exp)
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_linewidth(1.5)

    ax.plot(np.ones_like(exp)*1,exp,'s',color='silver')
    ax.plot([1],exp.mean(),'o',color='k')
    ax.plot([1],model,'ro')
    
    if model < ylim[0]:
        ylim[0] = 0.9 * model
    if model > ylim[1]:
        ylim[1] = 1.1 * model
        
    ax.set_ylim(ylim[0],ylim[1])
    ax.set_xlim(0.5,1.5)
    if has_legend:
        ax.legend(["Experiment","Exp. mean","Model"],
                   bbox_to_anchor=(1.85, 1.05),
                   frameon=False)
    ax.set_ylabel(ylabel)
    ax.set_xticks([1],[""])

def double_metric_plot(ax,exp_0,exp_1,model_0,model_1,ylim,ylabel,has_legend=False,x_ticks=["",""]):
    exp_0 = np.array(exp_0)
    exp_1 = np.array(exp_1)
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_linewidth(1.5)

    ax.plot(np.ones_like(exp_0)*1,exp_0,'s',color='silver')
    ax.plot([1],exp_0.mean(),'o',color='k')
    ax.plot([1],model_0,'ro')

    ax.plot(np.ones_like(exp_1)*2,exp_1,'s',color='silver')
    ax.plot([2],exp_1.mean(),'o',color='k')
    ax.plot([2],model_1,'ro')
    
    if model_0 < ylim[0]:
        ylim[0] = 0.9 * model_0
    if model_1 < ylim[0]:
        ylim[0] = 0.9 * model_1

    if model_0 > ylim[1]:
        ylim[1] =1.1 * model_0
    if model_1 > ylim[1]:
        ylim[1] = 1.1 * model_1
        
    ax.set_ylim(ylim[0],ylim[1])
    ax.set_xlim(0.5,2.5)
    if has_legend:
        ax.legend(["Experiment","Exp. mean","Model"],
                   bbox_to_anchor=(1.85, 1.05),
                   frameon=False)
    ax.set_ylabel(ylabel)
    ax.set_xticks([1,2],x_ticks)

def interpolate_2d(arr, target_shape):
    # Create a grid of coordinates for the input array
    N, M = arr.shape
    x_sparse, y_sparse = np.meshgrid(np.arange(M), np.arange(N))

    # Flatten the input array and coordinates
    x_sparse_flat = x_sparse.ravel()
    y_sparse_flat = y_sparse.ravel()
    arr_flat = arr.ravel()

    # Create a grid of coordinates for the target shape
    target_N, target_M = target_shape
    x_dense, y_dense = np.meshgrid(np.linspace(0, M-1, target_M), np.linspace(0, N-1, target_N))

    # Perform 2D interpolation using griddata
    z_dense_interpolated = scipy.interpolate.griddata((x_sparse_flat, y_sparse_flat), arr_flat, (x_dense, y_dense), method='linear')

    return z_dense_interpolated

def radial_mean(image, num_annuli):
    min_im_size = min(image.shape)
    image = image[image.shape[0]//2-min_im_size//2:image.shape[0]//2+min_im_size//2, image.shape[1]//2-min_im_size//2:image.shape[1]//2+min_im_size//2]
    if min_im_size // 2 != num_annuli:
        image = interpolate_2d(image, (num_annuli * 2,num_annuli * 2))
    center_x, center_y = num_annuli, num_annuli
    radius, angle = np.meshgrid(np.arange(num_annuli*2) - center_x, np.arange(num_annuli*2) - center_y, indexing='ij')
    radius = np.sqrt(radius**2 + angle**2)
    
    annulus_radii = np.linspace(0, num_annuli, num_annuli + 1)
    
    # Compute the average magnitude within each annulus
    radial_mean = np.zeros(num_annuli)
    for i in range(num_annuli):
        mask = (radius >= annulus_radii[i]) & (radius < annulus_radii[i + 1])
        radial_mean[i] = np.mean(image[mask])
    return radial_mean

def corr_wavelength(Cmaps,coords,s_res,array_width):
    select_size_um = 2500
    sel_min = select_size_um // 2 // s_res
    sel_max = array_width // s_res - sel_min
    sel_sz =  2 * sel_min
    mean_Cmap = np.array([Cmaps[i,coords[i][0]-sel_sz//2:coords[i][0]+sel_sz//2,coords[i][1]-sel_sz//2:coords[i][1]+sel_sz//2] for i in range(len(coords)) if coords[i][0] > sel_min and coords[i][1] > sel_min and coords[i][0] < sel_max and coords[i][1] < sel_max]).mean(axis=0)
    num_annuli = 200
    rmean = radial_mean(mean_Cmap, num_annuli)
    wavelength = scipy.signal.argrelmax(rmean,order=10)[0][-1] * (sel_sz * s_res // 2) / num_annuli
    return wavelength / 1000

def activity_wavelength(events,s_res):
    autocorrs = np.dstack([scipy.signal.correlate2d(events[:,:,i], events[:,:,i], mode='full', boundary='fill') for i in range(events.shape[2])])
    autocorr_rmeans = np.array([radial_mean(autocorrs[:,:,i], autocorrs.shape[0] // 2) for i in range(autocorrs.shape[2])])
    wls = np.linspace(0,autocorr_rmeans.shape[1]*s_res / 1000,autocorr_rmeans.shape[1])
    indices = []
    for i in range(autocorr_rmeans.shape[0]):
        arm = scipy.signal.argrelmin(autocorr_rmeans[i,:])
        if len(arm) > 0 and len(arm[0]) > 0:
            indices.append(arm[0][0])
    return wls[indices] * 2

def wavelength_hist(wls,bins):
    wls_binned, _ = np.histogram(wls,bins=bins)
    wls_binned = wls_binned.astype(float) / wls_binned.sum().astype(float)
    return wls_binned

def dim_random_sample_events(events, samples=30, repetitions=100):
    return np.array([dimensionality(events[:,:,numpy.random.choice(range(events.shape[2]),samples)]) for i in range(repetitions)])

def plot_hist_comparison(ax,e0,h0,e1,h1,title="",xlabel="",e1_center=False,ylim=[0,0.8]):
    e0, e1, h0, h1 = np.array(e0), np.array(e1), np.array(h0), np.array(h1), 
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_linewidth(1.5)
    ax.bar(e0,h0,width=e0[1]-e0[0],align='center',alpha=0.4,color='k') #edgecolor='black', color='none',lw=2)
    if not e1_center:
        e1 += (e1[1] - e1[0]) / 2
        e1 = e1[:-1]
    ax.bar(e1,h1,width=e1[1]-e1[0],align='center',alpha=0.4,color='r')#, edgecolor='red', color='none',lw=2)
    ax.legend(['Exp.','Model'],frameon=False,handlelength=0.8,loc='upper right')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Probability")
    ax.set_yticks([])
    mmin, mmax = min(e0.min(),e1.min()), max(e0.max(),e1.max())
    mmax = mmax + mmin
    mmin = 0
    ax.set_xlim(mmin,mmax)
    ax.set_xticks([mmin,mmin+(mmax-mmin)/2,mmax])
    ax.set_ylim(ylim[0],ylim[1])
    
from skimage.draw import disk
def correlation_map_similarity(C1,C2,coords,s_res,exclusion_radius=400):
    assert C1[0].shape == C2[0].shape
    s_map = np.zeros(C1[0].shape)
    results = []
    for i in range(len(coords)):
        x,y = coords[i]
        rr, cc = disk((x,y), exclusion_radius//s_res,shape=C1[i].shape)
        inv = np.zeros_like(C1[i])
        inv[rr,cc] = 1
        s_map[x,y], _ = scipy.stats.pearsonr(C1[i][inv < 1],C2[i][inv < 1])
    return s_map

def get_mask(array, radius):
    mask = np.zeros_like(array)
    center_row, center_col = array.shape[0] // 2, array.shape[1] // 2
    rr, cc = disk((array.shape[0] // 2, array.shape[1] // 2), radius)
    mask[rr, cc] = 1
    return mask

def mask_center(A,masks):
    ret = A.copy()
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            for k in range(A.shape[4]):
                ret[i,j,:,:,k][masks[i] > 0] = 0
    return ret

def mask_mean(A,masks):
    ret = np.zeros((A.shape[0],A.shape[1],A.shape[4]))
    for i in range(A.shape[0]):
        mask = masks[i] > 0
        for j in range(A.shape[1]):
            for k in range(A.shape[4]):
                ret[i,j,k] = np.nanmean(A[i,j,:,:,k][mask])
    return ret

def mask_sum(A,masks):
    ret = np.zeros((A.shape[0],A.shape[1],A.shape[4]))
    for i in range(A.shape[0]):
        mask = masks[i] > 0
        for j in range(A.shape[1]):
            for k in range(A.shape[4]):
                ret[i,j,k] = np.nansum(A[i,j,:,:,k][mask])
    return ret

def or_masks(or_map,masks_inv):
    from mozaik.tools.circ_stat import circ_mean
    def circ_dist_simple(a,b):
        return np.pi/2 - abs(np.pi/2 - abs(a-b))
    
    thresh = np.pi / 8
    means = np.array([circ_mean(or_map[mask == 0],axis=0,low=0,high=np.pi)[0] for mask in masks_inv])
    or_masks_close = np.array([np.logical_and(masks_inv[-1], circ_dist_simple(or_map,means[i]) < thresh) for i in range(len(masks_inv))])
    or_masks_far = ([np.logical_and(masks_inv[-1], circ_dist_simple(or_map,means[i]) > np.pi / 2 - thresh) for i in range(len(masks_inv))])
    return or_masks_close, or_masks_far
