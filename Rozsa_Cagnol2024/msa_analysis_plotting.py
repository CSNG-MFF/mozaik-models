from mozaik.analysis.analysis import Analysis
from mozaik.analysis.data_structures import SingleValue, AnalogSignal, PerNeuronValue
from mozaik.analysis.technical import NeuronAnnotationsToPerNeuronValues
import numpy as np
import numpy
import mozaik.tools.units as munits
from neo.core.analogsignal import AnalogSignal as NeoAnalogSignal
import quantities as qt
from mozaik.storage import queries
from mozaik.storage.datastore import DataStoreView  
import scipy
from skimage.draw import disk
import copy
import copy
from som import SOM
from mozaik.visualization.plotting import Plotting
import pylab
from parameters import ParameterSet
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
import logging
import scipy
import skimage
from mozaik.tools.distribution_parametrization import load_parameters
import os
from sklearn.decomposition import PCA
from mozaik.tools.mozaik_parametrized import MozaikParametrized

class RecordingArrayAnalysis(Analysis):

    required_parameters = ParameterSet(
        {
            "s_res": int,  # Space resolution (bin size in um) of orientation map
            "array_width": float, # Electrode array width (um)
        }
    )
    
    def electrode_positions(self, array_width, s_res):
        assert array_width % s_res == 0
        row_electrodes = int(array_width / s_res)

        electrode_pos = np.linspace(
            s_res / 2, array_width - s_res / 2, row_electrodes
        )
        electrode_x, electrode_y = np.meshgrid(electrode_pos, electrode_pos)
        electrode_x, electrode_y = electrode_x.flatten(), electrode_y.flatten()
        return electrode_x - array_width / 2, electrode_y - array_width / 2

    def get_st_ids(self,dsv):
        assert len(dsv.sheets()) == 1, "Analysis needs to be run on a single sheet!"
        return [s for s in dsv.get_segments() if len(s.spiketrains) > 0][0].get_stored_spike_train_ids()

    def get_s(self, dsv, s_res=None, neuron_ids=None):
        if s_res == None:
            s_res = 1
        if neuron_ids is None:
            neuron_ids = self.get_st_ids(dsv)
        sheet = dsv.sheets()[0]
        pos = dsv.get_neuron_positions()[sheet]
        posx = np.round((pos[0, dsv.get_sheet_indexes(sheet, neuron_ids)] / s_res * 1000)).astype(int)
        posy = np.round((pos[1, dsv.get_sheet_indexes(sheet, neuron_ids)] / s_res * 1000)).astype(int)
        return posx, posy

    def neuron_electrode_dists(self, x, y, electrode_x, electrode_y):
        # Returns distance matrix (neurons x electrodes)
        neuron_x, neuron_y = (
            np.tile(x, (len(electrode_x), 1)).T,
            np.tile(y, (len(electrode_y), 1)).T,
        )
        electrode_x, electrode_y = np.tile(electrode_x, (len(x), 1)), np.tile(
            electrode_y, (len(y), 1)
        )
        return np.sqrt((electrode_x - neuron_x) ** 2 + (electrode_y - neuron_y) ** 2)
    
    def perform_analysis(self):
        self.tags.extend(["s_res: %d" % self.parameters.s_res,"array_width: %d" % self.parameters.array_width])

class RecordingArrayOrientationMap(RecordingArrayAnalysis):

    def gen_or_map(self, dsv, sheet, s_res, array_width):
        x, y = self.get_s(dsv)
        electrode_x, electrode_y = self.electrode_positions(array_width, s_res)
        d = self.neuron_electrode_dists(x, y, electrode_x, electrode_y)
        analysis_result = dsv.full_datastore.get_analysis_result(
            sheet_name=sheet,
            identifier="PerNeuronValue",
            value_name="LGNAfferentOrientation",
        )
        if len(analysis_result) == 0:
            NeuronAnnotationsToPerNeuronValues(dsv, ParameterSet({})).analyse()
        result = dsv.full_datastore.get_analysis_result(
            sheet_name=sheet,
            identifier="PerNeuronValue",
            value_name="LGNAfferentOrientation",
        )[0]
        st_ids = [s for s in dsv.get_segments() if len(s.spiketrains) > 0][
            0
        ].get_stored_spike_train_ids()
        orientations = np.array(result.get_value_by_id(st_ids))

        closest_neuron_idx = np.argmin(d,axis=0)#.astype(int)
        or_map_orientations = orientations[closest_neuron_idx]
        square_side = int(np.sqrt(len(or_map_orientations)))
        return or_map_orientations.reshape(square_side,square_side)

    def perform_analysis(self):
        super().perform_analysis()
        for sheet in self.datastore.sheets():
            dsv = queries.param_filter_query(self.datastore, sheet_name=sheet)
            or_map = self.gen_or_map(dsv, sheet, self.parameters.s_res, self.parameters.array_width)  
            self.datastore.full_datastore.add_analysis_result(
                SingleValue(
                    value=or_map,
                    value_units=qt.radians,
                    value_name="orientation map",
                    tags=self.tags,
                    sheet_name=sheet,
                    analysis_algorithm=self.__class__.__name__,
                )
            )
    
class RecordingArrayTimecourse(RecordingArrayAnalysis):
    
    required_parameters = ParameterSet(
        {
            "t_res": int,  # Time resolution (bin size in ms) of activity maps
            "s_res": int,  # Space resolution (bin size in um) of activity maps
            "array_width": float, # Electrode array width (um)
            "electrode_radius": float, # Electrode radius (um)
        }
    )
    
    def get_t(self, seg, t_res=None):
        if t_res == None:
            t_res = 1
        return [list((st.magnitude / t_res).astype(int)) for st in seg.get_spiketrains()]
    
    def neuron_spike_array(self, t, stim_len):
        s = np.zeros((len(t), int(stim_len)))
        for i in range(len(t)):
            for j in range(len(t[i])):
                if t[i][j] < stim_len:
                    s[i, t[i][j]] += 1
        return s

    def get_electrode_recordings(self, s, d, radius):
        # The recordings are a mean of all activity in the electrode radius
        rec = np.zeros((d.shape[1], s.shape[1]))
        for i in range(d.shape[1]):
            rec[i, :] += s[d[:, i] < radius, :].mean(axis=0)
        rec = rec.reshape(int(np.sqrt(d.shape[1])), int(np.sqrt(d.shape[1])), -1)
        return rec
    
    def perform_analysis(self):
        super().perform_analysis()
        self.tags.extend(["t_res: %d" % self.parameters.t_res, "electrode_radius: %d" % self.parameters.electrode_radius])
        for sheet in self.datastore.sheets():
            dsv = queries.param_filter_query(self.datastore, sheet_name=sheet)
            x, y = self.get_s(dsv)
            segs, stims = dsv.get_segments(), dsv.get_stimuli()
            for i in range(len(segs)):
                if len(segs[i].spiketrains) < 1:
                    continue
                t = self.get_t(segs[i], t_res=self.parameters.t_res)
                stim_len = ParameterSet(str(stims[i]).replace("MozaikExtended",""))["duration"] // self.parameters.t_res
                electrode_x, electrode_y = self.electrode_positions(self.parameters.array_width, self.parameters.s_res)
                d = self.neuron_electrode_dists(x, y, electrode_x, electrode_y)
                s = self.neuron_spike_array(t, stim_len)
                electrode_recordings = np.nan_to_num(self.get_electrode_recordings(s / self.parameters.t_res * 1000, d, self.parameters.electrode_radius))
                electrode_recordings = electrode_recordings.transpose((2,0,1)) # Time should be first dimension
                self.datastore.full_datastore.add_analysis_result(
                    AnalogSignal(
                        NeoAnalogSignal(electrode_recordings, t_start=0, sampling_period=self.parameters.t_res*qt.ms,units=munits.spike / qt.s),
                        y_axis_name="recording array timecourse",
                        y_axis_units=munits.spike / qt.s,
                        tags=self.tags,
                        sheet_name=sheet,
                        stimulus_id=str(stims[i]),
                        analysis_algorithm=self.__class__.__name__,
                    )
                )

class SimulatedCalciumSignal(Analysis):
    
    required_parameters = ParameterSet(
        {
            "spatial_profile_path": str,  # numpy array
            "reference_dsv": DataStoreView,
        }
    )
    
    def calcium_light_spread_kernel(self,spatial_profile_path,s_res, array_width):
        x,y,I = np.load(spatial_profile_path)
        x_t, y_t = np.meshgrid(np.arange(-array_width//2,array_width//2+1,s_res),np.arange(-array_width//2,array_width//2+1,s_res))
        return scipy.interpolate.griddata((x.flatten(), y.flatten()), I.flatten(), (x_t, y_t), method='cubic')
    
    def t_kernel(self, t_res,length_ms=5000):
        # Based on https://doi.org/10.3389/fncir.2013.00201
        tau_on = 10 # ms rise time of calcium response
        tau_off = 1000 # ms fall time of calcium response
        if length_ms < 10 * t_res:
            length_ms = 10*t_res

        # We ignore the rise time for the moment as it is 100x smaller than fall time
        return np.exp(-np.linspace(0,length_ms,length_ms//t_res)/tau_off)

    def get_calcium_signal(self, A_in, t_res, s_res, array_width):
        A = A_in.copy()
        t_ker = self.t_kernel(t_res)
        A_t = scipy.ndimage.convolve1d(A, t_ker, axis=0, mode='wrap', origin=-(len(t_ker)//2))
        s_ker = self.calcium_light_spread_kernel(self.parameters.spatial_profile_path,s_res,array_width)
        A = np.stack([scipy.signal.convolve(A_t[i,:,:],s_ker,mode='same') for i in range(A.shape[0])])
        return A

    def normalize_calcium_signal(self, A, A_ref):
        tiling = np.ones_like(A.shape)
        tiling[0] = A.shape[0]
        f0 = np.tile(A_ref.mean(axis=0)[np.newaxis,...],tiling)
        return (A - f0) / f0
    
    def tag_value(self, tag, tags):
        filtered_tags = [t.split(":")[-1] for t in tags if ":".join(t.split(":")[:-1]) == tag]
        assert len(filtered_tags) == 1, "Duplicate tags are not allowed!"
        return eval(filtered_tags[0])
    
    def perform_analysis(self):
        datastore_equal_to_ref_dsv = self.datastore == self.parameters.reference_dsv
        for sheet in self.datastore.sheets():
            dsv = queries.param_filter_query(self.datastore, sheet_name=sheet,analysis_algorithm="RecordingArrayTimecourse")
            for anasig in dsv.get_analysis_result():                
                t_res, s_res, array_width = self.tag_value("t_res",anasig.tags), self.tag_value("s_res",anasig.tags), self.tag_value("array_width",anasig.tags)
                assert anasig.analog_signal.ndim == 3, "Analog signal should have 1 temporal and 2 spatial dimensions!"
                calcium_signal = self.get_calcium_signal(anasig.analog_signal, t_res, s_res, array_width)
                if not datastore_equal_to_ref_dsv:
                    ref_dsv = queries.param_filter_query(self.parameters.reference_dsv, sheet_name=sheet,y_axis_name="Calcium imaging signal")
                    assert len(ref_dsv.analysis_results) == 1, "Reference datastore must contain exactly 1 Calcium imaging signal per sheet, contains %d" % len(ref_dsv.analysis_results)
                    ref_anasig = ref_dsv.analysis_results[0]
                    assert ref_anasig.analog_signal.ndim == 3, "Reference analog signal should have 1 temporal and 2 spatial dimensions!"
                    assert t_res == self.tag_value("t_res",ref_anasig.tags) and s_res == self.tag_value("s_res",ref_anasig.tags), "Reference and analysis analog signal need to be of the same spatial and temporal resolution!"
                    ref_calcium_signal = np.array(ref_anasig.analog_signal)
                else:
                    ref_calcium_signal = calcium_signal
                calcium_signal_normalized = self.normalize_calcium_signal(calcium_signal, ref_calcium_signal)
                common_params = {
                    "y_axis_units": qt.dimensionless,
                    "tags": anasig.tags,
                    "sheet_name": sheet,
                    "stimulus_id": anasig.stimulus_id,
                    "analysis_algorithm": self.__class__.__name__,
                }
                self.datastore.full_datastore.add_analysis_result(
                    AnalogSignal(
                        NeoAnalogSignal(calcium_signal, t_start=0, sampling_period=t_res*qt.ms,units=qt.dimensionless),
                        y_axis_name="Calcium imaging signal",
                        **common_params,
                    )
                )
                self.datastore.full_datastore.add_analysis_result(
                    AnalogSignal(
                        NeoAnalogSignal(calcium_signal_normalized, t_start=0, sampling_period=t_res*qt.ms,units=qt.dimensionless),
                        y_axis_name="Calcium imaging signal (normalized)",
                        **common_params,
                    )
                )

class GaussianBandpassFilter(Analysis):
    
    required_parameters = ParameterSet(
        {
            "highpass_sigma_um": float,
            "lowpass_sigma_um": float,
        }
    )

    def tag_value(self, tag, tags):
        if len(tags) == 0:
            raise RuntimeError("No tags on recording!")
        filtered_tags = [t.split(":")[-1] for t in tags if ":".join(t.split(":")[:-1]) == tag]
        assert len(filtered_tags) == 1, "Duplicate tags are not allowed!"
        return eval(filtered_tags[0])
    
    def bandpass_filter(self, A_in, high_sigma_um, low_sigma_um, s_res):
        hp_sigma = high_sigma_um / s_res
        lp_sigma = low_sigma_um / s_res

        # Band-pass filtering
        filt = np.zeros(len(A_in.shape))
        filt[1:] = lp_sigma
        flp = scipy.ndimage.gaussian_filter(A_in, filt)
        filt[1:] = hp_sigma
        fhp = scipy.ndimage.gaussian_filter(flp, filt)
        fbp = flp - fhp 

        return fbp
    
    def perform_analysis(self):
        for sheet in self.datastore.sheets():
            dsv = queries.param_filter_query(self.datastore, sheet_name=sheet,identifier="AnalogSignal")
            for anasig in dsv.get_analysis_result():
                s_res, t_res = self.tag_value("s_res",anasig.tags), self.tag_value("t_res",anasig.tags)
                bpf = self.bandpass_filter(anasig.analog_signal, self.parameters.highpass_sigma_um, self.parameters.lowpass_sigma_um, s_res)
                self.datastore.full_datastore.add_analysis_result(
                    AnalogSignal(
                        NeoAnalogSignal(bpf, t_start=0, sampling_period=t_res*qt.ms,units=qt.dimensionless),
                        y_axis_units=qt.dimensionless,
                        tags=anasig.tags,
                        sheet_name=sheet,
                        stimulus_id=anasig.stimulus_id,
                        analysis_algorithm=self.__class__.__name__,
                    )
                )

class CorrelationMaps(Analysis):
    required_parameters = ParameterSet({})
    def correlation_maps(self,A):
        Av = (A - A.mean(axis=0)).transpose(1,2,0)
        Avss = (Av * Av).sum(axis=2)
        return np.array([np.nan_to_num(np.matmul(Av,Av[x,y,:])/ np.sqrt(Avss[x,y] * Avss)) for x in range(Av.shape[0]) for y in range(Av.shape[1])])
    
    def perform_analysis(self):
        for sheet in self.datastore.sheets():
            dsv = queries.param_filter_query(self.datastore, sheet_name=sheet,identifier="AnalogSignal")
            for anasig in dsv.get_analysis_result():
                assert anasig.analog_signal.ndim == 3, "Signal must have 1 temporal and 2 spatial dimensions!"
                cm = self.correlation_maps(np.array(anasig.analog_signal))
                self.datastore.full_datastore.add_analysis_result(
                    SingleValue(
                        value=cm,
                        value_units=qt.dimensionless,
                        value_name="correlation map",
                        tags=[t for t in anasig.tags if "t_res" not in t],
                        sheet_name=sheet,
                        stimulus_id=anasig.stimulus_id,
                        analysis_algorithm=self.__class__.__name__,
                    )
                )

class OrientationMapSimilarity(Analysis):
    required_parameters = ParameterSet({
        "or_map_dsv": DataStoreView,
    })

    def or_map_similarity(self,A,or_map):
        s_map = np.zeros(A.shape[0])
        or_map_s = np.sin(or_map).flatten()
        or_map_c = np.cos(or_map).flatten()
        for i in range(A.shape[0]):
            r_s = np.nan_to_num(scipy.stats.pearsonr(A[i,:,:].flatten(),or_map_s)[0])
            r_c = np.nan_to_num(scipy.stats.pearsonr(A[i,:,:].flatten(),or_map_c)[0])
            s_map[i] = np.sqrt(r_s*r_s + r_c*r_c)
        return s_map
        
    def perform_analysis(self):
        or_map_dsv_res = queries.param_filter_query(self.parameters.or_map_dsv,analysis_algorithm="RecordingArrayOrientationMap").get_analysis_result()
        assert len(or_map_dsv_res) == 1, "or_map_dsv can only contain 1 RecordingArrayOrientationMap per sheet, contains %d" % len(or_map_dsv_res)
        or_map = or_map_dsv_res[0].value
        for sheet in self.datastore.sheets():
            dsv = queries.param_filter_query(self.datastore, sheet_name=sheet)
            for res in dsv.get_analysis_result():
                tags = res.tags
                stimulus_id = res.stimulus_id
                if type(res) == AnalogSignal:
                    res = np.array(res.analog_signal)
                elif type(res) == SingleValue:
                    res = np.array(res.value)
                assert res.ndim == 3, "Signal must have 1 arbitrary and 2 spatial dimensions!"
                self.datastore.full_datastore.add_analysis_result(
                    SingleValue(
                        value=self.or_map_similarity(res,or_map),
                        value_units=qt.dimensionless,
                        value_name="orientation map similarity",
                        tags=tags,
                        sheet_name=sheet,
                        stimulus_id=stimulus_id,
                        analysis_algorithm=self.__class__.__name__,
                    )
                )

class CorrelationMapSimilarity(Analysis):
    required_parameters = ParameterSet({
        "corr_map_dsv": DataStoreView,
        "exclusion_radius": float,
    })

    def tag_value(self, tag, tags):
        if len(tags) == 0:
            raise RuntimeError("No tags on recording!")
        filtered_tags = [t.split(":")[-1] for t in tags if ":".join(t.split(":")[:-1]) == tag]
        assert len(filtered_tags) == 1, "Duplicate tags are not allowed!"
        return eval(filtered_tags[0])
    
    def correlation_map_similarity(self,C1,C2,s_res,exclusion_radius=400):
        assert C1[0].shape == C2[0].shape
        s_map = np.zeros(C1.shape[0])
        
        for i in range(C1.shape[0]):
            x,y = i // C1.shape[1], i % C1.shape[1]
            rr, cc = disk((x,y), exclusion_radius // s_res,shape=C1[i].shape)
            inv = np.zeros_like(C1[i])
            inv[rr,cc] = 1
            s_map[i], _ = scipy.stats.pearsonr(C1[i][inv < 1],C2[i][inv < 1])
        return s_map
        
    def perform_analysis(self):
        c1_dsv = queries.param_filter_query(self.datastore,analysis_algorithm="CorrelationMaps")
        c2_dsv = queries.param_filter_query(self.parameters.corr_map_dsv,analysis_algorithm="CorrelationMaps")

        # Find all unique pairs of correlation maps        
        unique_pairs = []
        for sheet_1 in c1_dsv.sheets():
            for sheet_2 in c2_dsv.sheets():
                c1_dsv_sh = queries.param_filter_query(c1_dsv,sheet_name=sheet_1)
                c2_dsv_sh = queries.param_filter_query(c2_dsv,sheet_name=sheet_2)
                for c1_res in c1_dsv_sh.get_analysis_result():
                    for c2_res in c2_dsv_sh.get_analysis_result():
                        if c1_res == c2_res:
                            continue
                        unique_pairs.append([c1_res,c2_res])

        unique_pairs = [sorted(p,key=lambda s: str(s)) for p in unique_pairs]
        unique_pairs = {str(p) : p for p in unique_pairs}
        unique_pairs = [unique_pairs[k] for k in set(unique_pairs.keys())]

        for c1_res, c2_res in unique_pairs:
            assert type(c1_res) == type(c2_res) == SingleValue 
            s_res = self.tag_value("s_res", c1_res.tags)
            assert c1_res.value.ndim == 3 and c2_res.value.ndim == 3, "Correlation maps must have 1 arbitrary and 2 spatial dimensions!"
            self.datastore.full_datastore.add_analysis_result(
                SingleValue(
                    value=self.correlation_map_similarity(c1_res.value, c2_res.value, s_res,self.parameters.exclusion_radius),
                    value_units=qt.dimensionless,
                    value_name="correlation map similarity",
                    tags=c1_res.tags,
                    stimulus_id=c1_res.stimulus_id,
                    analysis_algorithm=self.__class__.__name__,
                )
            )


class Smith_2018_Mulholland_2021_2024_spont_analyses(Analysis):
    required_parameters = ParameterSet({
    })

    def tag_value(self, tag, tags):
        if len(tags) == 0:
            raise RuntimeError("No tags on recording!")
        filtered_tags = [t.split(":")[-1] for t in tags if ":".join(t.split(":")[:-1]) == tag]
        if len(filtered_tags) == 0:
            return None
        assert len(filtered_tags) == 1, "Duplicate tags are not allowed!"
        return eval(filtered_tags[0])

    def fit_spatial_scale_correlation(self, distances, correlations):
        decay_func = lambda x,xi,c0 : np.exp(-x/xi) * (1-c0) + c0
        (xi, c0), _ = scipy.optimize.curve_fit(
            f=decay_func,
            xdata=distances,
            ydata=correlations,
            p0=[1,0],
        )
        return xi
        
    def find_local_maxima(self, arr, min_dist):
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
        # Calculate self.tag_value("s_res",anasig.tags) maps and their similarity to orientation map
        random_act = np.dstack([np.random.rand(len(or_map.flatten())).reshape((or_map.shape[0],or_map.shape[1])) for i in range(2400)])
        random_act = bandpass_filter(get_calcium_signal(random_act,s_res,t_res),s_res)
        random_corr = correlation_maps(random_act, coords)
        return correlation_or_map_similarity(random_corr, coords, or_map).flatten()
    
    def local_maxima_distance_correlations(self, Cmaps,  s_res):
        cs, ds = [], []
        for i in range(Cmaps.shape[0]):
            coords = np.array([i // Cmaps.shape[1], i % Cmaps.shape[1]])
            min_distance_between_maxima = 800 #um
            maxima = np.array(self.find_local_maxima(Cmaps[i,:,:],min_distance_between_maxima//s_res))[:,:2].astype(int)
            d = np.sqrt(np.sum((maxima - coords)**2,axis=1)) * s_res / 1000
            c = Cmaps[i][maxima[:,0],maxima[:,1]]
            order = np.argsort(d)
            ds.append(d[order])
            cs.append(c[order])
        return np.array([np.hstack(ds),np.hstack(cs)])

    def interpolate_2d(self, arr, target_shape):
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
    
    def radial_mean(self, image, num_annuli):
        min_im_size = min(image.shape)
        image = image[image.shape[0]//2-min_im_size//2:image.shape[0]//2+min_im_size//2, image.shape[1]//2-min_im_size//2:image.shape[1]//2+min_im_size//2]
        if min_im_size // 2 != num_annuli:
            image = self.interpolate_2d(image, (num_annuli * 2,num_annuli * 2))
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
        
    def corr_wavelength(self, Cmaps,s_res,array_width):
        select_size_um = 2500
        sel_min = select_size_um // 2 // s_res
        sel_max = array_width // s_res - sel_min
        sel_sz =  2 * sel_min
        coords = [(x,y) for x in range(Cmaps.shape[1]) for y in range(Cmaps.shape[2])]
        mean_Cmap = np.array([Cmaps[i,coords[i][0]-sel_sz//2:coords[i][0]+sel_sz//2,coords[i][1]-sel_sz//2:coords[i][1]+sel_sz//2] for i in range(len(coords)) if coords[i][0] > sel_min and coords[i][1] > sel_min and coords[i][0] < sel_max and coords[i][1] < sel_max]).mean(axis=0)
        num_annuli = 200
        rmean = self.radial_mean(mean_Cmap, num_annuli)
        wavelength = scipy.signal.argrelmax(rmean,order=10)[0][-1] * (sel_sz * s_res // 2) / num_annuli
        return wavelength / 1000

    def activity_wavelength(self,autocorr_rmeans,s_res):
        wls = np.linspace(0,autocorr_rmeans.shape[1]*s_res / 1000,autocorr_rmeans.shape[1])
        indices = []
        for i in range(autocorr_rmeans.shape[0]):
            arm = scipy.signal.argrelmin(autocorr_rmeans[i,:])
            if len(arm) > 0 and len(arm[0]) > 0:
                indices.append(arm[0][0])
        return wls[indices] * 2
    
    def autocorrelation_radial_mean(self,events):
        boundary = 5 # cut off the sides to avoid errors from autocorrelation wrap
        events = events[:,boundary:-boundary,boundary:-boundary]
        events = (events.transpose((1,2,0)) / np.linalg.norm(events,axis=(1,2)))
        autocorrs = np.stack([scipy.signal.correlate2d(events[:,:,i], events[:,:,i],mode='full',boundary='wrap') for i in range(events.shape[2])])
        autocorrs = autocorrs[:,events.shape[0]//4:-events.shape[0]//4,events.shape[1]//4:-events.shape[1]//4]
        return np.array([self.radial_mean(autocorrs[i,:,:], autocorrs.shape[1] // 2) for i in range(autocorrs.shape[0])])
    
    def modularity(self,autocorr):
        modularity = []
        for i in range(autocorr.shape[0]):
            armin = scipy.signal.argrelmin(autocorr[i,:],order=5)[0][0] # First minimum
            armax = scipy.signal.argrelmax(autocorr[i,armin:],order=5)[0][0] + armin # First maximum after the minimum
            modularity.append(np.abs(autocorr[i,armin] - autocorr[i,armax]))
        return np.array(modularity)
    
    def cart_to_pol(self, coeffs):
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
    
    def fit_ellipse(self, x, y):
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
    
    def local_correlation_eccentricity(self, Cmaps, margin=0):
        eccentricities = []
        for i in range(Cmaps.shape[0]):
            x,y = Cmaps.shape[0] // Cmaps.shape[1], Cmaps.shape[0] % Cmaps.shape[1]
            if x < margin or y < margin or x > Cmaps.shape[1] - margin or y > Cmaps.shape[2] - margin:
                continue
            C = Cmaps[i]
    
            # Crop the image to just the ellipse to make it faster!
            lw, num = scipy.ndimage.label(C>0.7)
            lw -= scipy.ndimage.binary_erosion(lw)
            X, Y = np.where(lw)
    
            try:
                coeffs = self.fit_ellipse(X, Y)
            except:
                break
            if len(coeffs) == 6:
                x0, y0, ap, bp, e, phi = self.cart_to_pol(coeffs)
                eccentricities.append(e)
    
        return np.array(eccentricities)

    def dim_random_sample_events(self, events, samples=30, repetitions=100):
        return np.array([self.dimensionality(events[np.random.choice(range(events.shape[0]),samples),:,:]) for i in range(repetitions)])

    def percentile_thresh(self, A, percentile):
        A_sorted = copy.deepcopy(A)
        A_sorted.sort(axis=0)
        thresh_idx = int(np.round((A.shape[0] - 1) * percentile))
        if len(A_sorted.shape) == 1:
            return A_sorted[thresh_idx]
        elif len(A_sorted.shape) == 3:
            return A_sorted[thresh_idx, :, :]
        else:
            return None

    def extract_event_indices(
        self, A, t_res, px_active_p=0.995, event_activity_p=0.8, min_segment_duration=100
    ):
        thresh = self.percentile_thresh(A, px_active_p)
        A_active = A.copy()
        A_active[A_active < thresh] = 0
        A_active[A_active >= thresh] = 1
        A_active_sum = A_active.sum(axis=(1, 2))
    
        thresh = self.percentile_thresh(A_active_sum, event_activity_p)
        A_active_zeroed = A_active_sum.copy()
        A_active_zeroed[A_active_zeroed < thresh] = 0
    
        segment_indices = []
        i = 0
        while i < A.shape[0]:
            if A_active_zeroed[i] > 0:
                segment_max = 0
                segment_max_idx = 0
                segment_start = i
                while A_active_zeroed[i] != 0:
                    if A_active_zeroed[i] > segment_max:
                        segment_max_idx = i
                        segment_max = A_active_zeroed[i]
                    i += 1
                    if i >= A.shape[0] - 1:
                        break
                if i - segment_start > min_segment_duration // t_res:
                    segment_indices.append(i)
            i += 1
        
        return segment_indices
    
    def kohonen_map(self,Cmaps,or_map):
        som = SOM(1,40,sigma_start=40)
        data = np.array([C.flatten() for C in Cmaps])
        som.fit(data,epochs=1000,verbose=False)
        return self.vector_readout(som,or_map)
    
    def find_ideal_rotation(self,ref,rot,rot_min,rot_max,n_steps=1000):
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
    
    def vector_readout(self,som,or_map):
        nodes = som.map.squeeze()
        angles = np.linspace(0,np.pi * 2,nodes.shape[0],endpoint=False)
        v = nodes.T * np.exp(1j * angles)
        v = (np.angle(v.sum(axis=1)).reshape((100,100)) + np.pi) / 2
        return self.find_ideal_rotation(or_map,v,0,np.pi)
    
    def circ_dist(a, b):
        return np.pi/2 - abs(np.pi/2 - abs(a-b))

    def resize_arr(self, A, new_width, new_height):
        A = np.asarray(A)
        shape = list(A.shape)
        shape[1] = new_width
        shape[2] = new_height
        ind = np.indices(shape, dtype=float)
        ind[1] *= (A.shape[1] - 1) / float(new_width - 1)
        ind[2] *= (A.shape[2] - 1) / float(new_height - 1)
        return scipy.ndimage.interpolation.map_coordinates(A, ind, order=1)
        
    def dimensionality(self, A):
        A = A.transpose((1,2,0))
        A = A.reshape((-1,A.shape[2]))
        try:
            cov_mat = numpy.cov(A)
            e = np.linalg.eigvalsh(cov_mat)
        except Exception as e:
            print(e)
            return -1
        return e.sum()**2 / (e*e).sum() 
    
    def perform_analysis(self):
        r = {
            "A_exc_calcium": queries.param_filter_query(self.datastore,y_axis_name="Calcium imaging signal (normalized)",sheet_name="V1_Exc_L2/3",st_name='InternalStimulus'),
            "A_inh_calcium": queries.param_filter_query(self.datastore,y_axis_name="Calcium imaging signal (normalized)",sheet_name="V1_Inh_L2/3",st_name='InternalStimulus'),
            "A_exc_bandpass": queries.param_filter_query(self.datastore,analysis_algorithm="GaussianBandpassFilter",sheet_name="V1_Exc_L2/3",st_name='InternalStimulus'),
            "A_inh_bandpass": queries.param_filter_query(self.datastore,analysis_algorithm="GaussianBandpassFilter",sheet_name="V1_Inh_L2/3",st_name='InternalStimulus'),
            # TODO: Maybe add case for raw correlation maps?
            "Cmaps_exc": queries.param_filter_query(self.datastore,analysis_algorithm="CorrelationMaps",sheet_name="V1_Exc_L2/3",st_name='InternalStimulus'),
            "Cmaps_inh": queries.param_filter_query(self.datastore,analysis_algorithm="CorrelationMaps",sheet_name="V1_Inh_L2/3",st_name='InternalStimulus'),
            "or_map": queries.param_filter_query(self.datastore,analysis_algorithm="RecordingArrayOrientationMap",sheet_name="V1_Exc_L2/3"),
        }

        tags = r["A_exc_calcium"].get_analysis_result()[0].tags
        stimulus_id = r["A_exc_calcium"].get_analysis_result()[0].stimulus_id
        s_res, t_res, array_width = self.tag_value("s_res", tags), self.tag_value("t_res", tags), self.tag_value("array_width", tags)
        
        for k in r.keys():
            ar = r[k].get_analysis_result()
            assert len(ar) == 1, "Can only contain single analysis result per sheet, contains %d" % len(ar)
            assert s_res == self.tag_value("s_res", ar[0].tags) and array_width == self.tag_value("array_width", ar[0].tags)
            if self.tag_value("t_res", ar[0].tags) is not None:
                assert t_res == self.tag_value("t_res", ar[0].tags)
            r[k] = ar[0].value if type(ar[0]) == SingleValue else ar[0].analog_signal
            r[k] = np.array(r[k])

        results = {}
    
        # Smith 2018
        event_idx_exc = self.extract_event_indices(r["A_exc_calcium"],t_res)
        event_idx_inh = self.extract_event_indices(r["A_inh_calcium"],t_res)
        small_spont_exc = self.resize_arr(r["A_exc_bandpass"], 50, 50)
        small_spont_inh = self.resize_arr(r["A_inh_bandpass"], 50, 50)
        results["Dimensionality"] = {"V1_Exc_L2/3" : self.dimensionality(small_spont_exc)}
        autocorr_radial_means = self.autocorrelation_radial_mean(r["A_exc_bandpass"][event_idx_exc,:,:])
        results["Event activity wavelength"] =  {"V1_Exc_L2/3" : self.activity_wavelength(autocorr_radial_means,s_res)}
        results["Event activity modularity"] =  {"V1_Exc_L2/3" : self.modularity(autocorr_radial_means)}                               
        
        results["Local maxima distance correlation"] = {"V1_Exc_L2/3" : self.local_maxima_distance_correlations(r["Cmaps_exc"], s_res),
                                                        "V1_Inh_L2/3" : self.local_maxima_distance_correlations(r["Cmaps_inh"], s_res)}
        results["Spatial scale of correlation"] = {"V1_Exc_L2/3" : self.fit_spatial_scale_correlation(results["Local maxima distance correlation"]["V1_Exc_L2/3"][0,:],
                                                                                                      results["Local maxima distance correlation"]["V1_Exc_L2/3"][1,:])}

        # Mulholland 2021
        results["Correlation map wavelength"] = {"V1_Exc_L2/3" : self.corr_wavelength(r["Cmaps_exc"],s_res,array_width),
                                                 "V1_Inh_L2/3" : self.corr_wavelength(r["Cmaps_inh"],s_res,array_width)}
        results["Local correlation eccentricity"] =  {"V1_Exc_L2/3" : self.local_correlation_eccentricity(r["Cmaps_exc"]),
                                                      "V1_Inh_L2/3" : self.local_correlation_eccentricity(r["Cmaps_inh"])}
        results["Dimensionality (random sampled events)"] =  {"V1_Exc_L2/3" : self.dim_random_sample_events(small_spont_exc),
                                                              "V1_Inh_L2/3" : self.dim_random_sample_events(small_spont_inh)}

        # Kohonen map
        results["Kohonen map"] = {"V1_Exc_L2/3" : self.kohonen_map(r["A_exc_bandpass"],r["or_map"])}
        tags = [t for t in tags if "t_res:" not in t]
        
        for name, vv in results.items():
            for sheet, value in vv.items():
                self.datastore.full_datastore.add_analysis_result(
                    SingleValue(
                        stimulus_id=stimulus_id,
                        value=value,
                        value_units=qt.dimensionless,
                        value_name=name,
                        tags=tags,
                        sheet_name=sheet,
                        analysis_algorithm=self.__class__.__name__,
                    )
                )

class Kenet_2003(Analysis):
    required_parameters = ParameterSet({
    "fullfield_gratings_dsv": DataStoreView,
    })

    def perform_analysis(self):
        sheets = self.datastore.sheets()
        for sheet in self.datastore.sheets():
            dsv = queries.param_filter_query(self.parameters.fullfield_gratings_dsv,st_name="FullfieldDriftingSinusoidalGrating",sheet_name=sheet)
            trials = max([load_parameters(s.replace("MozaikExtended",""))["trial"] for s in dsv.get_stimuli()]) + 1
            orientations = sorted(list(set([load_parameters(stim)["orientation"] for stim in dsv.get_stimuli()])))
            A_visual = np.array([[np.array(queries.param_filter_query(dsv,analysis_algorithm="RecordingArrayTimecourse",st_trial=trial,st_orientation=orientation).get_analysis_result()[0].analog_signal) for trial in range(trials)] for orientation in orientations])[:,:,2,:,:].mean(axis=1)
            A_spont_exc = np.array(queries.param_filter_query(self.datastore,analysis_algorithm="RecordingArrayTimecourse",st_name='InternalStimulus',sheet_name=sheet).get_analysis_result()[0].analog_signal)
    
            for reference,vname in zip([A_visual,np.flip(A_visual,axis=-1)],["Correlations","Correlations (control)"]):
                correlations = np.array([scipy.stats.pearsonr(A_spont_exc[i,:,:].flatten(),reference[j,:,:].flatten())[0] for i in range(A_spont_exc.shape[0]) for j in range(len(orientations))])
                self.datastore.full_datastore.add_analysis_result(
                    SingleValue(
                        value=correlations,
                        value_units=qt.dimensionless,
                        value_name=vname,
                        sheet_name=sheet,
                        stimulus_id=None,
                        analysis_algorithm=self.__class__.__name__,
                    )
                )

class Tsodyks_1999(Analysis):
    required_parameters = ParameterSet({
    "fullfield_gratings_dsv": DataStoreView,
    "n_neurons": int,
    })

    def tag_value(self, tag, tags):
        if len(tags) == 0:
            raise RuntimeError("No tags on recording!")
        filtered_tags = [t.split(":")[-1] for t in tags if ":".join(t.split(":")[:-1]) == tag]
        if len(filtered_tags) == 0:
            return None
        assert len(filtered_tags) == 1, "Duplicate tags are not allowed!"
        return eval(filtered_tags[0])
    
    def perform_analysis(self):
        sheets = self.datastore.sheets()
        for sheet in self.datastore.sheets():
            dsv = queries.param_filter_query(self.parameters.fullfield_gratings_dsv,st_name="FullfieldDriftingSinusoidalGrating",sheet_name=sheet)
            trials = max([load_parameters(s.replace("MozaikExtended",""))["trial"] for s in dsv.get_stimuli()]) + 1
            orientations = sorted(list(set([load_parameters(stim)["orientation"] for stim in dsv.get_stimuli()])))
            dsv_ar = queries.param_filter_query(dsv,analysis_algorithm="RecordingArrayTimecourse")
            t_res = self.tag_value("t_res", dsv_ar.get_analysis_result()[0].tags)
            A_visual = np.array([[np.array(queries.param_filter_query(dsv_ar,st_trial=trial,st_orientation=orientations[j]).get_analysis_result()[0].analog_signal) for j in range(len(orientations))] for trial in range(trials)])
            A_spont_exc = np.array(queries.param_filter_query(self.datastore,analysis_algorithm="RecordingArrayTimecourse",st_name='InternalStimulus',sheet_name=sheet).get_analysis_result()[0].analog_signal)

            seg = queries.param_filter_query(dsv).get_segments()[0]
            sampled_neuron_ids = np.random.choice(seg.get_stored_spike_train_ids(), size=self.parameters.n_neurons, replace=False)

            counts = [0 for i in range(self.parameters.n_neurons)]
            pcs = [np.zeros((A_visual.shape[-2],A_visual.shape[-1])) for i in range(self.parameters.n_neurons)]
            for i in range(trials):
                for j in range(len(orientations)):
                    #print(i,j)
                    spiketrains = queries.param_filter_query(dsv,st_trial=i,st_orientation=orientations[j]).get_segments()[0].get_spiketrain(sampled_neuron_ids)
                    for k in range(self.parameters.n_neurons):
                        counts[k] += len(spiketrains[k])
                        for spiketime in spiketrains[k]:
                            if spiketime >= A_visual.shape[0] * t_res:
                                continue
                            idx = int(spiketime // t_res)
                            pcs[k] += A_visual[i,j,idx,:,:]
            pcs = np.array([pcs[i] / counts[i] for i in range(self.parameters.n_neurons)])

            corrs_pcs = np.array([[scipy.stats.pearsonr(A_spont_exc[i,:,:].flatten(),pcs[j,:,:].flatten())[0] for i in range(A_spont_exc.shape[0])] for j in range(pcs.shape[0])])
            spiketrains = queries.param_filter_query(self.datastore,sheet_name=sheet,st_name='InternalStimulus').get_segments()[0].get_spiketrain(sampled_neuron_ids)
            spike_indices = [[int(spiketime // t_res) for spiketime in spiketrains[k] if int(spiketime // t_res) < corrs_pcs.shape[1]] for k in range(self.parameters.n_neurons)]
            corrs_spikes = np.array([[corrs_pcs[i,spike_indices[i][j]] for j in range(len(spike_indices[i]))] for i in range(self.parameters.n_neurons)], dtype=object)

            self.datastore.full_datastore.add_analysis_result(
                PerNeuronValue(
                    values=pcs,
                    value_units=qt.dimensionless,
                    value_name="PCS",
                    idds=sampled_neuron_ids,
                    sheet_name=sheet,
                    stimulus_id=None,
                    analysis_algorithm=self.__class__.__name__,
                )
            )
            self.datastore.full_datastore.add_analysis_result(
                PerNeuronValue(
                    values=corrs_pcs,
                    value_units=qt.dimensionless,
                    value_name="PCS correlations",
                    idds=sampled_neuron_ids,
                    sheet_name=sheet,
                    stimulus_id=None,
                    analysis_algorithm=self.__class__.__name__,
                )
            )
            self.datastore.full_datastore.add_analysis_result(
                PerNeuronValue(
                    values=corrs_spikes,
                    value_units=qt.dimensionless,
                    value_name="PCS spiketime correlations",
                    idds=sampled_neuron_ids,
                    sheet_name=sheet,
                    stimulus_id=None,
                    analysis_algorithm=self.__class__.__name__,
                )
            )

class SaveStimPatterns(Analysis):
    required_parameters = ParameterSet({
    })

    def get_stim_patterns(self, dsv, img_path, array_width, s_res):
        basepath = "/".join(os.path.realpath(dsv.full_datastore.parameters.root_directory).split('/')[:-2]) + "/"
        pattern = np.load(basepath+img_path).T
        return skimage.transform.resize(pattern, (array_width // s_res, array_width // s_res),anti_aliasing=False)

    def perform_analysis(self):
        stim_patterns = {}
        dsv = queries.param_filter_query(self.datastore,y_axis_name="recording array timecourse")
        for anasig in dsv.get_analysis_result():
            img_path = load_parameters(load_parameters(str(anasig))['stimulus_id'].replace("MozaikExtended",""))['direct_stimulation_parameters']['stimulating_signal_parameters']['image_path']
            tags = {t.split(":")[0] : eval(t.split(":")[1]) for t in anasig.tags}
            if img_path not in stim_patterns:
                stim_patterns[img_path] = (self.get_stim_patterns(dsv, img_path, tags['array_width'], tags['s_res']), anasig.tags)

        for k,v in stim_patterns.items():
            self.datastore.full_datastore.add_analysis_result(
                SingleValue(
                    value=v[0],
                    value_units=qt.dimensionless,
                    value_name="stimulation pattern",
                    tags=v[1],
                    stimulus_id=str({"module_path" :"mozaik.analysis.data_structures","name":'SingleValue',"identifier":k, "sheet_name":'V1_Exc_L2/3'}), # Hack because tags do not count into the signature of the analysis objects
                    analysis_algorithm=self.__class__.__name__,
                )
            )

class DistributionOfActivityAcrossOrientationDomainAnalysis(Analysis):

    required_parameters = ParameterSet({
        "or_map_dsv": DataStoreView,
        "use_stim_pattern_mask": bool,
        "n_orientation_bins": int,
    })

    def perform_analysis(self):
        for sheet in self.datastore.sheets():
            dsv = queries.param_filter_query(self.datastore,y_axis_name="recording array timecourse",sheet_name=sheet)
            or_map = queries.param_filter_query(self.parameters.or_map_dsv,value_name="orientation map",sheet_name=sheet).get_analysis_result()[0].value
            for anasig in dsv.get_analysis_result():
                if self.parameters.use_stim_pattern_mask:
                    img_path = load_parameters(load_parameters(str(anasig))['stimulus_id'].replace("MozaikExtended",""))['direct_stimulation_parameters']['stimulating_signal_parameters']['image_path']
                    daod_mask = queries.param_filter_query(self.datastore.full_datastore,st_identifier=img_path).get_analysis_result()[0].value
                else:
                    daod_mask = np.ones((anasig.analog_signal.shape[1],anasig.analog_signal.shape[2]))
                daod = np.zeros((anasig.analog_signal.shape[0],self.parameters.n_orientation_bins))
                A = anasig.analog_signal.reshape((anasig.analog_signal.shape[0],-1))
                A = A[:,daod_mask.flatten() > 0]
                orientations = or_map.flatten()[daod_mask.flatten() > 0]
                or_bins = np.linspace(0,np.pi,self.parameters.n_orientation_bins+1)
                for i in range(A.shape[0]):
                    daod[i,:], _ = np.histogram(orientations,bins=or_bins,weights=A[i])
                    daod[i,:] = daod[i,:] / len(orientations) * self.parameters.n_orientation_bins
                self.datastore.full_datastore.add_analysis_result(
                    SingleValue(
                        value=daod,
                        value_units=qt.dimensionless,
                        value_name="DAOD",
                        tags=self.tags,
                        sheet_name=sheet,
                        stimulus_id=anasig.stimulus_id,
                        analysis_algorithm=self.__class__.__name__,
                    )
                )

class MulhollandSmithPlots(Plotting):

    def truncate_colormap(self, cmap, minval=0.0, maxval=1.0, n=-1):
        if n == -1: 
            n = cmap.N 
        new_cmap = matplotlib.colors.LinearSegmentedColormap.from_list( 
             'trunc({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval, b=maxval), 
             cmap(np.linspace(minval, maxval, n))) 
        return new_cmap 

    def prediction_interval(self,data,confidence=0.95):
        n = len(data)
        t_critical = scipy.stats.t.ppf((1 + confidence) / 2, df=n-1)
        return t_critical * data.std(ddof=1) * np.sqrt(1+1/len(data))    
    
    def single_metric_plot(self,ax,exp,model,ylim,ylabel,has_legend=False):
        exp = np.array(exp)
        ax.spines[['top','right']].set_visible(False)
        ax.spines[['left','bottom']].set_linewidth(1.5)

        h0 = ax.errorbar(
            [1],
            exp.mean(),
            yerr=self.prediction_interval(exp),
            marker="",
            ls="-",
            alpha=1,
            color='k',
            markersize=6,
            #mec="",
            lw=1.5,
            markeredgewidth=1,
            ecolor='k', capsize=4,
            zorder=-1,
        )[0]
        
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

    def double_metric_plot(self,ax,exp_0,exp_1,model_0,model_1,ylim,ylabel,has_legend=False,x_ticks=["",""],ylabel_pad=0):
        exp_0 = np.array(exp_0)
        exp_1 = np.array(exp_1)
        ax.spines[['top','right']].set_visible(False)
        ax.spines[['left','bottom']].set_linewidth(1.7)
    
        h0 = ax.errorbar(
            [1],
            exp_0.mean(),
            yerr=self.prediction_interval(exp_0),
            marker="",
            ls="-",
            alpha=1,
            color='k',
            markersize=6,
            #mec="",
            lw=1.5,
            markeredgewidth=1,
            ecolor='k', capsize=4,
            zorder=-1,
        )[0]
        h1,= ax.plot(np.ones_like(exp_0)*1,exp_0,'o',color='silver',alpha=0.5)
        h2,= ax.plot([1],exp_0.mean(),'o',color='k')
        h3,= ax.plot([1],model_0,'ro')
        
        ax.errorbar(
            [2],
            exp_1.mean(),
            yerr=self.prediction_interval(exp_1),
            marker="",
            ls="-",
            alpha=1,
            color='k',
            markersize=6,
            #mec="",
            lw=1.5,
            markeredgewidth=1,
            ecolor='k', capsize=4,
            zorder=-1,
        )[0]
        ax.plot(np.ones_like(exp_1)*2,exp_1,'o',color='silver',alpha=0.5)
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
        fs = 9
        ax.set_ylim(ylim[0],ylim[1])
        ax.set_xlim(0.5,2.5)
        if has_legend:
            ax.legend([h1,h2,h0,h3],
                        ["Experiment","Exp. mean","95% Prediction Interval (PI)","Model"],
                       bbox_to_anchor=(0.2, -0.15),
                       handlelength = 0.8,
                       frameon=False,ncols=4,fontsize=fs)
        ax.set_ylabel(ylabel,labelpad=ylabel_pad,fontsize=fs)
        ax.tick_params(axis='both', labelsize=fs, size=6, width=1.7)
        ax.set_xticks([1,2],x_ticks,fontsize=fs)
        ax.set_yticks(ylim,[str(ylim[0]),str(ylim[1])],fontsize=fs)

    def get_experimental_data(self):
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
            "Mulholland 2024 may": {
                "exc opt similarity": [0.50349,0.56358,0.6047,0.55066,0.45385,0.37765],
                "spontaneous": {
                    "modularity": [0.113758,0.094554,0.103049,0.121333,0.134739,0.084584,0.136109],
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
                    "modularity": [0.119471,0.136224,0.138085,0.089227,0.103886,0.061541,0.191605],
                    "wavelength": {
                        "hist": {
                            "x": [0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35, 1.45, 1.55, 1.65, 1.75],
                            "y": [0.005945,0.008341,0.025811,0.194454,0.308177,0.251508,0.119363,0.04852,0.019643,0.01318,0.005614,0.008601,0.002685,0.003032,0.007663], 
                        },
                        "mean": [0.81685,0.816851,0.717572,0.763818,0.871507,0.874743,0.847903],
                        "std": [0.075143,0.187204,0.193154,0.211712,0.0649419,0.257613,0.133793],
                    },
                },
                "fullfield_timecourse": {
                    "x": [-0.9189999999999999, -0.793, -0.655, -0.502, -0.372, -0.23500000000000001, -0.097, 0.02500000000000001, 0.186, 0.29300000000000004, 0.45399999999999996, 0.622, 0.76, 0.9359999999999999, 0.989, 1.05, 1.119, 1.203, 1.28, 1.3410000000000002, 1.4020000000000001, 1.4560000000000002, 1.524, 1.586, 1.647, 1.7160000000000002, 1.83, 1.8920000000000001, 1.945, 2.006, 2.052, 2.106, 2.1750000000000003, 2.259, 2.335, 2.419, 2.496, 2.557, 2.618, 2.6870000000000003, 2.771, 2.878, 2.985, 3.092, 3.2070000000000003, 3.322, 3.482, 3.6510000000000002, 3.781, 3.926, 4.01, 4.129],
                    "y": [0.046,0.035,0.032,0.028,0.028,0.028,0.025,0.028,0.028,0.028,0.028,0.028,0.032,0.035,0.06,0.116,0.212,0.335,0.451,0.575,0.702,0.804,0.91,1.002,1.101,1.192,1.28,1.326,1.333,1.33,1.309,1.266,1.199,1.108,1.03,0.938,0.854,0.797,0.737,0.684,0.617,0.547,0.48,0.423,0.367,0.328,0.279,0.236,0.212,0.187,0.173,0.155],
                },
            },
            "Mulholland 2024 january": {
                "endo_inside_outside": {
                    "t": [-0.048600000000000004, -0.044500000000000005, -0.0442, -0.0396, -0.037500000000000006, 0.0063, 0.008099999999999996, 0.010699999999999994, 0.016399999999999998, 0.017100000000000004, 0.04859999999999999, 0.055999999999999994, 0.057300000000000004, 0.0762, 0.08039999999999999, 0.0957, 0.10640000000000001, 0.135, 0.14100000000000001, 0.1482, 0.14939999999999998, 0.19290000000000002, 0.19779999999999998, 0.2007, 0.2051, 0.24, 0.2409, 0.25420000000000004, 0.2692, 0.27840000000000004, 0.29460000000000003, 0.3115, 0.3254, 0.3428, 0.3484, 0.3773, 0.37820000000000004, 0.40700000000000003, 0.4096, 0.4338, 0.4464, 0.4684, 0.4915, 0.5, 0.5158999999999999, 0.5345, 0.5607, 0.5747, 0.5944999999999999, 0.6004999999999999, 0.6267999999999999, 0.6593, 0.6655, 0.6902999999999999, 0.6973999999999999, 0.7104999999999999, 0.7432, 0.7618999999999999, 0.7636999999999999, 0.7750999999999999, 0.8120999999999999, 0.8206, 0.8238, 0.8385999999999999, 0.8785999999999999, 0.8899999999999999, 0.8933, 0.8993, 0.9249999999999999, 0.9328, 0.9392999999999999, 0.9543999999999999, 0.9649999999999999, 0.9661, 0.9758, 0.9881, 0.9956, 0.9989999999999999, 1.0205, 1.0212999999999999, 1.0315999999999999, 1.0327, 1.044, 1.0591, 1.0671, 1.0722, 1.0778999999999999, 1.0918999999999999, 1.0996, 1.1040999999999999, 1.1194, 1.1256, 1.1286, 1.1403999999999999, 1.1482999999999999, 1.1672, 1.169, 1.1774, 1.1827999999999999, 1.2026, 1.2103, 1.2173, 1.2175, 1.2406, 1.2462, 1.2517, 1.2538, 1.2674999999999998, 1.2913999999999999, 1.2949, 1.2985, 1.3123, 1.3334, 1.3371, 1.3407, 1.3647, 1.3695, 1.3755, 1.3865, 1.3951, 1.4041, 1.4233, 1.4294, 1.4299, 1.4329, 1.4585, 1.4667999999999999, 1.4799, 1.4929999999999999, 1.5022, 1.51, 1.5164, 1.5214999999999999, 1.5433, 1.5436999999999999, 1.5487, 1.5677999999999999, 1.5742, 1.5802, 1.5855, 1.6074, 1.6140999999999999, 1.63, 1.6472, 1.6493, 1.666, 1.6729, 1.6809, 1.7068999999999999, 1.7130999999999998, 1.7221, 1.7402, 1.7527, 1.7558, 1.756, 1.7986, 1.7988, 1.7989, 1.8019, 1.8405, 1.8459999999999999, 1.8473, 1.857, 1.8626, 1.8833, 1.887, 1.894, 1.9022, 1.9087, 1.9303, 1.9329, 1.9339, 1.9396, 1.9555999999999998, 1.9587, 1.9669999999999999, 1.9732999999999998, 1.9795, 1.9915999999999998, 1.9963, 1.9965, 2.0128000000000004, 2.027, 2.0324, 2.0326, 2.0544000000000002, 2.0583, 2.0645000000000002, 2.0731, 2.084, 2.0965000000000003, 2.1032, 2.1083000000000003, 2.1124, 2.1334, 2.1467, 2.1481000000000003, 2.1522, 2.1603000000000003, 2.1697, 2.1823, 2.1853000000000002, 2.1947, 2.2153, 2.2225, 2.2229, 2.2267, 2.2451000000000003, 2.2598000000000003, 2.2609000000000004, 2.2727000000000004, 2.274, 2.3064, 2.309, 2.3113, 2.3331000000000004, 2.3401, 2.3487, 2.3586, 2.3804000000000003, 2.3846000000000003, 2.3898, 2.4013, 2.4214, 2.422, 2.4449, 2.4514, 2.4585000000000004, 2.4866, 2.4955000000000003, 2.5127, 2.5232, 2.5317000000000003, 2.5325, 2.5631000000000004, 2.5717000000000003, 2.5740000000000003, 2.5771, 2.6168, 2.6245000000000003, 2.6350000000000002, 2.6532, 2.6574, 2.6824000000000003, 2.6839000000000004, 2.7067, 2.7242, 2.7361, 2.7497000000000003, 2.7567000000000004, 2.7870000000000004, 2.8067, 2.8088, 2.8237, 2.8561, 2.8638000000000003, 2.8743000000000003, 2.9093, 2.9137, 2.9392, 2.9576000000000002, 2.972, 3.0243, 3.0317000000000003, 3.0349000000000004, 3.0392, 3.0956, 3.1011, 3.1495, 3.157, 3.1753, 3.1934, 3.2563, 3.2680000000000002, 3.2851000000000004, 3.3333000000000004, 3.3671, 3.3763, 3.3783000000000003, 3.4218, 3.4876, 3.4886000000000004, 3.4893, 3.4975, 3.5792, 3.5809, 3.5934000000000004, 3.5983, 3.6715, 3.6813000000000002, 3.6917, 3.6999, 3.7572, 3.7670000000000003, 3.7739000000000003, 3.7783, 3.841, 3.8537000000000003, 3.8554000000000004, 3.8625000000000003, 3.8696, 3.9064, 3.9284000000000003, 3.9321, 3.9444000000000004, 3.9507000000000003, 3.9828, 3.9864000000000006, 3.9880000000000004, 4.0016],
                    "inside": [0.05014,0.04995,0.04994,0.04974,0.04965,0.04964,0.04964,0.04964,0.04964,0.04973,0.04994,0.04999,0.04999,0.04982,0.04978,0.04964,0.04954,0.04965,0.04968,0.04971,0.04971,0.04922,0.04916,0.04899,0.04871,0.04655,0.04652,0.04615,0.04573,0.04547,0.04501,0.04433,0.04377,0.04307,0.04284,0.04186,0.04183,0.04085,0.04084,0.04069,0.04062,0.04049,0.04017,0.04005,0.03983,0.03958,0.03934,0.03921,0.03903,0.03903,0.03903,0.03902,0.03902,0.03907,0.03908,0.0391,0.03916,0.03918,0.03919,0.03934,0.03983,0.03995,0.03999,0.04033,0.04125,0.04151,0.04179,0.04229,0.04445,0.0451,0.04565,0.04869,0.05082,0.05103,0.05299,0.05545,0.05807,0.05925,0.06671,0.06698,0.07057,0.07093,0.07477,0.07986,0.08256,0.0843,0.08623,0.09109,0.0938,0.09537,0.1007,0.10287,0.10403,0.10871,0.11183,0.11925,0.12009,0.12391,0.12638,0.13545,0.13856,0.14136,0.14143,0.15073,0.153,0.15572,0.15677,0.16361,0.17556,0.1773,0.17904,0.18571,0.1959,0.19769,0.19944,0.21183,0.21434,0.21744,0.22337,0.22803,0.23291,0.24498,0.24881,0.24911,0.25076,0.26465,0.26918,0.27521,0.2812,0.28547,0.28956,0.29296,0.29567,0.30716,0.30732,0.30929,0.31678,0.31928,0.32161,0.32372,0.33397,0.33715,0.3446,0.35081,0.35157,0.35761,0.36008,0.36349,0.37456,0.37718,0.3806,0.38742,0.39214,0.39332,0.39337,0.41045,0.4105,0.41056,0.41177,0.42748,0.4293,0.42972,0.43292,0.43477,0.44044,0.44133,0.44298,0.44492,0.44646,0.44936,0.44971,0.44984,0.4506,0.45233,0.45267,0.45357,0.45414,0.4547,0.45582,0.45559,0.45558,0.45479,0.4541,0.45349,0.45346,0.45095,0.4505,0.4492,0.44743,0.44518,0.44253,0.4411,0.44003,0.43915,0.43311,0.42847,0.42798,0.42654,0.42385,0.42068,0.41673,0.4158,0.41285,0.40445,0.40149,0.40132,0.39972,0.39187,0.38666,0.38624,0.38207,0.3816,0.36976,0.36881,0.36796,0.35993,0.35731,0.35414,0.35052,0.34253,0.341,0.3391,0.33505,0.32793,0.32773,0.31989,0.31767,0.31523,0.30573,0.30272,0.29644,0.29263,0.28953,0.28921,0.28061,0.2782,0.27757,0.27669,0.26309,0.26104,0.2582,0.25332,0.25219,0.24556,0.24517,0.23914,0.23506,0.23231,0.22914,0.22752,0.21965,0.21455,0.2141,0.2109,0.20392,0.20261,0.20082,0.19485,0.1941,0.18973,0.18658,0.1841,0.17589,0.17473,0.17433,0.17378,0.16662,0.16598,0.16045,0.1596,0.15751,0.15544,0.14932,0.14819,0.14652,0.14242,0.13954,0.13877,0.13865,0.13613,0.13233,0.13227,0.13223,0.13179,0.12735,0.12726,0.12658,0.12631,0.12225,0.12171,0.12138,0.12111,0.11927,0.11895,0.11873,0.11859,0.11653,0.11611,0.11608,0.11596,0.11583,0.11519,0.11481,0.11475,0.11457,0.11448,0.11402,0.11397,0.11394,0.11375],
                    "outside": [0.03569,0.03569,0.03569,0.0357,0.0357,0.03578,0.03578,0.03579,0.03582,0.03582,0.03595,0.03588,0.03587,0.03568,0.03564,0.03549,0.03535,0.03495,0.03487,0.03477,0.03474,0.03377,0.03365,0.03359,0.03345,0.03235,0.03232,0.03226,0.03219,0.03214,0.0318,0.03144,0.03114,0.03102,0.03098,0.03078,0.03076,0.03016,0.0301,0.0296,0.02934,0.02889,0.02841,0.02824,0.02812,0.02798,0.02778,0.02784,0.02792,0.02794,0.02804,0.0279,0.02787,0.02776,0.02774,0.02769,0.02756,0.02749,0.0275,0.02761,0.02794,0.02802,0.02803,0.0281,0.02828,0.02837,0.02839,0.02844,0.02863,0.02932,0.02988,0.03121,0.03214,0.03223,0.03394,0.03608,0.03739,0.03799,0.04201,0.04215,0.04408,0.04428,0.04675,0.05002,0.05175,0.05302,0.05443,0.05787,0.05972,0.06079,0.06444,0.06604,0.0668,0.06986,0.0719,0.07629,0.07671,0.07865,0.07991,0.0849,0.08684,0.08839,0.08843,0.09359,0.09519,0.09674,0.09734,0.10123,0.10783,0.10879,0.10978,0.11393,0.12027,0.12139,0.12242,0.12927,0.13066,0.13237,0.13549,0.13795,0.14044,0.14577,0.14749,0.14763,0.14848,0.15567,0.15798,0.1616,0.16521,0.16788,0.17011,0.17196,0.17307,0.1778,0.17788,0.17897,0.18396,0.18563,0.18698,0.18821,0.19318,0.19472,0.19882,0.20327,0.2037,0.20714,0.20854,0.21019,0.21564,0.21693,0.21883,0.22245,0.22495,0.22558,0.22561,0.23419,0.23421,0.23425,0.23484,0.24295,0.24412,0.24438,0.24565,0.24639,0.2491,0.24959,0.25033,0.2512,0.25189,0.25417,0.25445,0.25453,0.255,0.25632,0.25657,0.25726,0.25778,0.25811,0.25877,0.25902,0.25903,0.25822,0.25751,0.25724,0.25723,0.25447,0.25397,0.25317,0.25206,0.25063,0.24901,0.24813,0.24743,0.24687,0.244,0.24218,0.24192,0.24114,0.23964,0.23788,0.23553,0.23497,0.2329,0.22837,0.22678,0.22669,0.22593,0.22219,0.21921,0.21894,0.21631,0.21602,0.20876,0.20827,0.20783,0.2037,0.20236,0.20073,0.19885,0.19457,0.19375,0.19273,0.19047,0.18634,0.18622,0.18153,0.18021,0.17887,0.17358,0.1719,0.16866,0.16669,0.16513,0.16496,0.15931,0.15798,0.15764,0.15716,0.15103,0.14985,0.14823,0.14538,0.14473,0.14079,0.14057,0.13745,0.13505,0.13343,0.13156,0.13069,0.12694,0.12451,0.12426,0.12258,0.11891,0.11804,0.11689,0.11303,0.11255,0.10975,0.10827,0.10711,0.10292,0.10232,0.10207,0.10172,0.09721,0.09676,0.09376,0.0933,0.09216,0.09124,0.08804,0.08745,0.08658,0.08413,0.08282,0.08246,0.08239,0.0807,0.07834,0.07831,0.07828,0.07799,0.07555,0.0755,0.07519,0.07507,0.07324,0.073,0.07274,0.07267,0.07219,0.07212,0.07207,0.07203,0.07157,0.07147,0.07146,0.07141,0.07137,0.07114,0.071,0.07098,0.07092,0.07089,0.07074,0.07072,0.07071,0.07065],
                },
            },
            "Kenet 2003": {
                "correlations": {
                    "x": [-0.597969,-0.556354,-0.526465,-0.490598,-0.46116,-0.436117,-0.415471,-0.398887,-0.381058,-0.361536,-0.343365,-0.323497,-0.307919,-0.287931,-0.275281,-0.235854,-0.224892,-0.187936,-0.176181,-0.170641,-0.138087,-0.115704,-0.102253,-0.093892,-0.083384,-0.069491,-0.057633,-0.043295,-0.029526,-0.018355,-0.008428,0.00037,0.010408,0.020669,0.032392,0.044,0.057971,0.071038,0.08669,0.100198,0.11449,0.128102,0.173341,0.184142,0.191344,0.222857,0.229386,0.271605,0.283993,0.293453,0.29931,0.322519,0.339196,0.35238,0.375373,0.391831,0.414716,0.425427,0.442567,0.46478,0.487447,0.510115,0.532106,0.566391,0.598985],
                    "y": [0.35,-0.51,0.54,1.94,3.68,6.47,10.81,14.29,20.02,26.97,34.96,46.42,55.28,71.25,82.71,124.03,137.92,185.83,202.15,210.66,256.31,288.95,307.7,317.25,329.4,342.25,351.8,361.35,367.25,370.9,372.12,372.64,373.16,371.43,367.62,362.07,353.91,343.68,328.58,312.44,292.83,271.14,208.5,192.71,182.64,141,133.54,87.21,75.76,66.91,62.4,47.3,38.11,30.48,21.98,16.78,10.71,8.63,6.38,3.61,1.88,1.36,0.15,-0.01,0.17],
                },
                "correlations (control)": {
                    "x": [-0.597857,-0.59436,-0.591597,-0.58968,-0.585507,-0.582349,-0.578797,-0.576315,-0.573045,-0.570282,-0.537745,-0.515639,-0.498666,-0.481128,-0.46455,-0.421861,-0.391241,-0.363835,-0.342294,-0.320245,-0.288778,-0.270394,-0.255223,-0.24975,-0.244164,-0.237785,-0.224974,-0.216903,-0.209171,-0.201608,-0.197766,-0.191035,-0.184079,-0.177854,-0.171687,-0.162524,-0.153699,-0.149341,-0.144002,-0.139685,-0.134174,-0.130538,-0.124347,-0.121163,-0.11719,-0.110374,-0.105148,-0.10066,-0.096507,-0.091442,-0.08814,-0.082221,-0.075505,-0.069529,-0.036412,-0.029577,-0.021125,-0.001671,0.00048,0.001497,0.005603,0.024682,0.028951,0.038492,0.044722,0.050223,0.052684,0.072514,0.077264,0.085638,0.091969,0.10084,0.107186,0.123407,0.135817,0.14428,0.148936,0.150492,0.174711,0.19015,0.199499,0.219231,0.238825,0.251602,0.283144,0.380478,0.51206,0.598739],
                    "y": [0.35,0.35,0.44,0.35,0.26,0.27,0.27,0.27,0.09,0.1,-0.24,-0.06,-0.14,-0.14,-0.48,-0.3,-0.37,-0.45,-0.53,-0.27,-0.17,0.27,1.58,3.23,5.74,9.82,16.16,20.85,24.5,29.01,33.79,46.37,58.96,72.67,85.26,102.88,120.41,129.96,154,173.7,199.39,216.31,244.08,260.31,276.54,307.18,330.78,351.52,374.61,403.16,423.21,456.71,495.5,530.22,689.59,709.7,732.85,789.71,794.75,795.88,789.11,730.71,719.78,691.75,673.01,657.04,644.8,526.18,499.36,450.11,410.53,359.8,329.41,254.63,195.37,156.01,135.24,130.43,81.23,50.18,31.38,19.58,9.74,2.53,0.35,-0.93,-0.46,-0.21],
                }, 
            },
            "Tsodyks 1999": {
                "x": [-0.625, -0.575, -0.525, -0.475, -0.425, -0.375, -0.32499999999999996, -0.27499999999999997, -0.22499999999999998, -0.175, -0.125, -0.07499999999999996, -0.02499999999999991, 0.025000000000000022, 0.07500000000000007, 0.125, 0.17500000000000004, 0.2250000000000001, 0.275, 0.32500000000000007, 0.375, 0.42500000000000004, 0.4750000000000001, 0.5250000000000001, 0.5750000000000002],
                "y": {
                    "D": [242.4,305.2,699.4,1337.1,2542.6,4266.5,6418.3,9257.3,11921.4,14205.9,16014.1,17448.7,18455.3,17551.2,16219.1,13946.7,11921.4,9468.2,7117.5,5068.2,3217.7,1903.7,1132.2,445,402.8],
                    "E": [0,0,2.185,0,3.1,3.917,13.053,17.121,33.095,51.268,75.371,111.953,118.173,131.134,176.658,172.406,177.43,152.952,144.062,141.101,81,60.586,39.12,20.237,16.224],
                    "F": [0.009,0.026,0.911,0.042,0.313,0.167,0.708,0.707,0.923,1.229,1.373,1.842,2.202,2.563,4.098,4.169,5.451,6.335,8.015,9.857,9.404,11.879,13.396,21.308,29.368],
                }
            }
        }

    def values_from_queries(self,q):
        v = {}
        for k in q.keys():
            v[k] = q[k].get_analysis_result()
            assert len(v[k]) == 1, "Must need exactly 1 %s, got %d" % (k,len(v[k]))
            assert type(v[k][0]) == SingleValue
            v[k] = v[k][0].value
        return v

class Smith2018Mulholland2024Plot(MulhollandSmithPlots):   
    required_parameters = ParameterSet({})   

    def circ_dist(self, a, b):
            return np.pi/2 - abs(np.pi/2 - abs(a-b))

    def plot_hist_comparison(self,ax,e0,h0,e1,h1,title="",xlabel="",e1_center=False,ylim=[0,0.8]):
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
        
    def subplot(self, subplotspec):

        dsv = queries.param_filter_query(self.datastore,analysis_algorithm="Smith_2018_Mulholland_2021_2024_spont_analyses")
        q = {
            "Orientation map similarity": queries.param_filter_query(self.datastore,value_name="orientation map similarity",sheet_name="V1_Exc_L2/3",st_name='InternalStimulus'),
            "Orientation map": queries.param_filter_query(self.datastore,value_name="orientation map",sheet_name="V1_Exc_L2/3"),
            "Excitatory correlation maps": queries.param_filter_query(self.datastore,analysis_algorithm="CorrelationMaps",sheet_name="V1_Exc_L2/3",st_name='InternalStimulus'),
            "Inhibitory correlation maps": queries.param_filter_query(self.datastore,analysis_algorithm="CorrelationMaps",sheet_name="V1_Inh_L2/3",st_name='InternalStimulus'),
            "Kohonen map": queries.param_filter_query(dsv,value_name="Kohonen map",sheet_name="V1_Exc_L2/3"),
            "Spatial scale of correlation": queries.param_filter_query(dsv,value_name="Spatial scale of correlation",sheet_name="V1_Exc_L2/3"),
            "Local correlation eccentricity": queries.param_filter_query(dsv,value_name="Local correlation eccentricity",sheet_name="V1_Exc_L2/3"),
            "Dimensionality": queries.param_filter_query(dsv,value_name="Dimensionality",sheet_name="V1_Exc_L2/3"),
            "Event activity wavelength": queries.param_filter_query(dsv,value_name="Event activity wavelength",sheet_name="V1_Exc_L2/3"),
            "Event activity modularity": queries.param_filter_query(dsv,value_name="Event activity modularity",sheet_name="V1_Exc_L2/3"),
        }
            
        d = self.get_experimental_data()["Smith 2018"]
        d_ = self.get_experimental_data()["Mulholland 2024 may"]["spontaneous"]
        plots = {}
        gs = matplotlib.gridspec.GridSpecFromSubplotSpec(
            10, 35, subplot_spec=subplotspec, hspace=0.3, wspace=0.2
        )
        v = self.values_from_queries(q)
        print(v["Dimensionality"])
        upper_w = 6
        # Orientation map
        ax = pylab.subplot(gs[0:3,0:upper_w-1])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        ormap = pylab.imshow(v["Orientation map"],'hsv', interpolation='none')
        pylab.title("Orientation map",fontsize=10)
        #pylab.axis('equal')
        cbar = pylab.colorbar(ormap)
        cbar.set_label(label='Orientation preference', labelpad=5, fontsize=10)
        cbar.set_ticks([0,np.pi],labels=["0","$\\pi$"])

        cmap_idx = 8186
        cmap_x, cmap_y = cmap_idx % v["Excitatory correlation maps"].shape[-1], cmap_idx // v["Excitatory correlation maps"].shape[-1]
        # Exc correlation map
        ax = pylab.subplot(gs[0:3,1*upper_w:2*upper_w-1])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        cmap = pylab.imshow(v["Excitatory correlation maps"][cmap_idx],'bwr',vmin=-1,vmax=1)
        pylab.title("Excitatory\ncorrelation map",fontsize=10)
        ax.scatter(cmap_x,cmap_y,color='k',marker='x')
        cbar = pylab.colorbar(cmap)
        cbar.set_label(label='Correlation', labelpad=5, fontsize=10)
        cbar.set_ticks([-1,1],labels=["-1","1"])

        # Inh correlation map
        ax = pylab.subplot(gs[0:3,2*upper_w:3*upper_w-1])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        cmap = pylab.imshow(v["Inhibitory correlation maps"][cmap_idx],'bwr',vmin=-1,vmax=1)
        pylab.title("Inhibitory\ncorrelation map",fontsize=10)
        ax.scatter(cmap_x,cmap_y,color='k',marker='x')
        #pylab.axis('equal')
        cbar = pylab.colorbar(cmap)
        cbar.set_label(label='Correlation', labelpad=5, fontsize=10)
        cbar.set_ticks([-1,1],labels=["-1","1"])
        
        # Similarity map
        ax=pylab.subplot(gs[0:3, 3*upper_w:4*upper_w-1])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        inferno_t = self.truncate_colormap(plt.get_cmap("hot"), 0.5, 1) 
        smap = pylab.imshow(v["Orientation map similarity"].reshape(v["Orientation map"].shape), vmin=0, vmax=1,cmap=inferno_t)
        #pylab.axis("equal")
        pylab.title("Orientation map similarity\nmean=%.2f" % v["Orientation map similarity"].mean(),fontsize=10)
        cbar = pylab.colorbar(smap)
        cbar.set_label(label='Similarity', labelpad=5, fontsize=10)    
        cbar.set_ticks([0,1])
        
        # Kohonen map
        ax = pylab.subplot(gs[0:3,4*upper_w:5*upper_w-1])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        kohonenmap = pylab.imshow(v["Kohonen map"],'hsv', interpolation='none')
        estimation_error=np.nanmean(self.circ_dist(v["Orientation map"],v["Kohonen map"])) / np.pi * 180
        pylab.title("Kohonen map; error=%.2f°" % estimation_error,fontsize=10)
        cbar = pylab.colorbar(kohonenmap)
        cbar.set_label(label='Estimated orientation', labelpad=5, fontsize=10)
        cbar.set_ticks([0,np.pi],labels=["0","$\\pi$"])

        # Experiment comparison
        # Similarity
        self.single_metric_plot(pylab.subplot(gs[4:8,0:3]),d["similarity"],v["Orientation map similarity"].mean(),[0,1],"Exc $\\rightarrow$ or. map similarity",has_legend=False)
        self.single_metric_plot(pylab.subplot(gs[4:8,5:8]),d["spatial scale of correlation"],v["Spatial scale of correlation"],[0,1.5],"Spatial scale of correlation",has_legend=False)
        self.single_metric_plot(pylab.subplot(gs[4:8,10:13]),d["dimensionality"],v["Dimensionality"],[0,20],"Dimensionality",has_legend=False)
        lce_exc_bins = d["local correlation eccentricity_hist"]["x"]
        lce_exc_bins = np.hstack([lce_exc_bins,lce_exc_bins[-1]+lce_exc_bins[0]*2])
        lce_exc_bins -= lce_exc_bins[0]
        lce_exc_hist, _ = np.histogram(v["Local correlation eccentricity"],bins=lce_exc_bins)
        lce_exc_hist = lce_exc_hist.astype(float) / lce_exc_hist.sum()
        self.plot_hist_comparison(pylab.subplot(gs[4:8,15:19]),d["local correlation eccentricity_hist"]["x"],d["local correlation eccentricity_hist"]["y"],d["local correlation eccentricity_hist"]["x"],lce_exc_hist,title="",xlabel="Local correlation eccentricity",e1_center=True,ylim=[0,0.2])
        self.single_metric_plot(pylab.subplot(gs[4:8,21:24]),d["mean eccentricity"],v["Local correlation eccentricity"].mean(),[0,1],"Mean eccentricity",has_legend=False)
        #self.single_metric_plot(pylab.subplot(gs[4:8,22:25]),d_["wavelength"]["mean"],v["Event activity wavelength"].mean(),[0,1.5],"Event activity wavelength",has_legend=False)
        #self.single_metric_plot(pylab.subplot(gs[4:8,27:30]),d_["modularity"],v["Event activity modularity"].mean(),[0,0.2],"Modularity",has_legend=True)
        return plots

class Mulholland2021Plot(MulhollandSmithPlots):   
    required_parameters = ParameterSet({})
        
    def subplot(self, subplotspec):

        dsv = queries.param_filter_query(self.datastore,analysis_algorithm="Smith_2018_Mulholland_2021_2024_spont_analyses")
        q = {
            "Exc-Inh correlation map similarity": queries.param_filter_query(self.datastore,analysis_algorithm="CorrelationMapSimilarity",st_name="InternalStimulus"),
            "Exc local maxima distance correlation": queries.param_filter_query(dsv,value_name="Local maxima distance correlation",sheet_name="V1_Exc_L2/3"),
            "Inh local maxima distance correlation": queries.param_filter_query(dsv,value_name="Local maxima distance correlation",sheet_name="V1_Inh_L2/3"),
            "Exc correlation map wavelength": queries.param_filter_query(dsv,value_name="Correlation map wavelength",sheet_name="V1_Exc_L2/3"),
            "Inh correlation map wavelength": queries.param_filter_query(dsv,value_name="Correlation map wavelength",sheet_name="V1_Inh_L2/3"),
            "Exc local correlation eccentricity": queries.param_filter_query(dsv,value_name="Local correlation eccentricity",sheet_name="V1_Exc_L2/3"),
            "Inh local correlation eccentricity": queries.param_filter_query(dsv,value_name="Local correlation eccentricity",sheet_name="V1_Inh_L2/3"),
            "Exc dimensionality (random sampled events)": queries.param_filter_query(dsv,value_name="Dimensionality (random sampled events)",sheet_name="V1_Exc_L2/3"),
            "Inh dimensionality (random sampled events)": queries.param_filter_query(dsv,value_name="Dimensionality (random sampled events)",sheet_name="V1_Inh_L2/3"),
        }
        v = self.values_from_queries(q)
        d = self.get_experimental_data()["Mulholland 2021"]
        plots = {}
        gs = matplotlib.gridspec.GridSpecFromSubplotSpec(
            1, 50, subplot_spec=subplotspec, hspace=0.3, wspace=0.2
        ) 
        self.single_metric_plot(pylab.subplot(gs[0:6]),d['exc inh similarity'],v["Exc-Inh correlation map similarity"].mean(),[0,1],"Inh $\\rightarrow$ Exc similarity")
        self.double_metric_plot(pylab.subplot(gs[10:16]),d['corr above 2 mm']['exc'],d['corr above 2 mm']['inh'],v["Exc local maxima distance correlation"][1,v["Exc local maxima distance correlation"][0,:] > 2].mean(),v["Inh local maxima distance correlation"][1,v["Inh local maxima distance correlation"][0,:] > 2].mean(),[0,0.5],"Correlation at maxima (>2 mm)",x_ticks=["Exc","Inh"])
        self.double_metric_plot(pylab.subplot(gs[20:26]),d['mean eccentricity']['exc'],d['mean eccentricity']['inh'],v["Exc local correlation eccentricity"].mean(),v["Inh local correlation eccentricity"].mean(),[0,1],"Mean local correlation eccentricity",x_ticks=["Exc","Inh"])
        self.double_metric_plot(pylab.subplot(gs[30:36]),d['corr wavelength']['exc'],d['corr wavelength']['inh'],v["Exc correlation map wavelength"],v["Inh correlation map wavelength"],[0.5,1.3],"Corr. map wavelength (mm)",x_ticks=["Exc","Inh"])
        self.double_metric_plot(pylab.subplot(gs[40:46]),d['dimensionality']['exc'],d['dimensionality']['inh'],v["Exc dimensionality (random sampled events)"].mean(),v["Inh dimensionality (random sampled events)"].mean(),[0,20],"Dimensionality",has_legend=True,x_ticks=["Exc","Inh"])
        return plots

class Kenet2003Tsodyks1999Plot(MulhollandSmithPlots):   
    required_parameters = ParameterSet({
    })

    def histogram_median(self,bin_edges, heights):
        # Compute the cumulative sum of the bin heights
        cumulative_counts = np.cumsum(heights)
        total_count = cumulative_counts[-1]
        
        # Find the bin where the median falls
        median_index = np.searchsorted(cumulative_counts, total_count / 2)
        
        # Compute the precise median within the bin
        left_edge = bin_edges[median_index]
        bin_width = bin_edges[median_index + 1] - left_edge
        bin_count = heights[median_index]
        prev_count = cumulative_counts[median_index - 1] if median_index > 0 else 0
        
        # Linear interpolation within the bin
        median = left_edge + bin_width * (total_count / 2 - prev_count) / bin_count
        
        return median
    
    def subplot(self, subplotspec):
        d_kenet = self.get_experimental_data()["Kenet 2003"]
        d_tsodyks = self.get_experimental_data()["Tsodyks 1999"]

        kenet_corr = queries.param_filter_query(self.datastore,analysis_algorithm="Kenet_2003",value_name="Correlations").get_analysis_result()[0].value
        kenet_corr_control = queries.param_filter_query(self.datastore,analysis_algorithm="Kenet_2003",value_name="Correlations (control)").get_analysis_result()[0].value

        tsodyks_corrs_pcs = queries.param_filter_query(self.datastore,analysis_algorithm="Tsodyks_1999",value_name="PCS correlations").get_analysis_result()[0].values.flatten()
        tsodyks_corrs_pcs_spikes = queries.param_filter_query(self.datastore,analysis_algorithm="Tsodyks_1999",value_name="PCS spiketime correlations").get_analysis_result()[0].values
        tsodyks_corrs_pcs_spikes = np.concatenate(tsodyks_corrs_pcs_spikes)
        
        plots = {}
        gs = matplotlib.gridspec.GridSpecFromSubplotSpec(
            10, 22, subplot_spec=subplotspec, hspace=0.3, wspace=0.2
        )
        fontsize = 10

        # Kenet 2003
        bins = np.linspace(-0.6,0.6,41)
        h0, h1 = np.histogram(kenet_corr.flatten(),bins=bins,density=True)[0], np.histogram(kenet_corr_control.flatten(),bins=bins,density=True)[0]
        ax_kenet = pylab.subplot(gs[1:9, 0:4])
        ax_kenet.set_title("Correlation of spont. activity\nto oriented grating responses",fontsize=fontsize,pad=10)
        ax_kenet.spines[['top','right']].set_visible(False)
        ax_kenet.spines[['left','bottom']].set_linewidth(1.5)
        ax_kenet.tick_params(labelsize=fontsize, size=5, width=1.5)
        b0 = ax_kenet.bar(bins[:-1],h0,width=np.diff(bins), alpha=0.5, color='r', align='edge')
        b1 = ax_kenet.bar(bins[:-1],h1,width=np.diff(bins), alpha=0.5, color='b', align='edge')
        kenet_y = d_kenet["correlations"]["y"] / np.trapz(d_kenet["correlations"]["y"],d_kenet["correlations"]["x"])
        kenet_y_control = d_kenet["correlations (control)"]["y"] / np.trapz(d_kenet["correlations (control)"]["y"],d_kenet["correlations (control)"]["x"])
        ha0, = ax_kenet.plot(d_kenet["correlations (control)"]["x"],kenet_y_control,color='b',lw=1.5)
        ha1, = ax_kenet.plot(d_kenet["correlations"]["x"],kenet_y,color='r',lw=1.5)
        ax_kenet.set_xlabel("Correlation coefficient",fontsize=fontsize)
        ax_kenet.set_ylabel("Density",fontsize=fontsize)         
        #empty_handle = matplotlib.lines.Line2D([], [], linestyle="none", label="Legend Title")
        leg1 = ax_kenet.legend([b0,b1],
                   ["Grating","Control"],
                   handlelength=0.9,bbox_to_anchor=(0.95, 1.08),
                  ncols=1,title="Model",fontsize=fontsize,frameon=False,title_fontsize=fontsize)
        leg2 = ax_kenet.legend([ha1,ha0],
                   ["Grating","Control"],
                   handlelength=0.9,bbox_to_anchor=(0.95, 0.55),
                  ncols=1,title="Kenet et al.",fontsize=fontsize,frameon=False,title_fontsize=fontsize)
        leg1._legend_box.align = "left"
        leg2._legend_box.align = "left"
        
        ax_kenet.add_artist(leg1)
        
        ax_kenet.set_xticks(np.linspace(-0.6,0.6,5),labels=["%.1f" % el for el in np.linspace(-0.6,0.6,5)],fontsize=fontsize)
        ax_kenet.set_yticks([0,2,4,6],["0","2","4","6"],fontsize=fontsize)
        ax_kenet.set_xlim(-0.6,0.6)
        ax_kenet.set_ylim(0,6)

        # Tsodyks 1999 D

        bins = np.linspace(-0.625,0.625,26)
        h0,_ = np.histogram(tsodyks_corrs_pcs.flatten(),bins)
        h1,_ = np.histogram(tsodyks_corrs_pcs_spikes.flatten(),bins)
        
        axes = [pylab.subplot(gs[1:9, 2+i*5:2+i*5+4]) for i in range(1,4)]
        plt.subplots_adjust(wspace=0.6)
        fontsize=9

        model_color = 'royalblue'
        line_color = 'blue'
        alpha = 0.5
        linewidth = 2
        
        # First subplot
        axes[0].set_title("Correlation of PCS\nto spont. frames",fontsize=fontsize)
        axes[0].bar(d_tsodyks["x"], d_tsodyks["y"]["D"] / np.trapz(d_tsodyks["y"]["D"], d_tsodyks["x"]), width=np.diff(d_tsodyks["x"])[0], alpha=alpha, color='k', align='edge')
        axes[0].plot(np.ones((2)) * self.histogram_median(d_tsodyks["x"], d_tsodyks["y"]["D"]).mean(),[0,2],'k--')
        axes[0].bar(bins[:-1], h0 / np.trapz(h0, bins[:-1]), width=np.diff(bins), alpha=alpha, color=model_color, align='edge')
        axes[0].plot(np.ones((2)) * tsodyks_corrs_pcs.mean(),[0,2],'--',color=line_color)
        axes[0].tick_params(labelsize=fontsize, size=5, width=1.5)
        axes[0].set_ylabel("Density",fontsize=fontsize,labelpad=-7)
        axes[0].set_xlabel("Correlation\ncoefficient",fontsize=fontsize)
        axes[0].spines[['top', 'right']].set_visible(False)
        axes[0].spines[['left', 'bottom']].set_linewidth(1.5)
        axes[0].set_yticks([0,2])
        axes[0].set_ylim(0, 2)
        
        # Second subplot
        axes[1].set_title("Correlation of PCS\n(spike time spont. frames)",fontsize=fontsize)
        axes[1].bar(d_tsodyks["x"], d_tsodyks["y"]["E"] / np.trapz(d_tsodyks["y"]["E"], d_tsodyks["x"]), width=np.diff(d_tsodyks["x"])[0], alpha=alpha, color='k', align='edge')
        axes[1].plot(np.ones((2)) * self.histogram_median(d_tsodyks["x"], d_tsodyks["y"]["E"]).mean(),[0,2],'k--')
        axes[1].bar(bins[:-1], h1 / np.trapz(h1, bins[:-1]), width=np.diff(bins), alpha=alpha, color=model_color, align='edge')
        axes[1].tick_params(labelsize=fontsize, size=5, width=1.5)
        axes[1].set_ylabel("Density",fontsize=fontsize,labelpad=-7)
        axes[1].set_xlabel("Correlation\ncoefficient",fontsize=fontsize)
        axes[1].spines[['top', 'right']].set_visible(False)
        axes[1].spines[['left', 'bottom']].set_linewidth(1.5)
        axes[1].set_ylim(0, 2)
        #print(corrs_spikes.mean(),self.histogram_median(bins, tsodyks_e_y).mean())
        axes[1].plot(np.ones((2)) * tsodyks_corrs_pcs_spikes.mean(),[0,2],'--',color=line_color)
        axes[1].set_yticks([0,2])
        
        # Third subplot
        axes[2].tick_params(labelsize=fontsize, size=5, width=1.5)
        axes[2].plot(np.array(d_tsodyks["x"]) + 0.025, d_tsodyks["y"]["F"], 'ks',alpha=0.8)
        axes[2].plot(bins[:-1] + 0.025, h1 / h0 / (50 / 1000), 'bs',alpha=1)
        axes[2].set_ylabel("Predicted firing\nrate (sp/s)",fontsize=fontsize,labelpad=-13)
        axes[2].set_xlabel("Correlation\ncoefficient",fontsize=fontsize)
        axes[2].spines[['top', 'right']].set_visible(False)
        axes[2].spines[['left', 'bottom']].set_linewidth(1.5)
        axes[2].set_yticks([0,35])
        #plt.tight_layout()
        axes[2].legend(["Tsodyks et al. 1999", "Model"], handlelength=1, ncols=2, bbox_to_anchor=(0.2, -0.42),fontsize=fontsize,frameon=False)
        
        return plots

class CenterStimulationPlot(Plotting):   
    required_parameters = ParameterSet({
        "center_stim_dsv_list": list,
    })
    
    def radial_mask(self, array, radius, center_row, center_col):
        mask = np.zeros_like(array)
        rr, cc = disk((center_row, center_col), radius)
        mask[rr, cc] = 1
        return mask
    
    def mask_mean(self,A,masks):
        return np.array([np.nanmean(A[i,:,:][masks[i] > 0]) for i in range(A.shape[0])])

    def mask_sum(self,A,masks):
        return np.array([np.nansum(A[i,:,:][masks[i] > 0]) for i in range(A.shape[0])])
    
    def mean_or(self, or_map, masks):
        from mozaik.tools.circ_stat import circ_mean
        return np.array([circ_mean(or_map[mask == 1],axis=0,low=0,high=np.pi)[0] for mask in masks])
    
    def or_masks(self,or_map,orientations,masks_inv):
        def circ_dist_simple(a,b):
            return np.pi/2 - abs(np.pi/2 - abs(a-b))
        thresh = np.pi / 8
        or_masks_close = np.array([np.logical_and(masks_inv[-1], circ_dist_simple(or_map,orientations[i]) < thresh) for i in range(len(masks_inv))])
        or_masks_far = ([np.logical_and(masks_inv[-1], circ_dist_simple(or_map,orientations[i]) > np.pi / 2 - thresh) for i in range(len(masks_inv))])
        return or_masks_close, or_masks_far
        
    def tag_value(self, tag, tags):
        if len(tags) == 0:
            raise RuntimeError("No tags on recording!")
        filtered_tags = [t.split(":")[-1] for t in tags if ":".join(t.split(":")[:-1]) == tag]
        if len(filtered_tags) == 0:
            return None
        assert len(filtered_tags) == 1, "Duplicate tags are not allowed!"
        return eval(filtered_tags[0])    

    def retrieve_ds_param_values(self, dsv, param_name):
        l=[]
        for s in dsv.get_stimuli():
            if MozaikParametrized.idd(s).direct_stimulation_parameters != None:
                l.append(MozaikParametrized.idd(s).direct_stimulation_parameters.stimulating_signal_parameters[param_name])
        return sorted([eval(s) for s in set([str(ll) for ll in l])])
    
    def subplot(self, subplotspec):
        sheets = ["V1_Exc_L2/3","V1_Inh_L2/3"]
        spont_dsv = queries.param_filter_query(self.datastore,analysis_algorithm="RecordingArrayTimecourse",st_name='InternalStimulus')
        A_spont = {}
        for sheet in sheets:
            A_spont[sheet] = queries.param_filter_query(spont_dsv,sheet_name=sheet).get_analysis_result()
            assert len(A_spont[sheet]) == 1, "Need exactly 1 spontaneous activity recording from sheet %s, got %d" % (sheet,len(A_spont[sheet]))
            A_spont[sheet] = np.array(A_spont[sheet][0].analog_signal).mean(axis=0)
        or_map = queries.param_filter_query(self.datastore,analysis_algorithm="RecordingArrayOrientationMap",sheet_name="V1_Exc_L2/3").get_analysis_result()[0].value
        s_res, array_width = self.tag_value("s_res", spont_dsv.get_analysis_result()[0].tags), self.tag_value("array_width", spont_dsv.get_analysis_result()[0].tags)
        A = {}
        r_margin = 100 # um
        radii = sorted(list(set([r for ds in self.parameters.center_stim_dsv_list for r in self.retrieve_ds_param_values(ds, "radius")])))
        print(radii)

        onset_time, offset_time = self.retrieve_ds_param_values(self.parameters.center_stim_dsv_list[0], "onset_time")[0], self.retrieve_ds_param_values(self.parameters.center_stim_dsv_list[0], "offset_time")[0]
        t_res = self.tag_value("t_res", queries.param_filter_query(self.parameters.center_stim_dsv_list[0],analysis_algorithm="RecordingArrayTimecourse").get_analysis_result()[0].tags)
        trials = max([load_parameters(s.replace("MozaikExtended",""))["trial"] for s in self.parameters.center_stim_dsv_list[0].get_stimuli()]) + 1
        
        for sheet in sheets:
            A[sheet] = np.stack([np.stack([np.array([ar for ar in queries.param_filter_query(self.parameters.center_stim_dsv_list[i],analysis_algorithm="RecordingArrayTimecourse",sheet_name=sheet,st_trial=trial).get_analysis_result() if "\\\'radius\\\': %d" % radii[i] in str(ar)][0].analog_signal) for trial in range(trials)]) for i in range(len(radii))])
            #A[sheet] = np.stack([np.stack([np.array(queries.param_filter_query(ds,analysis_algorithm="RecordingArrayTimecourse",sheet_name=sheet,st_trial=trial).get_analysis_result()[0].analog_signal) for trial in range(trials)]) for ds in self.parameters.center_stim_dsv_list])
            # Take trial mean of mean activity during stimulation 
            A[sheet] = A[sheet][:,:,onset_time//t_res:offset_time//t_res,:,:].mean(axis=(1,2))
            A[sheet] = np.vstack([A_spont[sheet][np.newaxis,:,:],A[sheet]])

        radii.insert(0,0)
        center = self.retrieve_ds_param_values(self.parameters.center_stim_dsv_list[0], "coords")
        assert len(center) == 1 and len(center[0]) == 1
        center = center[0][0]
        center_row, center_col = int((center[1] + array_width / 2)) // s_res , int((center[0] + array_width / 2)) // s_res
        masks = [self.radial_mask(A[sheets[0]][0],(radii[i]+r_margin)/s_res, center_row, center_col) for i in range(len(radii))]
        masks[0][masks[0].shape[0] // 2, masks[0].shape[1] // 2] = 1
        masks_inv = [1-mask for mask in masks]
        masks_max = [masks[-1] for mask in masks]
        masks_inv_max = [masks_inv[-1] for mask in masks_inv]
        reference_orientation_constant = True
        
        if reference_orientation_constant:
            mean_ors = self.mean_or(or_map, [masks[0] for m in masks])
        else:
            mean_ors = self.mean_or(or_map, masks)
        or_masks_close, or_masks_far = self.or_masks(or_map,mean_ors,masks_inv)
        or_masks_close, or_masks_far = np.vstack([np.ones_like(or_map)[np.newaxis,...],or_masks_close]),np.vstack([np.ones_like(or_map)[np.newaxis,...],or_masks_far])
        
        A_masked = {sheet: {} for sheet in sheets}
        for sheet in sheets:
            A_masked[sheet]["center_mean"] = self.mask_mean(A[sheet],masks_max)
            A_masked[sheet]["surround_mean"] = self.mask_mean(A[sheet],masks_inv_max)
            A_masked[sheet]["center_sum"] = self.mask_sum(A[sheet],masks)
            A_masked[sheet]["center_sum_total_spikes"] = (A_masked[sheet]["center_sum"] * t_res / 1000)
            
        fr_close_exc = self.mask_mean(A["V1_Exc_L2/3"],or_masks_close)
        fr_close_inh = self.mask_mean(A["V1_Inh_L2/3"],or_masks_close)
        fr_far_exc = self.mask_mean(A["V1_Exc_L2/3"],or_masks_far)
        fr_far_inh = self.mask_mean(A["V1_Inh_L2/3"],or_masks_far)
        
        plots = {}
        gs = matplotlib.gridspec.GridSpecFromSubplotSpec(
            2, 30, subplot_spec=subplotspec, hspace=0.3, wspace=0.2
        )

        def set_ticks(ax):
            ax.tick_params(labelsize=9, size=4, width=1)
            ax.spines[['right','top']].set_visible(False)
            ax.spines[['left','bottom']].set_linewidth(1)

        axs = pylab.subplot(gs[0,0:7]);set_ticks(axs)
        m_e = A["V1_Exc_L2/3"].mean(axis=(1,2))
        m_i = A["V1_Inh_L2/3"].mean(axis=(1,2))
        axs.set_title("Mean population activity",fontsize=10)
        axs.plot(radii,np.ones_like(m_e) * m_e[0],'r:')
        axs.plot(radii,np.ones_like(m_i) * m_i[0],'b:')
        axs.plot(radii,m_e,'r')
        axs.plot(radii,m_i,'b')
        axs.set_xlim(0,radii[-1])
        axs.set_xlabel("Radius (um)")
        axs.set_ylabel("Firing rate (sp/s)")
        
        axs = pylab.subplot(gs[0,10:17]);set_ticks(axs)
        axs.set_title("Mean center activity",fontsize=10)
        m_e = A_masked["V1_Exc_L2/3"]["center_mean"]
        m_i = A_masked["V1_Inh_L2/3"]["center_mean"]
        axs.plot(radii,np.ones_like(m_e) * m_e[0],'r:')
        axs.plot(radii,np.ones_like(m_i) * m_i[0],'b:')
        axs.plot(radii,m_e,'r')
        axs.plot(radii,m_i,'b')
        axs.set_xlim(0,radii[-1])
        axs.set_xlabel("Radius (um)")
        axs.set_ylabel("Firing rate (sp/s)")
     
        axs = pylab.subplot(gs[0,20:27]);set_ticks(axs)
        axs.set_title("Mean surround activity",fontsize=10)
        m_e = A_masked["V1_Exc_L2/3"]["surround_mean"]
        m_i = A_masked["V1_Inh_L2/3"]["surround_mean"]
        axs.plot(radii,np.ones_like(m_e) * m_e[0],'r:')
        axs.plot(radii,np.ones_like(m_i) * m_i[0],'b:')
        axs.plot(radii,m_e,'r')
        axs.plot(radii,m_i,'b')
        axs.set_xlim(0,radii[-1])
        axs.set_xlabel("Radius (um)")
        axs.set_ylabel("Firing rate (sp/s)")
        
        axs = pylab.subplot(gs[1,0:7]);set_ticks(axs)
        axs.ticklabel_format(style='sci', axis='y', scilimits=(4,4))
        axs.plot(radii,A_masked["V1_Exc_L2/3"]["center_sum_total_spikes"],'r')
        axs.plot(radii,A_masked["V1_Inh_L2/3"]["center_sum_total_spikes"],'b')
        axs.spines[['top','right']].set_visible(False)
        axs.spines[['left','bottom']].set_linewidth(1.5)
        axs.set_xlim(0,radii[-1])
        axs.set_xlabel("Radii (um)")
        axs.set_ylabel("Total spikes over stim. period\nunder stim area")
        
        axs = pylab.subplot(gs[1,10:17]);set_ticks(axs)
        axs.plot(radii,A_masked["V1_Exc_L2/3"]["center_sum"] / A_masked["V1_Inh_L2/3"]["center_sum"],'k')
        axs.spines[['top','right']].set_visible(False)
        axs.spines[['left','bottom']].set_linewidth(1.5)
        axs.set_xlim(0,radii[-1])
        axs.set_xlabel("Radii (um)")
        axs.set_ylabel("E/I ratio of sum of firing rates\nunder stim area")
        
        axs = pylab.subplot(gs[1,20:27]);set_ticks(axs)
        axs.plot([radii[0],radii[-1]],[fr_close_exc[0],fr_close_exc[0]],'r:',label="_nolegend_")
        axs.plot([radii[0],radii[-1]],[fr_close_inh[0],fr_close_inh[0]],'b:',label="_nolegend_")
        axs.plot(radii,fr_close_exc,'r')
        axs.plot(radii,fr_far_exc,'r',linestyle=(5, (8, 3)))
        axs.plot(radii,fr_close_inh,'b')
        axs.plot(radii,fr_far_inh,'b',linestyle=(5, (8, 3)))
        axs.set_ylim(1,9)
        axs.set_xlim(0,radii[-1])
        axs.set_xlabel("Radius (um)")
        axs.set_ylabel("Firing rate (sp/s)")
        axs.legend(["Close or. exc","Far or. exc","Close or. inh","Far or. inh"],bbox_to_anchor=(1.0, 1.1),frameon=False)

        self.parameters.center_stim_dsv_list = "list"
        return plots

class PatternedOptogeneticStimulationPlot(MulhollandSmithPlots):   
    required_parameters = ParameterSet({
        "fullfield_stim_dsv": DataStoreView,
        "endogenous_stim_dsv": DataStoreView,
        "surrogate_stim_dsv": DataStoreView,
    })
    
    def calc_dsv_correlations(self,dsv,A,stims,t_res):
        onset_time = self.retrieve_ds_param_values(dsv, "onset_time")[0]
        offset_time = self.retrieve_ds_param_values(dsv, "offset_time")[0]
        corrs = np.zeros((A.shape[0],A.shape[1]))
        assert A.shape[0] == len(stims)
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                A_im = A[i,j,onset_time//t_res:(offset_time+500)//t_res,:,:].mean(axis=0)
                corrs[i][j], _ = scipy.stats.pearsonr(A_im.flatten(),stims[i,:,:].flatten())
        return corrs
        
    def retrieve_ds_param_values(self, dsv, param_name):
        l=[]
        for s in dsv.get_stimuli():
            if MozaikParametrized.idd(s).direct_stimulation_parameters != None:
                l.append(MozaikParametrized.idd(s).direct_stimulation_parameters.stimulating_signal_parameters[param_name])
        return sorted(list(set(l)))
        
    def tag_value(self, tag, tags):
        if len(tags) == 0:
            raise RuntimeError("No tags on recording!")
        filtered_tags = [t.split(":")[-1] for t in tags if ":".join(t.split(":")[:-1]) == tag]
        assert len(filtered_tags) == 1, "Duplicate tags are not allowed!"
        return eval(filtered_tags[0])

    def get_insides_outsides(self, resp, stims):
        insides = []
        outsides = []
    
        for i in range(resp.shape[0]):
            for j in range(resp.shape[1]):
                stim = stims[i,...].flatten()
                orr = resp[i,j,:,:,:].reshape([resp.shape[2],-1])
                inside = orr[:,stim > 0].mean(axis=1)
                outside = orr[:,stim == 0].mean(axis=1)
                insides.append(inside)
                outsides.append(outside)
                
        return np.stack(insides),np.stack(outsides)

    def normalize(self,x,nmin=None,nmax=None):
        if nmin==None:
            nmin=x.min()
        if nmax==None:
            nmax=x.max()
        return (x-nmin) / (nmax-nmin)

    def plot_corr(self, v, x, ax, color="k", scattercolor="silver"):
        h1 = ax.errorbar(
            x.mean(),
            np.mean(v),
            0,#np.std(v),
            marker="o",
            ls="",
            alpha=1,
            color=color,
            markersize=5,
            mec="k",
            lw=3,
            markeredgewidth=2,
        )[0]
        h2 = ax.scatter(np.ones(len(v)) * x + (np.random.rand(len(v))-0.5) * 0.4, v, alpha=0.4, marker='s',
                   color=scattercolor, s=20)
        bar_range = np.array([x[0]-0.25,x[0]+0.25])
        lw = 1.5
        t_value = scipy.stats.t.ppf((1 + 0.95) / 2, len(v) - 1)
        error = t_value * v.std() / np.sqrt(len(v))
        h3, = plt.plot(bar_range,(v.mean()+error) * np.ones(2),'-',color='k',lw=lw)
        plt.plot(bar_range,(v.mean()-error) * np.ones(2),'-',color='k',lw=lw)
        return h2,h1,h3

    def calc_r(self, entropy, fr):
        X = np.stack([entropy.flatten(),fr.flatten()]).T
        pca = PCA(n_components=1).fit(X)
        slope = pca.components_[0, 1] / pca.components_[0, 0]
        intercept = np.mean(X[:, 1]) - slope * np.mean(X[:, 0])
        r = scipy.stats.pearsonr(X[:,0],X[:,1])[0]
        return r, slope, intercept

    def plot_endo_surr_timecourse(self, ax, dsv, A_calc, stims, exp_t=None, exp_inside=None, exp_outside=None, plot_legend=False, y_max=None, y_min=None, title="Endogenous"):

        ins, outs = self.get_insides_outsides(A_calc, stims)
        
        onset_time = self.retrieve_ds_param_values(dsv, "onset_time")[0]
        offset_time = self.retrieve_ds_param_values(dsv, "offset_time")[0]
        duration = self.retrieve_ds_param_values(dsv, "duration")[0]

        colors = ["darkgreen","#1010ff",'limegreen',"lightskyblue"]
        t_ff_resp = np.linspace(0, duration / 1000, A_calc.shape[2])

        rect = matplotlib.patches.Rectangle(
            (1, 0),
            1,
            1.3,
            linewidth=1,
            facecolor=(255 / 255, 223 / 255, 0),
            alpha=0.6,
            #label="_nolegend_",
        )
        plt.gca().add_patch(rect)

        if title == "Endogenous":
            h5, = plt.plot(exp_t,self.normalize(np.array(exp_inside)),'-',c=colors[2],lw=3)
            h6, = plt.plot(exp_t,self.normalize(np.array(exp_outside),min(exp_inside),max(exp_inside)),"-",c=colors[3],lw=3)
        if y_max is None:
            y_max, y_min = max(ins.mean(axis=0)), min(ins.mean(axis=0))

        h2, = plt.plot(t_ff_resp, self.normalize(ins.mean(axis=0),y_min,y_max), "--", c=colors[0],lw=1.5)
        h4, = plt.plot(t_ff_resp, self.normalize(outs.mean(axis=0),y_min,y_max), "--", c=colors[1],lw=1.5)
        
        ax.set_xlabel("Time (s)", fontsize=10)
        ax.set_ylabel("Normalized\nresponse", fontsize=10,labelpad=-6)
        ax.set_xlim(0, 4)
        ax.set_ylim(0, 1.1)
        ax.set_xticks([0,1,2,3,4])
        ax.set_yticks([0,1])
        ax.set_title(title,fontsize=10)
        ax.tick_params(labelsize=9, size=4, width=1)
        ax.spines[['right','top']].set_visible(False)
        ax.spines[['left','bottom']].set_linewidth(1)

        if plot_legend:
            fake_h = plt.plot([], [], color=(0, 0, 0, 0), label=" ")[0]
            plt.legend(
                [fake_h,h5,h6,fake_h,h2,h4],
                ["Experiment","Within ROI","Outside ROI","Model","Within ROI","Outside ROI"],
                loc="upper center",
                handlelength = 1.1,
                bbox_to_anchor=(1.0, -0.2),
                frameon=False,
                fontsize=10,
                ncol=2,
                columnspacing=7
            )
        return y_max, y_min

    def plot_ff_timecourse(self, ax, dsv_ff_calc, exp_x, exp_y):
        trials = max([load_parameters(an.stimulus_id.replace("MozaikExtended",""))["trial"] for an in dsv_ff_calc.get_analysis_result()]) + 1
        ff_activity = np.array([queries.param_filter_query(dsv_ff_calc,st_trial=trial).get_analysis_result()[0].analog_signal for trial in range(trials)]).mean(axis=(-2,-1))

        onset_time = self.retrieve_ds_param_values(dsv_ff_calc.full_datastore, "onset_time")[0]
        offset_time = self.retrieve_ds_param_values(dsv_ff_calc.full_datastore, "offset_time")[0]
        duration = self.retrieve_ds_param_values(dsv_ff_calc.full_datastore, "duration")[0]

        rect = matplotlib.patches.Rectangle(
            (onset_time / 1000, 0),
            (offset_time - onset_time) / 1000,
            2,
            linewidth=1,
            facecolor=(255 / 255, 223 / 255, 0),
            alpha=0.6,
        )
        
        ax.add_patch(rect)
        ymin, ymax = 0,1.1
        
        t_ff_resp = np.linspace(0, duration / 1000, ff_activity.shape[1])
        mean, sem = ff_activity.mean(axis=0), ff_activity.std(
            axis=0
        ) / np.sqrt(len(ff_activity))

        h1, = ax.plot(exp_x, exp_y,c='k')
        h3, = ax.plot(t_ff_resp, mean, "-", c="r")
        
        ax.spines[['right','top']].set_visible(False)
        ax.spines[['bottom','left']].set_linewidth(1.5)
        
        fs = 10
        ax.set_xlabel("Time (s)",fontsize=fs)
        ax.set_ylabel("Population mean $\\Delta$F/F",fontsize=fs)
        ax.set_xticks([0,1,2,3,4],labels=["0","1","2","3","4"],fontsize=fs)
        ax.set_yticks([0,1.5],labels=["0","0.15"],fontsize=fs)
        
        leg = plt.legend([h1,h3,rect],['Experiment','Mean\nresp.','Light\nON'],fontsize=10,frameon=True,handlelength=1.5,loc='upper left')
        
        ax.legend(
            [rect,h1,h3],
            ['Light ON','Experiment','Model'],
            loc="upper center",
            handlelength = 1.0,
            bbox_to_anchor=(0.36, 1.35),
            frameon=False,
            fontsize=10,
        )
        ax.yaxis.set_label_coords(-0.09,0.45)
        ax.set_ylim(0,1.7)
        ax.set_xlim(0,4)
        ax.tick_params(labelsize=9, size=4, width=1.5)

    def get_stim_response_and_patterns(self, dsv):
        trials = max([load_parameters(an.stimulus_id.replace("MozaikExtended",""))["trial"] for an in dsv.get_analysis_result()]) + 1
        # Retrieve and sort unique stim pattern paths
        stim_pattern_paths = sorted(list(set([load_parameters(ar.stimulus_id.replace("MozaikExtended",""))["direct_stimulation_parameters"]["stimulating_signal_parameters"]['image_path'] for ar in queries.param_filter_query(dsv,st_trial=0).get_analysis_result()])), key=lambda s: int(s.split('/')[-1].split('.')[0]))
        stim_patterns = np.array([queries.param_filter_query(dsv.full_datastore,analysis_algorithm="SaveStimPatterns",st_identifier=stim_path).get_analysis_result()[0].value for stim_path in stim_pattern_paths])
        
        act = [[] for i in range(len(stim_pattern_paths))]
        for i in range(len(stim_pattern_paths)):
            for trial in range(trials):
                ars = queries.param_filter_query(dsv,st_trial=trial).get_analysis_result()
                act[i].append(np.array(ars[[stim_pattern_paths[i] in str(ar) for ar in ars].index(True)].analog_signal))
        act = np.array(act)
        return stim_patterns, stim_pattern_paths, act

    def get_stim_daod(self, dsv, stim_pattern_path):
        dsv = queries.param_filter_query(dsv,value_name="DAOD")
        trials = max([load_parameters(an.stimulus_id.replace("MozaikExtended",""))["trial"] for an in dsv.get_analysis_result()]) + 1
        return np.array([[ar.value for ar in queries.param_filter_query(dsv,st_trial=trial).get_analysis_result() if stim_pattern_path in str(ar)][0] for trial in range(trials)])
    
    def subplot(self, subplotspec):
        ff_res = queries.param_filter_query(self.parameters.fullfield_stim_dsv,analysis_algorithm="RecordingArrayTimecourse").get_analysis_result()[0]
        s_res, t_res, array_width = self.tag_value("s_res", ff_res.tags), self.tag_value("t_res", ff_res.tags), self.tag_value("array_width", ff_res.tags)
        
        dsv_ff_calc = queries.param_filter_query(self.parameters.fullfield_stim_dsv,y_axis_name="Calcium imaging signal (normalized)",st_name="InternalStimulus")
        dsv_ff = queries.param_filter_query(self.parameters.fullfield_stim_dsv,analysis_algorithm="GaussianBandpassFilter",st_name="InternalStimulus")
        opt_correlation_maps = queries.param_filter_query(self.parameters.fullfield_stim_dsv,analysis_algorithm="CorrelationMaps").get_analysis_result()[0].value
        spont_correlation_maps = queries.param_filter_query(self.datastore,analysis_algorithm="CorrelationMaps",sheet_name="V1_Exc_L2/3",st_name='InternalStimulus').get_analysis_result()[0].value
        opt_spont_similarity_map = queries.param_filter_query(self.parameters.fullfield_stim_dsv,analysis_algorithm="CorrelationMapSimilarity").get_analysis_result()[0].value
        or_map = queries.param_filter_query(self.datastore,value_name="orientation map",sheet_name="V1_Exc_L2/3").get_analysis_result()[0].value
        
        # Endogenous / surrogate stimulation timecourse
        _, _, A_endo = self.get_stim_response_and_patterns(queries.param_filter_query(self.parameters.endogenous_stim_dsv,analysis_algorithm="RecordingArrayTimecourse"))
        _, _, A_surr = self.get_stim_response_and_patterns(queries.param_filter_query(self.parameters.surrogate_stim_dsv,analysis_algorithm="RecordingArrayTimecourse"))         
        endo_stims, endo_stim_paths, A_endo_calc = self.get_stim_response_and_patterns(queries.param_filter_query(self.parameters.endogenous_stim_dsv,y_axis_name="Calcium imaging signal (normalized)"))
        surr_stims, surr_stim_paths, A_surr_calc = self.get_stim_response_and_patterns(queries.param_filter_query(self.parameters.surrogate_stim_dsv,y_axis_name="Calcium imaging signal (normalized)"))
        A_spont = np.array(queries.param_filter_query(self.datastore,analysis_algorithm="RecordingArrayTimecourse",sheet_name="V1_Exc_L2/3",st_name="InternalStimulus",y_axis_name='recording array timecourse').get_analysis_result()[0].analog_signal)

        onset_time, offset_time = self.retrieve_ds_param_values(self.parameters.endogenous_stim_dsv, "onset_time")[0], self.retrieve_ds_param_values(self.parameters.endogenous_stim_dsv, "offset_time")[0]        
        daod_spont = queries.param_filter_query(self.datastore,value_name="DAOD",sheet_name="V1_Exc_L2/3").get_analysis_result()[0].value
        daod_endo = np.array([self.get_stim_daod(self.parameters.endogenous_stim_dsv, p) for p in endo_stim_paths])[:,:,onset_time//t_res:offset_time//t_res,:]
        daod_surr = np.array([self.get_stim_daod(self.parameters.surrogate_stim_dsv, p) for p in surr_stim_paths])[:,:,onset_time//t_res:offset_time//t_res,:]
        or_bins = np.linspace(0,np.pi,daod_spont.shape[1]+1)
        or_hist_endo = np.array([np.histogram(or_map[endo_stims[i] == 1],bins=or_bins)[0] for i in range(len(endo_stims))])
        or_hist_surr = np.array([np.histogram(or_map[surr_stims[i] == 1],bins=or_bins)[0] for i in range(len(surr_stims))])

        or_hist_endo_entropy = scipy.stats.entropy(or_hist_endo, axis=-1)
        or_hist_surr_entropy = scipy.stats.entropy(or_hist_surr, axis=-1)
        or_hist_A_endo_entropy = scipy.stats.entropy(daod_endo.mean(axis=(1,2)), axis=-1)
        or_hist_A_surr_entropy = scipy.stats.entropy(daod_surr.mean(axis=(1,2)), axis=-1)
        spont_or_hist_entropy = scipy.stats.entropy(daod_spont, axis=-1)        

        d_ff = self.get_experimental_data()["Mulholland 2024 may"]
        d_endo = self.get_experimental_data()["Mulholland 2024 january"]["endo_inside_outside"]
        plots = {}
        gs = matplotlib.gridspec.GridSpecFromSubplotSpec(
            10, 25, subplot_spec=subplotspec, hspace=0.3, wspace=0.2
        )

        self.plot_ff_timecourse(pylab.subplot(gs[0:3, 0:4]), dsv_ff_calc, d_ff["fullfield_timecourse"]["x"],d_ff["fullfield_timecourse"]["y"])

        cmap_idx = 8186
        cmap_x, cmap_y = cmap_idx % opt_correlation_maps.shape[-1], cmap_idx // opt_correlation_maps.shape[-1]

        # Spontaneous correlation map
        ax = pylab.subplot(gs[1:3,5:9])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        cmap = pylab.imshow(spont_correlation_maps[cmap_idx],'bwr',vmin=-1,vmax=1)
        pylab.title("Spontaneous act.\ncorrelation map",fontsize=10)
        ax.scatter(cmap_x,cmap_y,color='k',marker='x')
        cbar = pylab.colorbar(cmap)
        cbar.set_label(label='Correlation', labelpad=5, fontsize=10)
        cbar.set_ticks([-1,1],labels=["-1","1"])
        
        # Opto correlation map
        ax = pylab.subplot(gs[1:3,10:14])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        cmap = pylab.imshow(opt_correlation_maps[cmap_idx],'bwr',vmin=-1,vmax=1)
        pylab.title("Fullfield opto\ncorrelation map",fontsize=10)
        ax.scatter(cmap_x,cmap_y,color='k',marker='x')
        cbar = pylab.colorbar(cmap)
        cbar.set_label(label='Correlation', labelpad=5, fontsize=10)
        cbar.set_ticks([-1,1],labels=["-1","1"])
        
        # Similarity map
        ax=pylab.subplot(gs[1:3, 15:19])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        smap = pylab.imshow(opt_spont_similarity_map.reshape(opt_correlation_maps[cmap_idx].shape), vmin=-1, vmax=1,cmap='hot')
        pylab.title("Fullfield opto-spont\ncorr. map similarity", fontsize=10)
        cbar = pylab.colorbar(smap)
        cbar.set_label(label='Similarity', labelpad=5, fontsize=10)    
        cbar.set_ticks([-1,1])

        self.single_metric_plot(pylab.subplot(gs[0:3, 22:25]),d_ff['exc opt similarity'],opt_spont_similarity_map.mean(),[0,1],"Exc $\\rightarrow$ opto similarity")

        endo_ymax, endo_ymin = self.plot_endo_surr_timecourse(pylab.subplot(gs[5:8, 0:4]), self.parameters.endogenous_stim_dsv, A_endo_calc, endo_stims, d_endo["t"], d_endo["inside"], d_endo["outside"],plot_legend=True)
        self.plot_endo_surr_timecourse(pylab.subplot(gs[5:8, 5:9]), self.parameters.surrogate_stim_dsv, A_surr_calc, surr_stims, y_max=endo_ymax, y_min=endo_ymin, title="Surrogate")

        osc = ["#d62728", "#bcbd22","#e377c2", "#1f77b4"]
        # Endo / surr response correlation to stimulation pattern
        ax0 = pylab.subplot(gs[5:8, 10:14])
        endo_corrs = self.calc_dsv_correlations(self.parameters.endogenous_stim_dsv,A_endo_calc,endo_stims,t_res)
        surr_corrs = self.calc_dsv_correlations(self.parameters.surrogate_stim_dsv,A_surr_calc,surr_stims,t_res)
        h1,h2,h3 = self.plot_corr(endo_corrs.flatten(), 0 * np.ones_like(endo_corrs.flatten()), ax0)
        self.plot_corr(surr_corrs.flatten(), 1 * np.ones_like(surr_corrs.flatten()), ax0)
        fs = 10
        ax0.set_ylim(0.5, 0.9)
        ax0.tick_params(labelsize=9, size=4, width=1)
        ax0.spines[['right','top']].set_visible(False)
        ax0.spines[['left','bottom']].set_linewidth(1)
        ax0.set_yticks([0.5, 0.9],labels=["0.5", "0.8"],fontsize=10)
        ax0.set_xticks([0,1],labels=["Endog", "Surr"],fontsize=10)
        ax0.set_ylabel("Stimulus-\nresponse\ncorrelation",fontsize=9,labelpad=-16.2)
        ax0.legend([h1,h3,h2],["Trial","95% Confidence\nInterval","Mean"],fontsize=10,
                    handlelength = 0.75,
                    bbox_to_anchor=(1, -0.2),
                    frameon=False,
                   ncol=2, columnspacing=-4.5
                  )
        
        ax=pylab.subplot(gs[5:8, 15:19])
        fontsize=10
        ax.plot(or_hist_endo_entropy,or_hist_A_endo_entropy.reshape([8,-1]),'.',ms=8,c=osc[0])
        ax.plot(or_hist_surr_entropy,or_hist_A_surr_entropy.reshape([8,-1]),'.',ms=8,c=osc[2])
        ax.plot(np.linspace(2.0,3.5),np.linspace(2.0,3.5),'k--',lw=1.3)
        h0, = ax.plot([0],[0],'.',ms=10,c=osc[0])
        h1, = ax.plot([0],[0],'.',ms=10,c=osc[2])
        ax.set_xlabel("OD\nentropy",fontsize=fontsize,labelpad=-10)
        ax.set_ylabel("DAOD\nentropy",fontsize=fontsize,labelpad=-13)
        ax.set_xlim(2.0,3.5)
        ax.set_ylim(2.0,3.5)
        ax.set_xticks([2.0,3.5],labels=["2.0","3.5"],fontsize=fontsize)
        ax.set_yticks([2.0,3.5],labels=["2.0","3.5"],fontsize=fontsize)
        ax.tick_params(labelsize=fontsize, size=3, width=1)
        ax.spines[['right','top']].set_visible(False)
        ax.spines[['left','bottom']].set_linewidth(1)
        
        h_spont, = ax.plot([0],[0],'.',ms=5,c='k')
        h0, = ax.plot([0],[0],'.',ms=5,c=osc[0])
        h1, = ax.plot([0],[0],'.',ms=5,c=osc[2])
        ax.legend([h0,h1,h_spont],["Endogenous","Surrogate",'Spontaneous'],handlelength=1,fontsize=fontsize,
                   bbox_to_anchor=(-0.15, -0.20),loc='upper left',frameon=False)

        # DAOD entropy vs. firing rate
        ax=pylab.subplot(gs[5:8, 20:24])
        fontsize=10

        A_endo_fr = np.array([[[A_endo[i,j,k,:,:][endo_stims[i] == 1] for k in range(A_endo.shape[2])] for j in range(A_endo.shape[1])] for i in range(A_endo.shape[0])])[:,:,onset_time//t_res:offset_time//t_res,:].mean(axis=3)
        A_surr_fr = np.array([[[A_surr[i,j,k,:,:][surr_stims[i] == 1] for k in range(A_surr.shape[2])] for j in range(A_surr.shape[1])] for i in range(A_surr.shape[0])])[:,:,onset_time//t_res:offset_time//t_res,:].mean(axis=3)

        r_endo, slope_endo, intercept_endo = self.calc_r(scipy.stats.entropy(daod_endo, axis=-1), A_endo_fr)
        r_surr, slope_surr, intercept_surr = self.calc_r(scipy.stats.entropy(daod_surr, axis=-1), A_surr_fr)
        r_spont, slope_spont, intercept_spont = self.calc_r(spont_or_hist_entropy, A_spont.mean(axis=(1,2)))  
        
        x_endo = np.linspace(2.35,2.68,100)
        x_surr = np.linspace(3.2,3.35100)
        x_spont = np.linspace(3.1,3.42,100)
        ax.plot(x_endo,intercept_endo + slope_endo * x_endo,lw=0.7,c=osc[0])
        ax.text(2.62,22.5,"r=%.2f" % r_endo,fontsize=6,c=osc[0])
        ax.plot(scipy.stats.entropy(daod_endo, axis=-1).flatten(),A_endo_fr.flatten(),'.',ms=8,c=osc[0],alpha=0.05)
        ax.plot(scipy.stats.entropy(daod_surr, axis=-1).flatten(),A_surr_fr.flatten(),'.',ms=8,c=osc[2],alpha=0.05)
        ax.plot(spont_or_hist_entropy,A_spont.mean(axis=(1,2)),'.',ms=8,c='k',alpha=0.05)
        ax.set_ylabel("Firing\nrate (sp/s)",fontsize=fontsize,labelpad=-8)
        ax.set_xlabel("DAOD\nentropy",fontsize=fontsize,labelpad=-9)
        ax.set_xlim(2.2,3.5)
        ax.set_ylim(0,25)
        ax.set_xticks([2.2,3.5],labels=["2.2","3.5"],fontsize=fontsize)
        ax.set_yticks([0,25],labels=["0","25"],fontsize=fontsize)
        ax.tick_params(labelsize=fontsize, size=3, width=1)
        ax.spines[['right','top']].set_visible(False)
        ax.spines[['left','bottom']].set_linewidth(1)

        # We need to do this because Mozaik cannot elegantly serialize DataStoreViews
        self.parameters.fullfield_stim_dsv = "DataStoreView"
        self.parameters.endogenous_stim_dsv = "DataStoreView"
        self.parameters.surrogate_stim_dsv = "DataStoreView"
        return plots

class IndividualDAODPlot(MulhollandSmithPlots):   
    required_parameters = ParameterSet({
        "endogenous_stim_dsv": DataStoreView,
        "surrogate_stim_dsv": DataStoreView,
    })

    def retrieve_ds_param_values(self, dsv, param_name):
        l=[]
        for s in dsv.get_stimuli():
            if MozaikParametrized.idd(s).direct_stimulation_parameters != None:
                l.append(MozaikParametrized.idd(s).direct_stimulation_parameters.stimulating_signal_parameters[param_name])
        return sorted(list(set(l)))
        
    def tag_value(self, tag, tags):
        if len(tags) == 0:
            raise RuntimeError("No tags on recording!")
        filtered_tags = [t.split(":")[-1] for t in tags if ":".join(t.split(":")[:-1]) == tag]
        assert len(filtered_tags) == 1, "Duplicate tags are not allowed!"
        return eval(filtered_tags[0])

    def get_stim_response_and_patterns(self, dsv):
        trials = max([load_parameters(an.stimulus_id.replace("MozaikExtended",""))["trial"] for an in dsv.get_analysis_result()]) + 1
        # Retrieve and sort unique stim pattern paths
        stim_pattern_paths = sorted(list(set([load_parameters(ar.stimulus_id.replace("MozaikExtended",""))["direct_stimulation_parameters"]["stimulating_signal_parameters"]['image_path'] for ar in queries.param_filter_query(dsv,st_trial=0).get_analysis_result()])), key=lambda s: int(s.split('/')[-1].split('.')[0]))
        stim_patterns = np.array([queries.param_filter_query(dsv.full_datastore,analysis_algorithm="SaveStimPatterns",st_identifier=stim_path).get_analysis_result()[0].value for stim_path in stim_pattern_paths])
        
        act = [[] for i in range(len(stim_pattern_paths))]
        for i in range(len(stim_pattern_paths)):
            for trial in range(trials):
                ars = queries.param_filter_query(dsv,st_trial=trial).get_analysis_result()
                act[i].append(np.array(ars[[stim_pattern_paths[i] in str(ar) for ar in ars].index(True)].analog_signal))
        act = np.array(act)
        return stim_patterns, stim_pattern_paths, act

    def get_stim_daod(self, dsv, stim_pattern_path):
        dsv = queries.param_filter_query(dsv,value_name="DAOD")
        trials = max([load_parameters(an.stimulus_id.replace("MozaikExtended",""))["trial"] for an in dsv.get_analysis_result()]) + 1
        return np.array([[ar.value for ar in queries.param_filter_query(dsv,st_trial=trial).get_analysis_result() if stim_pattern_path in str(ar)][0] for trial in range(trials)])

    def color_to_rgb(self,c):
        if (type(c) == list or type(c) == tuple) and len(c) == 3:
            return c
        if type(c) == str:
            if c[0] == "#":
                return matplotlib.colors.hex2color(c)
            else:
                return self.color_to_rgb(matplotlib.colors.cnames[c])

    def fold_in_half(self,array):
        return np.hstack([array[len(array)//2:],array[:len(array)//2]])
    
    def subplot(self, subplotspec):
        osc = ["#d62728", "#bcbd22","#e377c2", "#1f77b4"]
        gs = matplotlib.gridspec.GridSpecFromSubplotSpec(
            8,5, subplot_spec=subplotspec, hspace=0.6,wspace=0.9
        )
        plots={}
        colors = [osc[0], osc[2]]
        or_map = queries.param_filter_query(self.datastore,value_name="orientation map",sheet_name="V1_Exc_L2/3").get_analysis_result()[0].value
        res = queries.param_filter_query(self.parameters.endogenous_stim_dsv,analysis_algorithm="RecordingArrayTimecourse").get_analysis_result()[0]
        s_res, t_res, array_width = self.tag_value("s_res", res.tags), self.tag_value("t_res", res.tags), self.tag_value("array_width", res.tags)
        endo_stims, endo_stim_paths, A_endo = self.get_stim_response_and_patterns(queries.param_filter_query(self.parameters.endogenous_stim_dsv,analysis_algorithm="RecordingArrayTimecourse"))
        surr_stims, surr_stim_paths, A_surr = self.get_stim_response_and_patterns(queries.param_filter_query(self.parameters.surrogate_stim_dsv,analysis_algorithm="RecordingArrayTimecourse"))         
    
        onset_time, offset_time = self.retrieve_ds_param_values(self.parameters.endogenous_stim_dsv, "onset_time")[0], self.retrieve_ds_param_values(self.parameters.endogenous_stim_dsv, "offset_time")[0]        
        daod_endo = np.array([self.get_stim_daod(self.parameters.endogenous_stim_dsv, p) for p in endo_stim_paths])[:,:,onset_time//t_res:offset_time//t_res,:]
        daod_surr = np.array([self.get_stim_daod(self.parameters.surrogate_stim_dsv, p) for p in surr_stim_paths])[:,:,onset_time//t_res:offset_time//t_res,:]
        or_bins = np.linspace(0,np.pi,daod_endo.shape[-1]+1)
        or_hist_endo = np.array([np.histogram(or_map[endo_stims[i] == 1],bins=or_bins)[0] for i in range(len(endo_stims))])
        or_hist_surr = np.array([np.histogram(or_map[surr_stims[i] == 1],bins=or_bins)[0] for i in range(len(surr_stims))])

        ors = or_bins[:-1] + or_bins[1]/2
        or_hist_endo_entropy = scipy.stats.entropy(or_hist_endo, axis=-1)
        or_hist_surr_entropy = scipy.stats.entropy(or_hist_surr, axis=-1)
        or_hist_A_endo_entropy = scipy.stats.entropy(daod_endo.mean(axis=(1,2)), axis=-1)
        or_hist_A_surr_entropy = scipy.stats.entropy(daod_surr.mean(axis=(1,2)), axis=-1)

        arrs_or = [or_hist_endo, or_hist_surr]
        arrs_A = [daod_endo, daod_surr]
    
        axs = np.array([[pylab.subplot(gs[i,j]) for j in range(5)] for i in range(8)])
        
        ax = axs.T
        fontsize = 7
        for i in range(8):
            ax[0,i].set_title(f"Pattern {i}",fontsize=fontsize)
            for k in range(2):
                pattern = surr_stims[i,...] if k == 1  else endo_stims[i,...]
                ax[k*2,i].imshow(np.ones_like(pattern) * 0.3,cmap='gray',vmin=0,vmax=1)
                for j in range(2):
                    # Stim patterns
                    pattern_img = np.zeros((100, 100, 4), dtype=np.uint8)
                    pattern_img[:, :, :3][pattern == j] = (np.array(self.color_to_rgb(osc[k*2+1-j])) * 255).astype(np.uint8)
                    pattern_img[:, :, 3][pattern == j] = int(0.8 * 255)
                    ax[k*2,i].imshow(pattern_img)
                    ax[k*2,i].get_xaxis().set_visible(False)
                    ax[k*2,i].get_yaxis().set_visible(False)
        
        
                # Line plots
        
                aa = arrs_A[k][i].mean(axis=(0, 1))
                h1, = ax[k*2+1,i].plot(ors-np.pi/2, self.fold_in_half(aa/np.trapz(aa,ors)),c=colors[k],lw=1.5)
                (h0,) = ax[k*2+1,i].plot(
                    ors-np.pi/2, self.fold_in_half(arrs_or[k][i, :] / np.trapz(arrs_or[k][i, :],ors)), c = "k", lw=1.5
                )
                bbox = (1.15, 2.0) if k == 0 else (1.15, 2.0)
                if i == 0:
                    leg = ax[k*2+1,i].legend(
                        [h0, h1],
                        ["Stimulated or.", "DAOD"],
                        title="Endogenous" if k == 0 else "Surrogate",
                        title_fontsize=fontsize,
                        handlelength=0.8,
                        fontsize=fontsize,
                        bbox_to_anchor=bbox,
                        frameon=False,
                    )
                    leg._legend_box.align = "left"
                ax[k*2+1,i].set_xlabel("Orientation", fontsize=fontsize, labelpad=-7.5)
                ax[k*2+1,i].set_ylabel("Density", fontsize=fontsize, labelpad=-6)
                ax[k*2+1,i].set_xlim(-np.pi/2, np.pi/2)
                ax[k*2+1,i].set_xticks([-np.pi/2, np.pi/2], ["$-\pi/2$", "$\pi/2$"], fontsize=fontsize)
                ax[k*2+1,i].set_ylim(0, 2)
                ax[k*2+1,i].set_yticks([0, 2])
        
                ax[k*2+1,i].tick_params(labelsize=fontsize, size=3, width=1)
                ax[k*2+1,i].spines[["right", "top"]].set_visible(False)
                ax[k*2+1,i].spines[["left", "bottom"]].set_linewidth(1)
            
            h0,= ax[4,i].plot(or_hist_A_endo_entropy[i],'s',c=colors[0])
            h1,= ax[4,i].plot(or_hist_A_surr_entropy[i],'s',c=colors[1])
            h2,= ax[4,i].plot(or_hist_endo_entropy[i],'o',c=colors[0],mec='k',mew=1.5)
            h3,= ax[4,i].plot(or_hist_surr_entropy[i],'o',c=colors[1],mec='k',mew=1.5)
            ax[4,i].set_ylim(2.2,3.5)
            ax[4,i].set_yticks([2.2,3.5])
            ax[4,i].set_xticks([])
            ax[4,i].tick_params(labelsize=fontsize, size=3, width=1)
            ax[4,i].spines[["right", "top"]].set_visible(False)
            ax[4,i].spines[["left", "bottom"]].set_linewidth(1)
            ax[4,i].set_ylabel("Entropy",fontsize=fontsize,labelpad=-9)
            #ax[4,i].plot([-np.pi/2, np.pi/2],np.log2(len(ors)) * np.ones((2)),'k--')
            #fig.savefig(
            #    "fig5_plots/D%d.svg" % k, dpi=dpi, bbox_inches="tight", transparent=True
            #)
        
        leg = ax[4,0].legend(
                        [h0, h1, h2, h3],
                        ["Endogenous","Surrogate","Endogenous","Surrogate"],
                        title="DAOD                     Stimulated or.",
                        title_fontsize=fontsize,
                        ncols = 2,
                        handlelength=0.5,
                        fontsize=fontsize,
                        bbox_to_anchor=(2.15, 2.0),
                        frameon=False,
        )
        leg._legend_box.align = "left"
        h0 = plt.gca().fill_between(
            [0],
            [0],
            [0.00000001],
            color=osc[0],
            alpha=1,
        )
        h1 = plt.gca().fill_between(
            [0],
            [0],
            [0.00000001],
            color=osc[1],
            alpha=1,
        )
        leg = ax[0,0].legend(
                        [h0, h1],
                        ["Stimulated area","Non-stim. area"],
                        title="Endogenous",
                        title_fontsize=fontsize,
                        handlelength=0.8,
                        fontsize=fontsize,
                        bbox_to_anchor=(1.3, 2.0),
                        frameon=False,
        )
        leg._legend_box.align = "left"
        h0 = plt.gca().fill_between(
            [0],
            [0],
            [0.00000001],
            color=osc[2],
            alpha=1,
        )
        h1 = plt.gca().fill_between(
            [0],
            [0],
            [0.00000001],
            color=osc[3],
            alpha=1,
        )
        leg = ax[2,0].legend(
                        [h0, h1],
                        ["Stimulated area","Non-stim. area"],
                        title="Surrogate",
                        title_fontsize=fontsize,
                        handlelength=0.8,
                        fontsize=fontsize,
                        bbox_to_anchor=(1.3, 2.0),
                        frameon=False,
        )
        leg._legend_box.align = "left"
        plt.text(-0.92,18.2,"A",fontsize=14,weight='bold')
        plt.text(-0.76,18.2,"B",fontsize=14,weight='bold')
        plt.text(-0.51,18.2,"C",fontsize=14,weight='bold')
        plt.text(-0.34,18.2,"D",fontsize=14,weight='bold')
        plt.text(-0.12,18.2,"E",fontsize=14,weight='bold')

        # We need to do this because Mozaik cannot elegantly serialize DataStoreViews
        self.parameters.endogenous_stim_dsv = "DataStoreView"
        self.parameters.surrogate_stim_dsv = "DataStoreView"
        return plots

class ChernovPlot(MulhollandSmithPlots):   
    required_parameters = ParameterSet({
        "visual_dsv": DataStoreView,
        "visual_opto_dsv": DataStoreView,
        "time_cutoff_ms": int,
    })

    def tag_value(self, tag, tags):
        if len(tags) == 0:
            raise RuntimeError("No tags on recording!")
        filtered_tags = [t.split(":")[-1] for t in tags if ":".join(t.split(":")[:-1]) == tag]
        assert len(filtered_tags) == 1, "Duplicate tags are not allowed!"
        return eval(filtered_tags[0])
    
    def significant_pixels(self, A):
            p_map = np.zeros((A.shape[-2], A.shape[-1]))
            for x in range(A.shape[-2]):
                for y in range(A.shape[-1]):
                    _, p_map[x, y] = scipy.stats.ttest_rel(A[:, 0, x, y], A[:, 1, x, y])
            p0 = 0.05
            p_map = p_map > (p0 / (A.shape[-2] * A.shape[-1]))
            or_map = (A[:,0,...] - A[:,1,...]).mean(axis=0)
            or_map[p_map] = np.nan
            return or_map

    # Format p-values
    def format_p(self, p):
        if p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        else:
            return f'p={p:.3f}'
    
    def add_p_value(self, ax, x1, x2, data1, data2, p, y_offset=0.05):
        """
        Add p-value annotation between two bar groups.
        
        Parameters:
        - ax: matplotlib axis object
        - x1, x2: x-positions of the bars to compare
        - data1, data2: arrays of values for each group (used to calculate height)
        - p: p-value to display
        - y_offset: vertical offset for the annotation
        """
        # Calculate max height from the data
        y = max(np.mean(data1), np.mean(data2)) + y_offset
        h = 0.2  # height of the "T" shape
        
        # Draw the comparison line
        ax.plot([x1, x1, x2, x2], 
                [y, y+h, y+h, y], 
                lw=1, color='k')
        
        # Add the p-value text
        ax.text((x1+x2)/2, y+h*1.5, 
                self.format_p(p), 
                ha='center', va='bottom')
    
    def subplot(self, subplotspec):
        res = queries.param_filter_query(self.datastore,analysis_algorithm="RecordingArrayTimecourse").get_analysis_result()[0]
        s_res, t_res, array_width = self.tag_value("s_res", res.tags), self.tag_value("t_res", res.tags), self.tag_value("array_width", res.tags)

        orientations_chernov = sorted(list(set([load_parameters(stim.replace("MozaikExtended",""))["orientation"] for stim in queries.param_filter_query(self.parameters.visual_opto_dsv,st_name="FullfieldDriftingSquareGrating").get_stimuli()])))
        orientations_chernov_visual = np.array(sorted(list(set([load_parameters(stim.replace("MozaikExtended",""))["orientation"] for stim in queries.param_filter_query(self.parameters.visual_dsv,st_name="FullfieldDriftingSquareGrating").get_stimuli()]))))
        n_trials_visual = max([load_parameters(stim.replace("MozaikExtended",""))["trial"] for stim in queries.param_filter_query(self.parameters.visual_dsv,st_name="FullfieldDriftingSquareGrating").get_stimuli()]) + 1
        n_trials_visual_opto = max([load_parameters(stim.replace("MozaikExtended",""))["trial"] for stim in queries.param_filter_query(self.parameters.visual_opto_dsv,st_name="FullfieldDriftingSquareGrating").get_stimuli()]) + 1
        n_trials = min(n_trials_visual,n_trials_visual_opto)
        
        A_spont_exc = np.array(queries.param_filter_query(self.datastore,analysis_algorithm="RecordingArrayTimecourse",st_name='InternalStimulus',sheet_name="V1_Exc_L2/3").get_analysis_result()[0].analog_signal)
        A_ch_visual = np.array([[np.array(queries.param_filter_query(self.parameters.visual_dsv,sheet_name="V1_Exc_L2/3",analysis_algorithm="RecordingArrayTimecourse",st_trial=trial,st_name="FullfieldDriftingSquareGrating",st_orientation=orientation).get_analysis_result()[0].analog_signal) for orientation in orientations_chernov_visual] for trial in range(n_trials)])
        A_ch_visuopto = np.array([[np.array(queries.param_filter_query(self.parameters.visual_opto_dsv,sheet_name="V1_Exc_L2/3",analysis_algorithm="RecordingArrayTimecourse",st_trial=trial,st_name="FullfieldDriftingSquareGrating",st_orientation=orientation).get_analysis_result()[0].analog_signal) for orientation in orientations_chernov] for trial in range(n_trials)])
        A_ch_visual -= A_spont_exc.mean(axis=0)
        A_ch_visuopto -= A_spont_exc.mean(axis=0)

        sp_visual = self.significant_pixels(A_ch_visual[:,:,:self.parameters.time_cutoff_ms//t_res,:,:].mean(axis=(2)))
        sp_visuopto = self.significant_pixels(A_ch_visuopto[:,:,:self.parameters.time_cutoff_ms//t_res,:,:].mean(axis=(2)))
        
        y, x = np.ogrid[:sp_visuopto.shape[0], :sp_visuopto.shape[1]]
        mask_2_mm_circle = (x - sp_visuopto.shape[0]//2) ** 2 + (y - sp_visuopto.shape[1]//2) ** 2 > (sp_visuopto.shape[0]//2) ** 2  # Circular mask
        center_circle_radius_um = 200
        mask_center_circle = (x - sp_visuopto.shape[0]//2) ** 2 + (y - sp_visuopto.shape[1]//2) ** 2 < int(np.ceil(center_circle_radius_um/s_res)) ** 2  # Circular mask
        sp_visual[mask_2_mm_circle] = np.nan
        sp_visuopto[mask_2_mm_circle] = np.nan
        sp_visual[mask_center_circle] = np.nan
        sp_visuopto[mask_center_circle] = np.nan

        # Difference across trials, mean across area
        _, p_vertical = scipy.stats.wilcoxon(
            np.mean([A_ch_visual[trial,...].mean(axis=1)[0][sp_visual > 0] for trial in range(n_trials)], axis=1), # Response per trial (visual-only)
            np.mean([A_ch_visuopto[trial,...].mean(axis=1)[0][sp_visuopto > 0] for trial in range(n_trials)], axis=1) # Response per trial (visual + opto)
        )
        _, p_horizontal = scipy.stats.wilcoxon(
            np.mean([A_ch_visual[trial,...].mean(axis=1)[1][sp_visual < 0] for trial in range(n_trials)], axis=1),   
            np.mean([A_ch_visuopto[trial,...].mean(axis=1)[1][sp_visuopto < 0] for trial in range(n_trials)], axis=1)
        )

        plots = {}
        gs = matplotlib.gridspec.GridSpecFromSubplotSpec(
            10, 20, subplot_spec=subplotspec, hspace=0.3, wspace=0.2
        )

        for i, title, A in zip(range(4),["Visual\nhorizontal","Visual\nvertical","Visual + opto\nhorizontal","Visual + opto\nvertical"],
                                [A_ch_visual[:,0,:,:,:],A_ch_visual[:,1,:,:,:],A_ch_visuopto[:,0,:,:,:],A_ch_visuopto[:,1,:,:,:]]):
            ax = pylab.subplot(gs[:4,i*5:i*5+4])
            ax.set_facecolor('k')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title,fontsize=10)
            im = ax.imshow(A.mean(axis=(0,1)),cmap='gray',vmin=0,vmax=80)
            pylab.colorbar(im,ax=ax,label="Firing rate diff.\nfrom spont. (sp/s)")

        for i, title, sp in zip(range(2),["Visual","Visual + opto"],[sp_visual,sp_visuopto]):
            ax = pylab.subplot(gs[6:,i*5:i*5+4])
            ax.set_facecolor('k')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title,fontsize=10)
            im = ax.imshow(sp,cmap='bwr_r',vmin=-30,vmax=30)
            cbar = pylab.colorbar(im,ax=ax,label="Firing rate\ndiff. (sp/s)")
            cbar.set_ticks([-30,0, 30])#,labels=["0","$\pi$"],fontsize=9)

        ax = pylab.subplot(gs[6:,12:15])
        ax.spines[['right','top']].set_visible(False)
        ax.spines[['bottom','left']].set_linewidth(1.5)
        y_h_vis = np.mean(A_ch_visual.mean(axis=(0,2))[0][sp_visual > 0])
        y_v_vis = np.mean(A_ch_visual.mean(axis=(0,2))[1][sp_visual < 0])
        y_h_vis_opto = np.mean(A_ch_visuopto.mean(axis=(0,2))[0][sp_visuopto > 0])
        y_v_vis_opto = np.mean(A_ch_visuopto.mean(axis=(0,2))[1][sp_visuopto < 0])
        ax.bar([0,2],[y_h_vis,y_h_vis_opto],color='darkblue')
        ax.bar([1,3],[y_v_vis,y_v_vis_opto],color='crimson')
        ax.set_xticks([0,1,2,3])
        ax.set_ylim(14,17)
        ax.set_yticks([14,17])
        ax.set_xticklabels(["        Visual","","          Visual+\n        opto",""], ha='center')
        plt.legend(["Iso","Ortho"],fontsize=9,
            handlelength = 0.75,
            bbox_to_anchor=(1.05, 1.2),
            frameon=False
          )
        ax.set_ylabel("Firing rate (sp/s)")
        self.add_p_value(ax, 0, 2, y_h_vis, y_h_vis_opto, p_horizontal)
        self.add_p_value(ax, 1, 3, y_v_vis, y_v_vis_opto, p_vertical,1)
        
        # We need to do this because Mozaik cannot elegantly serialize DataStoreViews
        self.parameters.visual_dsv = "DataStoreView"
        self.parameters.visual_opto_dsv = "DataStoreView"
        
        return plots
