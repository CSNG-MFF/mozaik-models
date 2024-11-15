import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from mozaik.storage.datastore import PickledDataStore
import logging
import sys
import matplotlib.gridspec as gs
import os
from msa_analysis_functions import *

experimental_data = get_experimental_data()
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

assert len(sys.argv) == 4, "Usage: python msa_analysis.py /path/to/spontaneous_activity_datastore /path/to/patterned_optogenetic_stimulation_datastore /path/to/central_optogenetic_stimulation_datastore"
spont_ds_path, patterned_ds_path, center_stim_ds_path = sys.argv[1:]

assert os.path.isdir(spont_ds_path), "Spontaneous activity datastore directory not found!"
assert os.path.isdir(patterned_ds_path), "Patterned optogenetic stimulation datastore directory not found!"
assert os.path.isdir(center_stim_ds_path), "Central optogenetic stimulation datastore directory not found!"

t_res = 50
s_res = 40
array_width = 4000

spont_ds = load_ds(os.path.abspath(spont_ds_path))
pattern_ds = load_ds(patterned_ds_path)
center_stim_ds = load_ds(center_stim_ds_path)

spont_ds_exc = load_ds(spont_ds_path,"V1_Exc_L2/3")
spont_ds_inh = load_ds(spont_ds_path,"V1_Inh_L2/3")
ff_ds = [ds for ds in pattern_ds if "fullfield" in ds.get_stimuli()[0]][0]
endo_ds = [ds for ds in pattern_ds if "endogenous" in ds.get_stimuli()[0]]
surr_ds = [ds for ds in pattern_ds if "surrogate" in ds.get_stimuli()[0]]
center_stim_ds_exc = load_ds(center_stim_ds_path,"V1_Exc_L2/3")
center_stim_ds_inh = load_ds(center_stim_ds_path,"V1_Inh_L2/3")

endo_stims = get_stim_patterns(endo_ds, array_width, s_res)
surr_stims = get_stim_patterns(surr_ds, array_width, s_res)

stim_ds = [ds for dss in [endo_ds, surr_ds, [ff_ds]] for ds in dss]
durations = np.unique([retrieve_ds_param_values(ds, "duration")[0] for ds in stim_ds])
stim_starts = np.unique([retrieve_ds_param_values(ds, "onset_time")[0] for ds in stim_ds])
stim_ends = np.unique([retrieve_ds_param_values(ds, "offset_time")[0] for ds in stim_ds])
assert len(durations) == 1, 'Multiple durations in dsvs: %s' % durations
assert len(stim_starts) == 1, 'Multiple stim_starts in dsvs: %s' % stim_starts
assert len(stim_ends) == 1, 'Multiple stim_ends in dsvs: %s' % stim_ends
duration, stim_start, stim_end = durations[0], stim_starts[0], stim_ends[0]

# Orientation map
or_map = gen_or_map(spont_ds_exc, "V1_Exc_L2/3", s_res, array_width)

# Calculate activity traces
A_spont_exc = gen_st_array(spont_ds_exc, s_res, t_res, array_width,50)
A_spont_inh = gen_st_array(spont_ds_inh, s_res, t_res, array_width,50)
A_ff = get_A_multitrial(ff_ds, s_res, t_res, array_width, 50)
A_endo = get_A_multitrial(endo_ds,s_res,t_res,array_width,50)
A_surr = get_A_multitrial(surr_ds,s_res,t_res,array_width,50)
A_dot_exc = get_A_multitrial(center_stim_ds_exc, s_res, t_res, array_width, 50)
A_dot_inh = get_A_multitrial(center_stim_ds_inh, s_res, t_res, array_width, 50)

A_spont_exc_calcium = get_calcium_signal(A_spont_exc,s_res,t_res)
A_spont_inh_calcium = get_calcium_signal(A_spont_inh,s_res,t_res)
A_ff_calcium = get_calcium_signal(A_ff,s_res,t_res)
A_endo_calcium = get_calcium_signal(A_endo,s_res,t_res)
A_surr_calcium = get_calcium_signal(A_surr,s_res,t_res)

A_spont_exc_bandpass = bandpass_filter(A_spont_exc_calcium, s_res)
A_spont_inh_bandpass = bandpass_filter(A_spont_inh_calcium, s_res)
A_ff_bandpass = bandpass_filter(A_ff_calcium, s_res)
A_ff_bandpass_last_stim_frame = np.dstack([A_ff_bandpass[i,:,:,stim_end//t_res+1] for i in range(A_ff_bandpass.shape[0])])
A_endo_bandpass = bandpass_filter(A_endo_calcium, s_res)
A_surr_bandpass = bandpass_filter(A_surr_calcium, s_res)

# Calculate correlation and similarity maps
coords = np.flip(np.dstack(np.meshgrid(range(array_width // s_res),range(array_width // s_res))).reshape([-1,2]),axis=-1)
Cmaps_ff = correlation_maps(A_ff_bandpass_last_stim_frame, coords)
Cmaps_spont_exc = correlation_maps(A_spont_exc, coords)
Cmaps_spont_inh = correlation_maps(A_spont_inh, coords)
Cmaps_spont_exc_bandpass = correlation_maps(A_spont_exc_bandpass, coords)
Cmaps_spont_inh_bandpass = correlation_maps(A_spont_inh_bandpass, coords)

sim_exc = correlation_or_map_similarity(Cmaps_spont_exc_bandpass, coords, or_map)
sim_inh = correlation_or_map_similarity(Cmaps_spont_inh_bandpass,coords,or_map)
sim_ff = correlation_or_map_similarity(Cmaps_ff,coords,or_map)
sim_exc_inh = correlation_map_similarity(Cmaps_spont_exc_bandpass,Cmaps_spont_inh_bandpass,coords,s_res)
sim_exc_ff = correlation_map_similarity(Cmaps_spont_exc_bandpass,Cmaps_ff,coords,s_res)

chance_sim = chance_similarity(or_map, s_res, t_res, coords)
km = kohonen_map(Cmaps_spont_exc,or_map)

# Experimental comparison analyses
# Smith 2018
event_idx_exc = extract_event_indices(A_spont_exc_calcium,t_res)
event_idx_inh = extract_event_indices(A_spont_inh_calcium,t_res)
small_spont_exc = resize_arr(A_spont_exc_bandpass[:,:,event_idx_exc], 50, 50)
small_spont_inh = resize_arr(A_spont_inh_bandpass[:,:,event_idx_inh], 50, 50)
exc_dim_full = dimensionality(small_spont_exc)
dc_exc = local_maxima_distance_correlations(Cmaps_spont_exc_bandpass, coords, s_res)
dc_inh = local_maxima_distance_correlations(Cmaps_spont_inh_bandpass, coords, s_res)
ssc = fit_spatial_scale_correlation(dc_exc[0,:],dc_exc[1,:])

# Mulholland 2021
exc_corr_wavelength = corr_wavelength(Cmaps_spont_exc_bandpass,coords,s_res,array_width)
inh_corr_wavelength = corr_wavelength(Cmaps_spont_inh_bandpass,coords,s_res,array_width)
lce_exc = local_correlation_eccentricity(Cmaps_spont_exc_bandpass,coords)
lce_inh = local_correlation_eccentricity(Cmaps_spont_inh_bandpass,coords)
exc_dim_sample = dim_random_sample_events(small_spont_exc)
inh_dim_sample = dim_random_sample_events(small_spont_inh)

# Mulholland 2024
model_wl_bins = np.linspace(0,2,21)
wls_spont = activity_wavelength(A_spont_exc_bandpass[:,:,event_idx_exc],s_res)
wls_spont_binned = wavelength_hist(wls_spont,bins=model_wl_bins)
wls_opt =  activity_wavelength(A_ff_bandpass[...,stim_end // t_res - 1].transpose(1,2,0),s_res)
wls_opt_binned = wavelength_hist(wls_opt,bins=model_wl_bins)

# Endogenous & surrogate stimuli

endo_corrs = calc_dsv_correlations(A_endo_bandpass,endo_stims,t_res,stim_start,stim_end)
surr_corrs = calc_dsv_correlations(A_surr_bandpass,surr_stims,t_res,stim_start,stim_end)

endo_corrs_raw = calc_dsv_correlations(A_endo,endo_stims,t_res,stim_start,stim_end)
surr_corrs_raw = calc_dsv_correlations(A_surr,surr_stims,t_res,stim_start,stim_end)

endo_insides, endo_outsides = get_insides_outsides(A_endo, endo_stims)
surr_insides, surr_outsides = get_insides_outsides(A_surr, surr_stims)
endo_insides_calcium, endo_outsides_calcium = get_insides_outsides(A_endo_calcium, endo_stims)
surr_insides_calcium, surr_outsides_calcium = get_insides_outsides(A_surr_calcium, surr_stims)

A_spont_exc_ = np.tile(A_spont_exc.mean(axis=-1),(1,A_dot_exc.shape[1],A_dot_exc.shape[4],1,1)).transpose(0,1,3,4,2)
A_spont_inh_ = np.tile(A_spont_inh.mean(axis=-1),(1,A_dot_exc.shape[1],A_dot_exc.shape[4],1,1)).transpose(0,1,3,4,2)
A_dot_exc_ = np.vstack([A_spont_exc_,A_dot_exc])
A_dot_inh_ = np.vstack([A_spont_inh_,A_dot_inh])

radii = np.arange(0,A_dot_exc_.shape[0]) * 50
r_margin = 100 # um
masks = [get_mask(A_dot_exc_[0,0,:,:,0],(radii[i]+r_margin)/s_res) for i in range(len(radii))]
masks_inv = [1-get_mask(A_dot_exc_[0,0,:,:,0],(radii[i]+r_margin)/s_res) for i in range(len(radii))]
masks_max = [masks[-1] for mask in masks]
masks_inv_max = [masks_inv[-1] for mask in masks_inv]

A_dot_exc_cmean = mask_mean(A_dot_exc_,masks)
A_dot_inh_cmean = mask_mean(A_dot_inh_,masks)
A_dot_exc_smean = mask_mean(A_dot_exc_,masks_inv)
A_dot_inh_smean = mask_mean(A_dot_inh_,masks_inv)

A_dot_exc_c_const_mean = mask_mean(A_dot_exc_,masks_max)
A_dot_inh_c_const_mean = mask_mean(A_dot_inh_,masks_max)
A_dot_exc_s_const_mean = mask_mean(A_dot_exc_,masks_inv_max)
A_dot_inh_s_const_mean = mask_mean(A_dot_inh_,masks_inv_max)

A_dot_exc_csum = mask_sum(A_dot_exc_,masks)
A_dot_inh_csum = mask_sum(A_dot_inh_,masks)
A_dot_exc_csum_total = (A_dot_exc_csum * t_res / 1000)[...,20:40].sum(axis=(1,2))
A_dot_inh_csum_total = (A_dot_inh_csum * t_res / 1000)[...,20:40].sum(axis=(1,2))

or_masks_close, or_masks_far = or_masks(or_map,masks_inv)
or_masks_close, or_masks_far = np.vstack([np.ones_like(or_map)[np.newaxis,...],or_masks_close]),np.vstack([np.ones_like(or_map)[np.newaxis,...],or_masks_far])

fr_close_exc = mask_mean(A_dot_exc_,or_masks_close)[...,20:40].mean(axis=(1,2))
fr_close_inh = mask_mean(A_dot_inh_,or_masks_close)[...,20:40].mean(axis=(1,2))
fr_far_exc = mask_mean(A_dot_exc_,or_masks_far)[...,20:40].mean(axis=(1,2))
fr_far_inh = mask_mean(A_dot_inh_,or_masks_far)[...,20:40].mean(axis=(1,2))

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

def get_or_hist(A, or_map, n_bins):
    ret = np.zeros((A.shape[2],n_bins))
    for k in range(A.shape[2]):
        ret[k,:], ors = np.histogram(or_map.flatten(),n_bins,weights=A[:,:,k].flatten())
        ret[k,:] = ret[k,:] / len(or_map.flatten()) * n_bins
    ors = ors[:-1]
    ors += (ors[1] - ors[0]) / 2
    return ret, ors

def get_or_hist_multidim(A,or_map,n_bins):
    return np.array([[get_or_hist(A[i,j,...], or_map, n_bins)[0] for j in range(A.shape[1]) ]for i in range(A.shape[0])])

def fold_in_half(array):
    return np.hstack([array[len(array)//2:],array[:len(array)//2]])

def rot_or(ors,rot=np.pi/2):
    return np.fmod(ors+rot,np.pi) - rot

def shuffle(A):
    A_sh = A.copy()
    np.random.shuffle(A_sh)
    return A_sh

def shuffle_entropies(A,patterns,n=100):
    entropies = np.zeros((n,A.shape[0]))
    for n in range(entropies.shape[0]):
        # Shuffle activity around spatially
        A_shuffled = np.array([[[shuffle(A[i,j,:,k]) for i in range(A.shape[0])] for j in range(A.shape[1])] for k in range(A.shape[-1])]).transpose(2,1,3,0)
        or_hist_A_shuffled = np.array([[get_or_hist(A_shuffled[i,j,:,:][np.newaxis,:,:], or_map[patterns[i] == 1], len(ors))[0] for j in range(A.shape[1])] for i in range(A.shape[0])]).mean(axis=(1,2))
        or_hist_A_entropy = scipy.stats.entropy(or_hist_A_shuffled, axis=-1)
        entropies[n,:] = or_hist_A_entropy
    return entropies

def get_p(shuffle, d):
    return np.array([scipy.stats.norm.sf(abs((d[i] - shuffle[:,i].mean()) / shuffle[:,i].std())) * 2 for i in range(len(d))])

ors = get_or_hist(np.ones_like(or_map)[:,:,np.newaxis], or_map,30)[1]

A_endo_stim = np.array([[[A_endo[i,j,:,:,k][endo_stims[i] == 1] for j in range(A_endo.shape[1])]for i in range(A_endo.shape[0])]for k in range(A_endo.shape[-1])])[20:40,...].transpose(1,2,3,0)
A_surr_stim = np.array([[[A_surr[i,j,:,:,k][surr_stims[i] == 1] for j in range(A_surr.shape[1])]for i in range(A_surr.shape[0])]for k in range(A_surr.shape[-1])])[20:40,...].transpose(1,2,3,0)

or_hist_A_endo = np.array([[get_or_hist(A_endo_stim[i,j,:,:][np.newaxis,:,:], or_map[endo_stims[i] == 1], len(ors))[0] for j in range(A_endo.shape[1])]for i in range(A_endo.shape[0])])
or_hist_A_surr = np.array([[get_or_hist(A_surr_stim[i,j,:,:][np.newaxis,:,:], or_map[surr_stims[i] == 1], len(ors))[0] for j in range(A_surr.shape[1])]for i in range(A_surr.shape[0])])
ors_bins = np.hstack([ors-ors[0],ors[-1]+ors[0]])
or_hist_endo = np.array([np.histogram(or_map[endo_stims[i] == 1],bins=ors_bins)[0] for i in range(len(endo_stims))])
or_hist_surr = np.array([np.histogram(or_map[surr_stims[i] == 1],bins=ors_bins)[0] for i in range(len(surr_stims))])

or_hist_endo_entropy = scipy.stats.entropy(or_hist_endo, axis=-1)
or_hist_surr_entropy = scipy.stats.entropy(or_hist_surr, axis=-1)
or_hist_A_endo_entropy = scipy.stats.entropy(or_hist_A_endo.mean(axis=(1,2)), axis=-1)
or_hist_A_surr_entropy = scipy.stats.entropy(or_hist_A_surr.mean(axis=(1,2)), axis=-1)

# Hypothesis: The entropy difference between the DAOD and the stimulation pattern is larger than if the same activation was taking place at random(ly shuffled) orientations
A_endo_entropy_shuffled = shuffle_entropies(A_endo_stim,endo_stims,1000)
A_surr_entropy_shuffled = shuffle_entropies(A_surr_stim,surr_stims,1000)

A_endo_p = get_p(np.abs(A_endo_entropy_shuffled-or_hist_endo_entropy), np.abs(or_hist_A_endo_entropy-or_hist_endo_entropy))
A_surr_p = get_p(np.abs(A_surr_entropy_shuffled-or_hist_surr_entropy), np.abs(or_hist_A_surr_entropy-or_hist_surr_entropy))

spont_or_hist, ors = get_or_hist(A_spont_exc, or_map, 30)
spont_or_hist_entropy = scipy.stats.entropy(spont_or_hist, axis=-1)

#########################################################
# Figure 3
#########################################################

#######################
# Panel A
#######################
from matplotlib import cm
import matplotlib.patheffects as PathEffects
sz = 0.7
dpi = 900
fig = plt.figure(figsize=(2.8*sz,2.5*sz))
import colorsys
def scale_lightness(rgb, scale_l):
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # manipulate h, l, s values and return as rgb
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s = s)

# 1. Timecourse plot
ax1 = fig.add_subplot()
tmin = 5
tmax = 35
sptc = normalize(A_spont_exc_calcium.mean(axis=(0,1))[tmin  * 1000 // t_res : tmax * 1000 // t_res] / t_res)
std = normalize(A_spont_exc_calcium.std(axis=(0,1)))[tmin  * 1000 // t_res : tmax * 1000 // t_res]
tt = np.linspace(0,len(sptc)/1000*t_res,len(sptc))
ax1.plot(tt,sptc,c='k',lw=1)
ax1.set_xlabel("Time (s)",fontsize=8)
ax1.set_ylabel("Normalized\nspont. activity",fontsize=8,labelpad=-12)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax1.spines['bottom'].set_linewidth(1.5)
ax1.spines['left'].set_linewidth(1.5)
    
select_indices = np.array([98, 246, 451])

index_colors = [None] * len(select_indices)
for i in range(len(select_indices)):
    h, b = np.histogram(or_map.flatten(), 50, weights=normalize(A_spont_exc_calcium[:,:,select_indices[i] + tmin * 1000 // t_res]).flatten())
    b /= or_map.max()
    c = cm.hsv((b[np.argmax(h)] + b[np.argmax(h)+1])/2)
    c = scale_lightness(c[:3], 0.9)
    index_colors[i] = c
    surr = 3
    ax1.plot(tt[select_indices[i]-surr*3:select_indices[i]+3*surr],
             sptc[select_indices[i]-surr*3:select_indices[i]+3*surr],c=c,lw=1)

ax1.set_yticks([0, 1])
ax1.set_xticks([0,10,20,30])
ax1.set_xlim(0,30)
ax1.set_ylim(0,1)
ax1.tick_params(axis='both', labelsize=8, size=6, width=1.5)

fig.savefig(os.path.join(spont_ds_path,"Figure3A0.svg"),dpi=dpi, bbox_inches = "tight",transparent=True)

select_indices = [1638,1023,117]
for i in range(len(select_indices)):
    idx = select_indices[i]
    sz = 0.9
    fig = plt.figure(figsize=(sz,sz))
    A_img = A_spont_exc_calcium[:,:,idx+tmin * 1000 // t_res].copy()
    if 1:
        lw = 4
        A_new = np.zeros((A_spont_exc_calcium.shape[0]+lw,A_spont_exc_calcium.shape[1]+lw))
        A_new[lw//2:-lw//2,lw//2:-lw//2] = A_img
        A_img = A_new
    plt.imshow(A_img,cmap='gray')
    plt.axis('off')
    # Coloured frames
    if 1:
        ymax, xmax, _ = A_spont_exc_calcium.shape
        frame_x = [0,xmax,xmax,0,0]
        frame_y = [0,0,ymax,ymax,0]
        plt.plot(frame_x, frame_y, c=index_colors[i],lw=lw)
        plt.xlim(0,xmax)
        plt.ylim(ymax,0)
    fig.savefig(os.path.join(spont_ds_path,"Figure3A%d.svg" % (i+1)),dpi=dpi, bbox_inches = "tight",transparent=True)
    
#######################
# Panel B
#######################
Cmap_idxs = [2708,8186]
Cmap_x, Cmap_y = np.stack([coords[p][0] for p in Cmap_idxs]), np.stack([coords[p][1] for p in Cmap_idxs])

fig = plt.figure(figsize=(1.2,1.0))   
ax3 = fig.add_subplot()
im = plt.imshow(or_map,cmap='hsv', interpolation='none')
cbar = plt.colorbar(im,aspect=13,ax=ax3,fraction=0.069,location='left')
ax3.tick_params(labelsize=13)
cbar.set_label(label='Orientation', labelpad=0,fontsize=9)
cbar.set_ticks([0, np.pi],labels=["0","$\pi$"],fontsize=9)
ax3.get_xaxis().set_visible(False)
ax3.get_yaxis().set_visible(False)
ms = 9
mew = 1*sz

if 1:
    ax3.plot(Cmap_y,Cmap_x,'o',color='k',markersize=ms,mec='black',mew=mew)
    ax3.text(Cmap_y[0]-0.2, Cmap_x[0]+0.5, "1", color='white', weight='bold', ha='center', va='center',fontsize=8)
    ax3.text(Cmap_y[1]-0.2, Cmap_x[1]+0.5, "2", color='white', weight='bold', ha='center', va='center',fontsize=8)

fig.savefig(os.path.join(spont_ds_path,"Figure3B.svg"),dpi=dpi, bbox_inches = "tight",transparent=True)

#######################
# Panel C
#######################

fig = plt.figure(figsize=(2.45,1.0))
gs00 = gs.GridSpec(1, 10, figure=fig)
ax2 = [fig.add_subplot(gs00[:,0:5]), fig.add_subplot(gs00[:,5:10])]
fig.suptitle("Exc",fontsize=10,y=1.05,x=0.53)
for i in range(2):
    ax_loop = ax2[i]
    ax_loop.get_xaxis().set_visible(False)
    ax_loop.get_yaxis().set_visible(False)
    
    ax_loop.text(Cmap_y[i]-0.2, Cmap_x[i]+1, i+1, color='white', weight='bold', ha='center', va='center',fontsize=8)
    ax_loop.plot(Cmap_y[i],Cmap_x[i],'o',color='k',markersize=ms,mec='k',mew=mew)

    im = ax_loop.imshow(Cmaps_spont_exc_bandpass[Cmap_idxs[i]],cmap='bwr',vmin=-1,vmax=1)
    if i == 0:
        cbar = plt.colorbar(im,aspect=13,fraction=0.069,location='left',ax=ax_loop)
        cbar.set_label(label='Correlation', labelpad=-1,fontsize=9)
        cbar.set_ticks([-1, 1],labels=["-1","1"],fontsize=9)
    
fig.savefig(os.path.join(spont_ds_path,"Figure3C.svg"),dpi=dpi, bbox_inches = "tight",transparent=True)

#######################
# Panel D
#######################

fig = plt.figure(figsize=(1.2,1.0))   
ax5 = fig.add_subplot()
fig.suptitle("Inh",fontsize=10,y=1.05,x=0.58)
ax5.get_xaxis().set_visible(False) 
ax5.get_yaxis().set_visible(False) 
ax5.plot(Cmap_y[1],Cmap_x[1],'o',color='k',markersize=ms,mec='k',mew=mew)
ax5.text(Cmap_y[1]+0, Cmap_x[1]+1, "2", color='white', weight='bold', ha='center', va='center',fontsize=8)
im = ax5.imshow(Cmaps_spont_exc_bandpass[Cmap_idxs[1]],cmap="bwr",vmax=1,vmin=-1) 
cbar = plt.colorbar(im,aspect=13,ax=ax5,fraction=0.069,location='left') 
cbar.set_label(label='Correlation', labelpad=-7,fontsize=9)
cbar.set_ticks([-1,1])
[l.set_fontsize(9) for l in cbar.ax.yaxis.get_ticklabels()]

fig.savefig(os.path.join(spont_ds_path,"Figure3D.svg"),dpi=dpi, bbox_inches = "tight",transparent=True)

#######################
# Panel E
#######################

import matplotlib.colors as mcolors
def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=-1):
    if n == -1: 
        n = cmap.N 
    new_cmap = mcolors.LinearSegmentedColormap.from_list( 
         'trunc({name},{a:.2f},{b:.2f})'.format(name=cmap.name, a=minval, b=maxval), 
         cmap(np.linspace(minval, maxval, n))) 
    return new_cmap 

fig = plt.figure(figsize=(2.5,1.0))

fig.suptitle("Exc - or.map",fontsize=10,y=1.1,x=0.42)
fig.subplots_adjust(wspace=0.05)
inferno_t = truncate_colormap(plt.get_cmap("hot"), 0.5, 1) 
plt.gca().get_xaxis().set_visible(False) 
plt.gca().get_yaxis().set_visible(False) 
im = plt.imshow(sim_exc,cmap=inferno_t,vmax=1,vmin=0) 

cbar = plt.colorbar(im,aspect=13,ax=plt.gca(),fraction=0.069,location='left') 
cbar.set_label(label='Similarity', labelpad=0,fontsize=9)
cbar.set_ticks([0,1])
[l.set_fontsize(9) for l in cbar.ax.yaxis.get_ticklabels()]

fig.savefig(os.path.join(spont_ds_path,"Figure3E.svg"),dpi=dpi, bbox_inches = "tight",transparent=True)

#######################
# Panel F
#######################

fig = plt.figure(figsize=(1.2,1.0))
fig.suptitle("Exc - Inh",fontsize=10,y=1.05,x=0.58)
ax5 = fig.add_subplot()
ax5.get_xaxis().set_visible(False) 
ax5.get_yaxis().set_visible(False)
im = ax5.imshow(sim_exc_inh,cmap="hot",vmax=1,vmin=-1) 
cbar = plt.colorbar(im,aspect=13,ax=ax5,fraction=0.069,location='left') 
cbar.set_label(label='Similarity', labelpad=-5,fontsize=9)
cbar.set_ticks([-1,1])
[l.set_fontsize(9) for l in cbar.ax.yaxis.get_ticklabels()]
fig.savefig(os.path.join(spont_ds_path,"Figure3F.svg"),dpi=dpi, bbox_inches = "tight",transparent=True)

#######################
# Panel G
#######################

fig = plt.figure(figsize=(1.2,1.0)) 
ax3 = fig.add_subplot()
im = plt.imshow(km,cmap='hsv', interpolation='none')
cbar = plt.colorbar(im,aspect=13,ax=ax3,fraction=0.069,location='left')
ax3.tick_params(labelsize=13)
cbar.set_label(label='Estimated\norientation', labelpad=0,fontsize=9)
cbar.set_ticks([0, np.pi],labels=["0","$\pi$"],fontsize=9)
#cbar.set_ticks([0, 2, 4,6],labels=["0","2","4","0"],fontsize=9)
ax3.get_xaxis().set_visible(False)
ax3.get_yaxis().set_visible(False)
fig.savefig(os.path.join(spont_ds_path,"Figure3G.svg"),dpi=dpi, bbox_inches = "tight",transparent=True)

#######################
# Panel H
#######################

def prediction_interval(data,confidence=0.95):
    n = len(data)
    t_critical = scipy.stats.t.ppf((1 + confidence) / 2, df=n-1)
    return t_critical * data.std(ddof=1) * np.sqrt(1+1/len(data))
    
def double_metric_plot(ax,exp_0,exp_1,model_0,model_1,ylim,ylabel,has_legend=False,x_ticks=["",""],ylabel_pad=0):
    exp_0 = np.array(exp_0)
    exp_1 = np.array(exp_1)
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_linewidth(1.7)

    h0 = ax.errorbar(
        [1],
        exp_0.mean(),
        yerr=prediction_interval(exp_0),
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
        yerr=prediction_interval(exp_1),
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
                   bbox_to_anchor=(5.2, -0.15),
                   handlelength = 0.8,
                   frameon=False,ncols=4,fontsize=fs)
    ax.set_ylabel(ylabel,labelpad=ylabel_pad,fontsize=fs)
    ax.tick_params(axis='both', labelsize=fs, size=6, width=1.7)
    ax.set_xticks([1,2],x_ticks,fontsize=fs)
    ax.set_yticks(ylim,[str(ylim[0]),str(ylim[1])],fontsize=fs)
    
# similarity, ie similarity, correlation at maxima > 2mm, dimensionality, local correlation eccentricity  
fig, ax = plt.subplots(1,5,figsize=(8.3,1.8))
fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.75)
d = experimental_data["Mulholland 2021"]
double_metric_plot(ax[0],experimental_data["Smith 2018"]["similarity"],d['exc inh similarity'],sim_exc.mean(),sim_exc_inh.mean(),[0,1],"Similarity",x_ticks=["E$\\rightarrow$Or.","I$\\rightarrow$E"],ylabel_pad=-7)
double_metric_plot(ax[1],d['dimensionality']['exc'],d['dimensionality']['inh'],exc_dim_sample.mean(),inh_dim_sample.mean(),[0,20],"Dimensionality",has_legend=True,x_ticks=["E","I"],ylabel_pad=-12)
double_metric_plot(ax[2],d['corr above 2 mm']['exc'],d['corr above 2 mm']['inh'],dc_exc[1,dc_exc[0,:] > 2].mean(),dc_inh[1,dc_inh[0,:] > 2].mean(),[0,0.5],"Correlation at\nmaxima (>2 mm)",x_ticks=["E","I"],ylabel_pad=-16)
double_metric_plot(ax[3],d['mean eccentricity']['exc'],d['mean eccentricity']['inh'],lce_exc.mean(),lce_inh.mean(),[0,1],"Local correlation\neccentricity",x_ticks=["E","I"],ylabel_pad=-12)
double_metric_plot(ax[4],d['corr wavelength']['exc'],d['corr wavelength']['inh'],exc_corr_wavelength,inh_corr_wavelength,[0,1.5],"Correlation\nwavelength (mm)",x_ticks=["E","I"],ylabel_pad=-16)

fig.savefig(os.path.join(spont_ds_path,"Figure3H.svg"),dpi=dpi, bbox_inches = "tight",transparent=True)

#########################################################
# Figure 4
#########################################################

#######################
# Panel B
#######################

stim_start, stim_end, ff_duration = 1000,2000, 4000
ymin, ymax = 0,1.1
mulholland_x = 0.1 + np.array([-1.019,-0.893,-0.755,-0.602,-0.472,-0.335,-0.197,-0.075,0.086,0.193,0.354,0.522,0.66,0.836,0.889,0.95,1.019,1.103,1.18,1.241,1.302,1.356,1.424,1.486,1.547,1.616,1.73,1.792,1.845,1.906,1.952,2.006,2.075,2.159,2.235,2.319,2.396,2.457,2.518,2.587,2.671,2.778,2.885,2.992,3.107,3.222,3.382,3.551,3.681,3.826,3.91,4.029])
mulholland_y = np.array([0.046,0.035,0.032,0.028,0.028,0.028,0.025,0.028,0.028,0.028,0.028,0.028,0.032,0.035,0.06,0.116,0.212,0.335,0.451,0.575,0.702,0.804,0.91,1.002,1.101,1.192,1.28,1.326,1.333,1.33,1.309,1.266,1.199,1.108,1.03,0.938,0.854,0.797,0.737,0.684,0.617,0.547,0.48,0.423,0.367,0.328,0.279,0.236,0.212,0.187,0.173,0.155])

rect = matplotlib.patches.Rectangle(
    (stim_start / 1000, 0),
    (stim_end - stim_start) / 1000,
    2,
    linewidth=1,
    facecolor=(255 / 255, 223 / 255, 0),
    alpha=0.6,
)

fig = plt.figure(figsize=(1.7,2.5))
plt.gca().add_patch(rect)

t_ff_resp = np.linspace(0, ff_duration / 1000, ff_duration // t_res)
ff_timecourses = A_ff_calcium.mean(axis=(1,2))
mean, sem = ff_timecourses.mean(axis=0), ff_timecourses.std(
    axis=0
) / np.sqrt(len(ff_timecourses))
h1, = plt.plot(mulholland_x, mulholland_y,c='k')
h3, = plt.plot(t_ff_resp, mean, "-", c="r")

if 0:
    h2 = plt.gca().fill_between(
        t_ff_resp,
        mean - sem,
        mean + sem,
        color="k",
        alpha=0.4,
        #label="_nolegend_",
    )

plt.gca().spines[['right','top']].set_visible(False)
plt.gca().spines[['bottom','left']].set_linewidth(1.5)

fs = 9
plt.xlabel("Time (s)",fontsize=fs)
plt.ylabel("Population mean $\\Delta$F/F",fontsize=fs)
plt.xticks([0,1,2,3,4],labels=["0","1","2","3","4"],fontsize=fs)
plt.yticks([0,1.5],fontsize=fs)

leg = plt.legend([h1,h3,rect],['Experiment','Mean\nresp.','Light\nON'],fontsize=8,frameon=True,handlelength=1.5,loc='upper left')

plt.legend(
    [rect,h1,h3],
    ['Light ON','Experiment','Model'],
    loc="upper center",
    handlelength = 1.0,
    bbox_to_anchor=(0.36, 1.35),
    frameon=False,
    fontsize=9,
)
plt.gca().yaxis.set_label_coords(-0.09,0.45)
plt.ylim(0,1.7)
plt.xlim(0,4)
plt.tick_params(labelsize=9, size=4, width=1.5)
#plt.tick_params(axis='y', labelsize=10, size=3, width=1.5)
if 0:
    scalebar = AnchoredSizeBar(plt.gca().transData,
                   1, '1 s', 'center',
                   bbox_to_anchor = (77,74),
                   #bbox_to_anchor = (41,51),
                   fontproperties=fm.FontProperties(size=10),
                   pad=0,
                   color='k',
                   frameon=False,
                   size_vertical=0.016)
    plt.gca().add_artist(scalebar)
    
fig.savefig(os.path.join(patterned_ds_path,"fig4_B.svg"),dpi=dpi, bbox_inches = "tight",transparent=True)

#######################
# Panel C
#######################

fig = plt.figure(figsize=(1.2*1.15,2.1*1.15))
gs00 = gs.GridSpec(20, 41, figure=fig)
ax2 = [fig.add_subplot(gs00[0:8,:]), fig.add_subplot(gs00[9:17,1:])]
opt_Cmaps = [Cmaps_spont_exc_bandpass[Cmap_idxs[1]],Cmaps_ff[Cmap_idxs[1]]]

for i in range(2):
    ax_loop = ax2[i]
    ax_loop.get_xaxis().set_visible(False)
    ax_loop.get_yaxis().set_visible(False)
    
    ax_loop.plot(Cmap_y[1],Cmap_x[1],'o',color='k',markersize=ms,mec='k',mew=mew)
    ax_loop.text(Cmap_y[1]+0, Cmap_x[1]+0.5, "2", color='white', weight='bold', ha='center', va='center',fontsize=8)
    im = ax_loop.imshow(opt_Cmaps[i],cmap='bwr',vmin=-1,vmax=1)
    if i == 0:
        cbar = plt.colorbar(im,aspect=13,fraction=0.069,location='left',ax=ax_loop)
        #.tick_params(labelsize=13)
        cbar.set_label(label='Correlation', labelpad=0,fontsize=9)
        cbar.set_ticks([-1, 1],labels=["-1","1"],fontsize=9)

fig.savefig(os.path.join(patterned_ds_path,"fig4_C.svg"),dpi=dpi, bbox_inches = "tight",transparent=True)

#######################
# Panel D
#######################

fig = plt.figure(figsize=(1.2*0.92,1.0*0.92))
ax0 = fig.add_subplot()
ax0.get_xaxis().set_visible(False)
ax0.get_yaxis().set_visible(False)
im = ax0.imshow(sim_exc_ff,cmap='hot',vmin=-1,vmax=1)

cbar = plt.colorbar(im,aspect=13,ax=ax0,fraction=0.069,location='left')
ax0.tick_params(labelsize=13*sz)
cbar.set_label(label='Similarity', labelpad=0,fontsize=9)
cbar.set_ticks([-1,1],labels=["-1","1"],fontsize=9)
fig.savefig(os.path.join(patterned_ds_path,"fig4_D.svg"),dpi=dpi, bbox_inches = "tight",transparent=True)

#######################
# Panel E
#######################

fig = plt.figure(figsize=(1.1,2.0))
ax = fig.add_subplot()
exp = np.array([0.50349,0.56358,0.6047,0.55066,0.45385,0.37765])
ax.spines[['top','right']].set_visible(False)
ax.spines[['left','bottom']].set_linewidth(1.5)

h0 = ax.errorbar(
    [1],
    exp.mean(),
    yerr=prediction_interval(exp),
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

h1, = ax.plot(np.ones_like(exp)*1,exp,'o',color='silver',alpha=0.5)
h2, = ax.plot([1],exp.mean(),'o',color='k')
h3, = ax.plot([1],sim_exc_ff.mean(),'ro')

ax.set_ylim(0,1)
ax.set_yticks([0,1],labels=[0,1],fontsize=9)
ax.set_xlim(0.5,1.5)
ax.tick_params(labelsize=9, size=4, width=1.5)

ax.legend([h1,h2,h0,h3],
            ["Exp. trial","Exp. mean","95% PI","Model"],
           bbox_to_anchor=(1.2, 1.6),
           handlelength = 0.8,
           frameon=False,fontsize=9)

ax.set_ylabel("Spontaneous - opto\nsimilarity",labelpad=-10,fontsize=9)
ax.set_xticks([1],["Opto"],fontsize=9)
fig.savefig(os.path.join(patterned_ds_path,"fig4_E.svg"),dpi=dpi, bbox_inches = "tight",transparent=True)

#######################
# Panel G
#######################

mean_endo_resp = A_endo_calcium[0,1,:,:,stim_start//t_res:(stim_end)//t_res].mean(axis=(-1))
mean_surr_resp = A_surr_calcium[0,1,:,:,stim_start//t_res:(stim_end)//t_res].mean(axis=(-1))
cmaps = ['hsv','gray']
maps = [[or_map,or_map],[normalize(mean_endo_resp),normalize(mean_surr_resp)]]
patterns = [endo_stims[0,:,:], endo_stims[0,:,:]]
patterns = [resize(p, or_map.shape,anti_aliasing=False) for p in patterns]

for k in range(2):
    fig = plt.figure(figsize=(2.4,0.98))
    gs00 = gs.GridSpec(1, 10, figure=fig)
    ax2 = [fig.add_subplot(gs00[:,0:5]), fig.add_subplot(gs00[:,5:10])]

    for i in range(2):
        if i == 1:
            c0,c1 = 'silver', 'white'
        else:
            c0,c1 = 'white','silver'
        ax_loop = ax2[i]
        ax_loop.get_xaxis().set_visible(False)
        ax_loop.get_yaxis().set_visible(False)

        if k == 0:
            alpha_min, alpha_max = 0.2, 1.0
        else:
            alpha_min, alpha_max = 0.6, 1.0
        alpha_arr = alpha_min + (alpha_max - alpha_min) * patterns[i].astype(float)

        if k == 0:
            ax_loop.set_title("Endogenous" if i == 0 else "Surrogate",fontsize=9)
            ax_loop.imshow(np.zeros_like(maps[k][i]),cmap='gray')
            im = ax_loop.imshow(maps[k][i],cmap=cmaps[k],alpha=alpha_arr,vmin=0, interpolation=None)
        else:
            im = ax_loop.imshow(maps[k][i],cmap='gray',vmin=0, interpolation=None)
            rgba_image = np.zeros((100, 100, 4), dtype=np.uint8)
            rgba_image[:, :, :3] = (255, 223, 0)
            rgba_image[:, :, 3] = (patterns[i] * 0.6 * 255).astype(np.uint8)  # Set alpha channel
            im_pattern = ax_loop.imshow(rgba_image)
        if i == 0:
            cbar = plt.colorbar(im,aspect=13,fraction=0.069,ax=ax_loop,location='left')
            if k == 0:
                cbar.set_label(label='Orientation', labelpad=0,fontsize=9)
                cbar.set_ticks([0, np.pi],labels=["0","$\pi$"],fontsize=9)
            else:
                cbar.set_label(label='Normalized\nresponse', labelpad=-7,fontsize=9)
                cbar.set_ticks([0, 1],labels=["0","1"],fontsize=9)
                
                rect = matplotlib.patches.Rectangle(
                    (1, 0),
                    1,
                    1.3,
                    linewidth=1,
                    facecolor=(255 / 255, 223 / 255, 0),
                    alpha=0.6,
                )
                ax_loop.legend([rect],["Stim. ROI"],handlelength=0.8,frameon=False,bbox_to_anchor=(1.6, -0.0),fontsize=9)
    fig.savefig(os.path.join(patterned_ds_path,"fig4_G.svg"),dpi=dpi, bbox_inches = "tight",transparent=True)
    
#######################
# Panel H
#######################

io_time = [0.0014,0.0055,0.0058,0.0104,0.0125,0.0563,0.0581,0.0607,0.0664,0.0671,0.0986,0.106,0.1073,0.1262,0.1304,0.1457,0.1564,0.185,0.191,0.1982,0.1994,0.2429,0.2478,0.2507,0.2551,0.29,0.2909,0.3042,0.3192,0.3284,0.3446,0.3615,0.3754,0.3928,0.3984,0.4273,0.4282,0.457,0.4596,0.4838,0.4964,0.5184,0.5415,0.55,0.5659,0.5845,0.6107,0.6247,0.6445,0.6505,0.6768,0.7093,0.7155,0.7403,0.7474,0.7605,0.7932,0.8119,0.8137,0.8251,0.8621,0.8706,0.8738,0.8886,0.9286,0.94,0.9433,0.9493,0.975,0.9828,0.9893,1.0044,1.015,1.0161,1.0258,1.0381,1.0456,1.049,1.0705,1.0713,1.0816,1.0827,1.094,1.1091,1.1171,1.1222,1.1279,1.1419,1.1496,1.1541,1.1694,1.1756,1.1786,1.1904,1.1983,1.2172,1.219,1.2274,1.2328,1.2526,1.2603,1.2673,1.2675,1.2906,1.2962,1.3017,1.3038,1.3175,1.3414,1.3449,1.3485,1.3623,1.3834,1.3871,1.3907,1.4147,1.4195,1.4255,1.4365,1.4451,1.4541,1.4733,1.4794,1.4799,1.4829,1.5085,1.5168,1.5299,1.543,1.5522,1.56,1.5664,1.5715,1.5933,1.5937,1.5987,1.6178,1.6242,1.6302,1.6355,1.6574,1.6641,1.68,1.6972,1.6993,1.716,1.7229,1.7309,1.7569,1.7631,1.7721,1.7902,1.8027,1.8058,1.806,1.8486,1.8488,1.8489,1.8519,1.8905,1.896,1.8973,1.907,1.9126,1.9333,1.937,1.944,1.9522,1.9587,1.9803,1.9829,1.9839,1.9896,2.0056,2.0087,2.017,2.0233,2.0295,2.0416,2.0463,2.0465,2.0628,2.077,2.0824,2.0826,2.1044,2.1083,2.1145,2.1231,2.134,2.1465,2.1532,2.1583,2.1624,2.1834,2.1967,2.1981,2.2022,2.2103,2.2197,2.2323,2.2353,2.2447,2.2653,2.2725,2.2729,2.2767,2.2951,2.3098,2.3109,2.3227,2.324,2.3564,2.359,2.3613,2.3831,2.3901,2.3987,2.4086,2.4304,2.4346,2.4398,2.4513,2.4714,2.472,2.4949,2.5014,2.5085,2.5366,2.5455,2.5627,2.5732,2.5817,2.5825,2.6131,2.6217,2.624,2.6271,2.6668,2.6745,2.685,2.7032,2.7074,2.7324,2.7339,2.7567,2.7742,2.7861,2.7997,2.8067,2.837,2.8567,2.8588,2.8737,2.9061,2.9138,2.9243,2.9593,2.9637,2.9892,3.0076,3.022,3.0743,3.0817,3.0849,3.0892,3.1456,3.1511,3.1995,3.207,3.2253,3.2434,3.3063,3.318,3.3351,3.3833,3.4171,3.4263,3.4283,3.4718,3.5376,3.5386,3.5393,3.5475,3.6292,3.6309,3.6434,3.6483,3.7215,3.7313,3.7417,3.7499,3.8072,3.817,3.8239,3.8283,3.891,3.9037,3.9054,3.9125,3.9196,3.9564,3.9784,3.9821,3.9944,4.0007,4.0328,4.0364,4.038,4.0516]
i_exp = [0.05014,0.04995,0.04994,0.04974,0.04965,0.04964,0.04964,0.04964,0.04964,0.04973,0.04994,0.04999,0.04999,0.04982,0.04978,0.04964,0.04954,0.04965,0.04968,0.04971,0.04971,0.04922,0.04916,0.04899,0.04871,0.04655,0.04652,0.04615,0.04573,0.04547,0.04501,0.04433,0.04377,0.04307,0.04284,0.04186,0.04183,0.04085,0.04084,0.04069,0.04062,0.04049,0.04017,0.04005,0.03983,0.03958,0.03934,0.03921,0.03903,0.03903,0.03903,0.03902,0.03902,0.03907,0.03908,0.0391,0.03916,0.03918,0.03919,0.03934,0.03983,0.03995,0.03999,0.04033,0.04125,0.04151,0.04179,0.04229,0.04445,0.0451,0.04565,0.04869,0.05082,0.05103,0.05299,0.05545,0.05807,0.05925,0.06671,0.06698,0.07057,0.07093,0.07477,0.07986,0.08256,0.0843,0.08623,0.09109,0.0938,0.09537,0.1007,0.10287,0.10403,0.10871,0.11183,0.11925,0.12009,0.12391,0.12638,0.13545,0.13856,0.14136,0.14143,0.15073,0.153,0.15572,0.15677,0.16361,0.17556,0.1773,0.17904,0.18571,0.1959,0.19769,0.19944,0.21183,0.21434,0.21744,0.22337,0.22803,0.23291,0.24498,0.24881,0.24911,0.25076,0.26465,0.26918,0.27521,0.2812,0.28547,0.28956,0.29296,0.29567,0.30716,0.30732,0.30929,0.31678,0.31928,0.32161,0.32372,0.33397,0.33715,0.3446,0.35081,0.35157,0.35761,0.36008,0.36349,0.37456,0.37718,0.3806,0.38742,0.39214,0.39332,0.39337,0.41045,0.4105,0.41056,0.41177,0.42748,0.4293,0.42972,0.43292,0.43477,0.44044,0.44133,0.44298,0.44492,0.44646,0.44936,0.44971,0.44984,0.4506,0.45233,0.45267,0.45357,0.45414,0.4547,0.45582,0.45559,0.45558,0.45479,0.4541,0.45349,0.45346,0.45095,0.4505,0.4492,0.44743,0.44518,0.44253,0.4411,0.44003,0.43915,0.43311,0.42847,0.42798,0.42654,0.42385,0.42068,0.41673,0.4158,0.41285,0.40445,0.40149,0.40132,0.39972,0.39187,0.38666,0.38624,0.38207,0.3816,0.36976,0.36881,0.36796,0.35993,0.35731,0.35414,0.35052,0.34253,0.341,0.3391,0.33505,0.32793,0.32773,0.31989,0.31767,0.31523,0.30573,0.30272,0.29644,0.29263,0.28953,0.28921,0.28061,0.2782,0.27757,0.27669,0.26309,0.26104,0.2582,0.25332,0.25219,0.24556,0.24517,0.23914,0.23506,0.23231,0.22914,0.22752,0.21965,0.21455,0.2141,0.2109,0.20392,0.20261,0.20082,0.19485,0.1941,0.18973,0.18658,0.1841,0.17589,0.17473,0.17433,0.17378,0.16662,0.16598,0.16045,0.1596,0.15751,0.15544,0.14932,0.14819,0.14652,0.14242,0.13954,0.13877,0.13865,0.13613,0.13233,0.13227,0.13223,0.13179,0.12735,0.12726,0.12658,0.12631,0.12225,0.12171,0.12138,0.12111,0.11927,0.11895,0.11873,0.11859,0.11653,0.11611,0.11608,0.11596,0.11583,0.11519,0.11481,0.11475,0.11457,0.11448,0.11402,0.11397,0.11394,0.11375]
o_exp = [0.03569,0.03569,0.03569,0.0357,0.0357,0.03578,0.03578,0.03579,0.03582,0.03582,0.03595,0.03588,0.03587,0.03568,0.03564,0.03549,0.03535,0.03495,0.03487,0.03477,0.03474,0.03377,0.03365,0.03359,0.03345,0.03235,0.03232,0.03226,0.03219,0.03214,0.0318,0.03144,0.03114,0.03102,0.03098,0.03078,0.03076,0.03016,0.0301,0.0296,0.02934,0.02889,0.02841,0.02824,0.02812,0.02798,0.02778,0.02784,0.02792,0.02794,0.02804,0.0279,0.02787,0.02776,0.02774,0.02769,0.02756,0.02749,0.0275,0.02761,0.02794,0.02802,0.02803,0.0281,0.02828,0.02837,0.02839,0.02844,0.02863,0.02932,0.02988,0.03121,0.03214,0.03223,0.03394,0.03608,0.03739,0.03799,0.04201,0.04215,0.04408,0.04428,0.04675,0.05002,0.05175,0.05302,0.05443,0.05787,0.05972,0.06079,0.06444,0.06604,0.0668,0.06986,0.0719,0.07629,0.07671,0.07865,0.07991,0.0849,0.08684,0.08839,0.08843,0.09359,0.09519,0.09674,0.09734,0.10123,0.10783,0.10879,0.10978,0.11393,0.12027,0.12139,0.12242,0.12927,0.13066,0.13237,0.13549,0.13795,0.14044,0.14577,0.14749,0.14763,0.14848,0.15567,0.15798,0.1616,0.16521,0.16788,0.17011,0.17196,0.17307,0.1778,0.17788,0.17897,0.18396,0.18563,0.18698,0.18821,0.19318,0.19472,0.19882,0.20327,0.2037,0.20714,0.20854,0.21019,0.21564,0.21693,0.21883,0.22245,0.22495,0.22558,0.22561,0.23419,0.23421,0.23425,0.23484,0.24295,0.24412,0.24438,0.24565,0.24639,0.2491,0.24959,0.25033,0.2512,0.25189,0.25417,0.25445,0.25453,0.255,0.25632,0.25657,0.25726,0.25778,0.25811,0.25877,0.25902,0.25903,0.25822,0.25751,0.25724,0.25723,0.25447,0.25397,0.25317,0.25206,0.25063,0.24901,0.24813,0.24743,0.24687,0.244,0.24218,0.24192,0.24114,0.23964,0.23788,0.23553,0.23497,0.2329,0.22837,0.22678,0.22669,0.22593,0.22219,0.21921,0.21894,0.21631,0.21602,0.20876,0.20827,0.20783,0.2037,0.20236,0.20073,0.19885,0.19457,0.19375,0.19273,0.19047,0.18634,0.18622,0.18153,0.18021,0.17887,0.17358,0.1719,0.16866,0.16669,0.16513,0.16496,0.15931,0.15798,0.15764,0.15716,0.15103,0.14985,0.14823,0.14538,0.14473,0.14079,0.14057,0.13745,0.13505,0.13343,0.13156,0.13069,0.12694,0.12451,0.12426,0.12258,0.11891,0.11804,0.11689,0.11303,0.11255,0.10975,0.10827,0.10711,0.10292,0.10232,0.10207,0.10172,0.09721,0.09676,0.09376,0.0933,0.09216,0.09124,0.08804,0.08745,0.08658,0.08413,0.08282,0.08246,0.08239,0.0807,0.07834,0.07831,0.07828,0.07799,0.07555,0.0755,0.07519,0.07507,0.07324,0.073,0.07274,0.07267,0.07219,0.07212,0.07207,0.07203,0.07157,0.07147,0.07146,0.07141,0.07137,0.07114,0.071,0.07098,0.07092,0.07089,0.07074,0.07072,0.07071,0.07065]
io_time, i_exp, o_exp = np.array(io_time), np.array(i_exp), np.array(o_exp)
io_time -= 0.05
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

all_insides, all_outsides = np.stack([endo_insides_calcium[:surr_insides_calcium.shape[0],:], surr_insides_calcium]), np.stack([endo_outsides_calcium[:surr_insides_calcium.shape[0],:], surr_outsides_calcium])
fignames  = [os.path.join(patterned_ds_path,"fig4_H0.svg"),os.path.join(patterned_ds_path,"fig4_H1.svg")]
amax, amin = np.max(all_insides[0,:,:].mean(axis=0)), np.min(all_insides[0,:,:].mean(axis=0))
all_insides = normalize(all_insides,amin,amax)
all_outsides = normalize(all_outsides,amin,amax)
titles = [""]
for i in range(2):
    colors = ["darkgreen","#1010ff",'limegreen',"lightskyblue"]
    t_ff_resp = np.linspace(
        0, all_insides[i].shape[1] * t_res / 1000, all_insides[i].shape[1]
    )
    fig = plt.figure(figsize=(1.1, 1.3))
    ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])  # left, bottom, width, height
    if i == 0:
        h5, = plt.plot(io_time,normalize(i_exp),'-',c=colors[2],lw=3)
        h6, = plt.plot(io_time,normalize(o_exp,i_exp.min(),i_exp.max()),"-",c=colors[3],lw=3)

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
    
    mean, sem = all_insides[i,:,:].mean(axis=0), all_insides[i,:,:].std(
        axis=0
    ) / np.sqrt(all_insides[i,:,:].shape[0])
    if 0:
        h1 = plt.gca().fill_between(
            t_ff_resp,
            mean - sem,
            mean + sem,
            color=colors[0],
            alpha=0.4,
            #label="_nolegend_",
        )
    
    h2, = plt.plot(t_ff_resp, mean, "--", c=colors[0],lw=1.5)

    mean, sem = all_outsides[i,:,:].mean(axis=0), all_outsides[i,:,:].std(
        axis=0
    ) / np.sqrt(all_outsides[i,:,:].shape[0])
    if 0:
        h3 = plt.gca().fill_between(
            t_ff_resp,
            mean - sem,
            mean + sem,
            color=colors[1],
            alpha=0.4,
            #label="_nolegend_",
        )
    h4, = plt.plot(t_ff_resp, mean, "--", c=colors[1],lw=1.5)
    
         
    plt.xlabel("Time (s)", fontsize=9)
    if i == 0:
        plt.ylabel("Normalized\nresponse", fontsize=9,labelpad=-6)
    plt.xlim(0, 4)
    plt.ylim(0, 1.1)
    plt.xticks([0,1,2,3,4])
    plt.yticks([0,1])
    if i == 1: 
        plt.gca().spines['left'].set_visible(False)
        plt.yticks([])
    plt.gca().yaxis.set_label_coords(-0.05,0.5)

    if i == 0:
        fake_h = plt.plot([], [], color=(0, 0, 0, 0), label=" ")[0]
        plt.legend(
            [fake_h,h5,h6,fake_h,h2,h4],
            ["Experiment","Within ROI","Outside ROI","Model","Within ROI","Outside ROI"],
            loc="upper center",
            handlelength = 1.1,
            bbox_to_anchor=(1.2, -0.35),
            frameon=False,
            fontsize=9,
            ncol=2
        )
        
    plt.tick_params(labelsize=9, size=4, width=1.5)
    plt.title("Endogenous" if i == 0 else "Surrogate",fontsize=9)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['bottom'].set_linewidth(1.5)
    plt.gca().spines['left'].set_linewidth(1.5)
    fig.savefig(fignames[i],dpi=dpi, bbox_inches="tight",transparent=True)
    
#######################
# Panel I
#######################

def plot_corr(v, x, ax, color="k", scattercolor="silver"):
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
    

fig = plt.figure(figsize=(1.3, 1.3))
ax0 = fig.add_axes([0.15, 0.15, 0.8, 0.8])
h1,h2,h3 = plot_corr(endo_corrs.flatten(), 0 * np.ones_like(endo_corrs.flatten()), ax0)
plot_corr(surr_corrs.flatten(), 1 * np.ones_like(surr_corrs.flatten()), ax0)

fs = 10
#plt.text(0.62,surr_corrs.mean()-surr_corrs.std()*2 -0.021,"*",fontsize=10,ha='right')
#plt.text(0.63,surr_corrs.mean()-surr_corrs.std()*3-0.021,"**",fontsize=10,ha='right')

ax0.set_ylim(0.5, 0.9)
#ax0.set_xlim(-2.1, 2.1)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['bottom'].set_linewidth(1.5)
plt.gca().spines['left'].set_linewidth(1.5)

plt.tick_params(labelsize=9, size=4, width=1.5)
ax0.set_yticks([0.5, 0.9],labels=["0.5", "0.8"],fontsize=9)
ax0.set_xticks([0,1],labels=["Endog", "Surr"],fontsize=9)
ax0.set_ylabel("Stimulus-\nresponse\ncorrelation",fontsize=9,labelpad=-16.2)
plt.legend([h1,h3,h2],["Trial","95% Confidence\nInterval","Mean"],fontsize=9,
            handlelength = 0.75,
            bbox_to_anchor=(1.19, -0.34),
            frameon=False,
           ncol=2, columnspacing=-4.5
          )
fig.savefig(os.path.join(patterned_ds_path,"Figure4I.svg"),dpi=dpi, bbox_inches="tight",transparent=True)

#########################################################
# Figure 5
#########################################################

#######################
# Panel A
#######################

osc = ["#f58231", "#1cb41b", "#911eb4", "#1244f4"]  # or surr colors
def color_to_rgb(c):
    if (type(c) == list or type(c) == tuple) and len(c) == 3:
        return c
    if type(c) == str:
        if c[0] == "#":
            return matplotlib.colors.hex2color(c)
        else:
            return color_to_rgb(matplotlib.colors.cnames[c])


def imshow_pattern_opponent_color(
    im,
    pattern,
    c0,
    c1,
    filename,
    legend_labels=None,
    legend_title=None,
    cbar_label=None,
    vmin=None,
    vmax=None,
):
    if cbar_label:
        fig = plt.figure(figsize=(1.5, 1.1))
    else:
        fig = plt.figure(figsize=(1.1, 1.1))
    vmin = im.min() if vmin is None else vmin
    vmax = im.max() if vmax is None else vmax
    gray_im = plt.imshow(im, cmap="gray", vmin=vmin, vmax=vmax)
    alpha = 0.25
    fontsize = 7
    c = [color_to_rgb(c0), color_to_rgb(c1)]
    for i in range(2):
        pattern_img = np.zeros((100, 100, 4), dtype=np.uint8)
        pattern_img[:, :, :3][pattern == i] = (np.array(c[i]) * 255).astype(np.uint8)
        pattern_img[:, :, 3][pattern == i] = int(alpha * 255)
        plt.imshow(pattern_img)
    plt.gca().get_xaxis().set_visible(False)
    plt.gca().get_yaxis().set_visible(False)
    if legend_labels is not None:
        h0 = plt.gca().fill_between(
            [0],
            [0],
            [0.00000001],
            color=c0,
            alpha=1,
        )
        h1 = plt.gca().fill_between(
            [0],
            [0],
            [0.00000001],
            color=c1,
            alpha=1,
        )
        leg = plt.legend(
            [h1, h0],
            legend_labels,
            title=legend_title,
            title_fontsize=fontsize,
            fontsize=fontsize,
            handlelength=0.7,
            bbox_to_anchor=(1.146, -0.0),
            frameon=False,
        )
        leg._legend_box.align = "left"
    if cbar_label is not None:
        cbar = plt.colorbar(mappable=gray_im, aspect=13, location="left")
        cbar.set_label(label=cbar_label, labelpad=-11.5, fontsize=8)
        if np.isclose(vmin, np.round(vmin)) and np.isclose(vmax, np.round(vmax)):
            cbar.set_ticks([vmin, vmax], labels=["%d" % vmin, "%d" % vmax], fontsize=8)
        else:
            cbar.set_ticks(
                [vmin, vmax], labels=["%.1f" % vmin, "%.1f" % vmax], fontsize=8
            )
    fig.savefig(
        filename, dpi=dpi, bbox_inches="tight", transparent=True
    )


pattern_idx = 5
imshow_pattern_opponent_color(
    A_surr[pattern_idx, 0, :, :, 39],
    surr_stims[pattern_idx, ...],
    osc[3],
    osc[2],
    os.path.join(patterned_ds_path,"Figure5A00.svg"),
    vmin=0,
    vmax=40,
)
imshow_pattern_opponent_color(
    A_surr_calcium[pattern_idx, 0, :, :, 39],
    surr_stims[pattern_idx, ...],
    osc[3],
    osc[2],
    os.path.join(patterned_ds_path,"Figure5A01.svg"),
    vmin=0,
    vmax=1.5,
    legend_labels=["Stim. area (SA)", "Non-stim.\narea (NSA)"],
    legend_title="Surrogate",
)
imshow_pattern_opponent_color(
    A_endo[pattern_idx, 2, :, :, 39],
    endo_stims[pattern_idx, ...],
    osc[1],
    osc[0],
    os.path.join(patterned_ds_path,"Figure5A10.svg"),
    cbar_label="Firing\nrate (sp/s)",
    vmin=0,
    vmax=40,
)
imshow_pattern_opponent_color(
    A_endo_calcium[pattern_idx, 2, :, :, 39],
    endo_stims[pattern_idx, ...],
    osc[1],
    osc[0],
    os.path.join(patterned_ds_path,"Figure5A11.svg"),
    cbar_label="$\Delta$F/F",
    vmin=0,
    vmax=1.5,
    legend_labels=["Stim. area (SA)", "Non-stim.\narea (NSA)"],
    legend_title="Endogenous",
)

#######################
# Panel B
#######################

def plot_linesem(x,y,color,style):
    mean, sem = y.mean(axis=0), y.std(axis=0) / np.sqrt(y.shape[0])
    h1 = plt.gca().fill_between(
        x,
        mean - sem,
        mean + sem,
        color=color,
        alpha=0.2,
        label="_nolegend_",
    )
    plt.plot(x,mean,style,c=color,alpha=1,label="_nolegend_",lw=1.5)
    
arrs = [[endo_insides,endo_outsides,surr_insides,surr_outsides],
        [endo_insides_calcium,endo_outsides_calcium,surr_insides_calcium,surr_outsides_calcium]]
t = np.linspace(0,4,arrs[0][0].shape[1])
ylabels = ["Mean firing\nrate (sp/s)","$\Delta$F/F"]
ylabelpads=[-16,-9]
yticks = [[2.5,20],[0,1]]

for k in range(2):
    fig = plt.figure(figsize=(0.85,1.1))
    fontsize=7
    rect = matplotlib.patches.Rectangle(
        (1, 0),
        1,
        yticks[k][1],
        linewidth=1,
        facecolor=(255 / 255, 223 / 255, 0),
        alpha=0.6,
    )
    plt.gca().add_patch(rect)
    
    for i in range(4):
        plot_linesem(t,arrs[k][i],osc[i],'-')
        
    plt.xlabel("Time (s)", fontsize=fontsize,labelpad=1)
    plt.ylabel(ylabels[k], fontsize=fontsize,labelpad=ylabelpads[k])
    plt.xlim(0,4)
    plt.ylim(yticks[k][0],yticks[k][1])
    plt.xticks([0,1,2,3,4])
    plt.yticks(yticks[k])

    plt.tick_params(labelsize=fontsize, size=3, width=1)
    plt.gca().spines[['right','top']].set_visible(False)
    plt.gca().spines[['bottom','left']].set_linewidth(1)
    if k == 1:
        plt.legend(["Light ON"],handlelength=0.8,
                   fontsize=7,
                   bbox_to_anchor=(1.0, -0.415),
                   frameon=False)
    fig.savefig(os.path.join(patterned_ds_path,"Figure5B%d.svg" % k),dpi=dpi, bbox_inches="tight",transparent=True)
    
#######################
# Panel C
#######################

colors = [osc[0], osc[2]]
arrs_or = [or_hist_endo, or_hist_surr]
arrs_A = [or_hist_A_endo, or_hist_A_surr]
for k in range(2):
    fig = plt.figure(figsize=(1.0, 1.1))
    fontsize = 6.5
    aa = arrs_A[k][pattern_idx].mean(axis=(0, 1))
    plt.plot(ors-np.pi/2, fold_in_half(aa/aa.sum()),c=colors[k],lw=1.5)
    (h0,) = plt.plot(
        ors-np.pi/2, fold_in_half(arrs_or[k][pattern_idx, :] / arrs_or[k][pattern_idx, :].sum()), "k", lw=1.5
    )
    if k == 0:
        (h1,) = plt.plot([0], [0], c=colors[0])
        (h2,) = plt.plot([0], [0], c=colors[1])
        plt.legend(
            [h0, h1, h2],
            ["Stimulated or.", "Endog. DAOD", "Surr. DAOD"],
            handlelength=0.5,
            fontsize=fontsize,
            bbox_to_anchor=(1.08, -0.20),
            frameon=False,
        )
    plt.xlabel("Orientation", fontsize=fontsize, labelpad=-6.5)
    plt.ylabel("Probability\ndensity", fontsize=fontsize, labelpad=-15)
    plt.xlim(-np.pi/2, np.pi/2)
    plt.xticks([-np.pi/2, np.pi/2], ["$-\pi/2$", "$\pi/2$"], fontsize=fontsize)
    plt.ylim(0, 0.15)
    plt.yticks([0, 0.15])

    plt.tick_params(labelsize=fontsize, size=3, width=1)
    plt.gca().spines[["right", "top"]].set_visible(False)
    plt.gca().spines[["left", "bottom"]].set_linewidth(1)
    fig.savefig(
        os.path.join(patterned_ds_path,"Figure5C%d.svg" % k), dpi=dpi, bbox_inches="tight", transparent=True
    )
    
#######################
# Panel D
#######################

fig=plt.figure(figsize=(1,1))
fontsize=6.5
plt.plot(or_hist_endo_entropy,or_hist_A_endo_entropy.reshape([8,-1]),'.',ms=5,c=osc[0])
plt.plot(or_hist_surr_entropy,or_hist_A_surr_entropy.reshape([8,-1]),'.',ms=5,c=osc[2])
plt.plot(np.linspace(2.0,3.5),np.linspace(2.0,3.5),'k--',lw=1.3)
h0, = plt.plot([0],[0],'.',ms=5,c=osc[0])
h1, = plt.plot([0],[0],'.',ms=5,c=osc[2])
plt.xlabel("OD\nentropy",fontsize=fontsize,labelpad=-10)
plt.ylabel("DAOD\nentropy",fontsize=fontsize,labelpad=-13)
plt.xlim(2.0,3.5)
plt.ylim(2.0,3.5)
plt.xticks([2.0,3.5],fontsize=fontsize)
plt.yticks([2.0,3.5],fontsize=fontsize)
plt.tick_params(labelsize=fontsize, size=3, width=1)
plt.gca().spines[['right','top']].set_visible(False)
plt.gca().spines[['left','bottom']].set_linewidth(1)

h_spont, = plt.plot([0],[0],'.',ms=5,c='k')
h0, = plt.plot([0],[0],'.',ms=5,c=osc[0])
h1, = plt.plot([0],[0],'.',ms=5,c=osc[2])
plt.legend([h0,h1,h_spont],["Endogenous","Surrogate",'Spontaneous'],handlelength=1,fontsize=fontsize,
           bbox_to_anchor=(-0.15, -0.20),loc='upper left',frameon=False)
fig.savefig(os.path.join(patterned_ds_path,"Figure5D.svg"),dpi=dpi, bbox_inches="tight",transparent=True)

#######################
# Panel E
#######################

def calc_r(entropy, fr):
    X = np.stack([entropy.flatten(),fr.flatten()]).T
    pca = PCA(n_components=1).fit(X)
    slope = pca.components_[0, 1] / pca.components_[0, 0]
    intercept = np.mean(X[:, 1]) - slope * np.mean(X[:, 0])
    r = scipy.stats.pearsonr(X[:,0],X[:,1])[0]
    return r, slope, intercept

from sklearn.decomposition import PCA
fig=plt.figure(figsize=(1.1,1.1))
fontsize=6.5
r_endo, slope_endo, intercept_endo = calc_r(scipy.stats.entropy(or_hist_A_endo, axis=-1), A_endo_stim.mean(axis=2))
r_surr, slope_surr, intercept_surr = calc_r(scipy.stats.entropy(or_hist_A_surr, axis=-1), A_surr_stim.mean(axis=2))
r_spont, slope_spont, intercept_spont = calc_r(spont_or_hist_entropy, A_spont_exc.mean(axis=(0,1)))  

x_endo = np.linspace(2.35,2.68,100)
x_surr = np.linspace(3.2,3.35100)
x_spont = np.linspace(3.1,3.42,100)
plt.plot(x_endo,intercept_endo + slope_endo * x_endo,lw=0.7,c=osc[0])
plt.text(2.62,22.5,"r=%.2f" % r_endo,fontsize=6,c=osc[0])
plt.plot(scipy.stats.entropy(or_hist_A_endo, axis=-1).flatten(),A_endo_stim.mean(axis=(2)).flatten(),'.',ms=5,c=osc[0],alpha=0.05)
plt.plot(scipy.stats.entropy(or_hist_A_surr, axis=-1).flatten(),A_surr_stim.mean(axis=2).flatten(),'.',ms=5,c=osc[2],alpha=0.05)
plt.plot(spont_or_hist_entropy,A_spont_exc.mean(axis=(0,1)),'.',ms=5,c='k',alpha=0.05)
plt.ylabel("Firing\nrate (sp/s)",fontsize=fontsize,labelpad=-8)
plt.xlabel("DAOD\nentropy",fontsize=fontsize,labelpad=-9)
plt.xlim(2.2,3.5)
plt.ylim(0,25)
plt.xticks([2.2,3.5],fontsize=fontsize)
plt.yticks([0,25],fontsize=fontsize)
plt.tick_params(labelsize=fontsize, size=3, width=1)
plt.gca().spines[['right','top']].set_visible(False)
plt.gca().spines[['left','bottom']].set_linewidth(1)
fig.savefig(os.path.join(patterned_ds_path,"Figure5F.svg"),dpi=dpi, bbox_inches="tight",transparent=True)

#########################################################
# Figure 6, Figure S8
#########################################################

fig, ax = plt.subplots(2,3,figsize=(10,6))

plt.subplots_adjust(wspace=0.45,hspace=0.45)
for i in range(2):
    for j in range(3):
        ax[i,j].tick_params(labelsize=9, size=4, width=1)
        ax[i,j].spines[['right','top']].set_visible(False)
        ax[i,j].spines[['left','bottom']].set_linewidth(1)

axs = ax[0,0]
m_e = A_dot_exc_[...,20:40].mean(axis=(1,2,3,4))
m_i = A_dot_inh_[...,20:40].mean(axis=(1,2,3,4))
axs.plot(radii,np.ones_like(m_e) * m_e[0],'r:')
axs.plot(radii,np.ones_like(m_i) * m_i[0],'b:')
axs.plot(radii,m_e,'r')
axs.plot(radii,m_i,'b')
axs.set_xlabel("Radius (um)")
axs.set_ylabel("Firing rate (sp/s)")

axs = ax[0,1]
m_e = A_dot_exc_c_const_mean[...,20:40].mean(axis=(1,2))
m_i = A_dot_inh_c_const_mean[...,20:40].mean(axis=(1,2))
axs.plot(radii,np.ones_like(m_e) * m_e[0],'r:')
axs.plot(radii,np.ones_like(m_i) * m_i[0],'b:')
axs.plot(radii,m_e,'r')
axs.plot(radii,m_i,'b')
axs.set_xlabel("Radius (um)")
axs.set_ylabel("Firing rate (sp/s)")

axs = ax[0,2]
m_e = A_dot_exc_s_const_mean[...,20:40].mean(axis=(1,2))
m_i = A_dot_inh_s_const_mean[...,20:40].mean(axis=(1,2))
axs.plot(radii,np.ones_like(m_e) * m_e[0],'r:')
axs.plot(radii,np.ones_like(m_i) * m_i[0],'b:')
axs.plot(radii,m_e,'r')
axs.plot(radii,m_i,'b')
axs.set_xlabel("Radius (um)")
axs.set_ylabel("Firing rate (sp/s)")

axs = ax[1,0]
axs.ticklabel_format(style='sci', axis='y', scilimits=(4,4))
axs.plot(radii,A_dot_exc_csum_total,'r')
axs.plot(radii,A_dot_inh_csum_total,'b')
axs.spines[['top','right']].set_visible(False)
axs.spines[['left','bottom']].set_linewidth(1.5)
axs.set_xlabel("Radii (um)")
axs.set_ylabel("Total spikes over stim. period\nunder stim area")

axs = ax[1,1]
axs.plot(radii,A_dot_exc_csum[...,20:40].mean(axis=(1,2)) /A_dot_inh_csum[...,20:40].mean(axis=(1,2)),'k')
axs.spines[['top','right']].set_visible(False)
axs.spines[['left','bottom']].set_linewidth(1.5)
axs.set_xlabel("Radii (um)")
axs.set_ylabel("E/I ratio of sum of firing rates\nunder stim area")

axs = ax[1,2]
axs.plot([radii[0],radii[-1]],[fr_close_exc[0],fr_close_exc[0]],'r:',label="_nolegend_")
axs.plot([radii[0],radii[-1]],[fr_close_inh[0],fr_close_inh[0]],'b:',label="_nolegend_")
axs.plot(radii,fr_close_exc,'r')
axs.plot(radii,fr_far_exc,'r',linestyle=(5, (8, 3)))
axs.plot(radii,fr_close_inh,'b')
axs.plot(radii,fr_far_inh,'b',linestyle=(5, (8, 3)))
axs.set_ylim(1,9)
axs.set_xlabel("Radius (um)")
axs.set_ylabel("Firing rate (sp/s)")
axs.legend(["Close or. exc","Far or. exc","Close or. inh","Far or. inh"],bbox_to_anchor=(1.0, 1.1),frameon=False)

fig.savefig(os.path.join(center_stim_ds_path,"Figure6.svg"),dpi=dpi, bbox_inches="tight",transparent=True)
