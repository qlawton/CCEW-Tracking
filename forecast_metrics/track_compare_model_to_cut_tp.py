#!/usr/bin/env python
# coding: utf-8
# %%

# %%


import xarray as xr
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
import sys
import cartopy.crs as ccrs
import pandas as pd
import pickle 
from cartopy.mpl.ticker import LongitudeFormatter

model_in = sys.argv[1]

save_dir = '/glade/work/qlawton/CCEW_TRACKS/TP/FULL_DAILY_MEAN/'+model_in.upper()+'/'
olr_data = '/glade/work/qlawton/DATA/IMERG/DAILY_FILTERED_WITH_LATE/daily_mean_padded_imerg-daily.Kelvin.nc'
ref_data = '/glade/work/qlawton/DATA/IMERG/DAILY_FILTERED_WITH_LATE/VAR/daily_mean_var-daily-Kelvin.nc'
master_ref_tracks = '/glade/work/qlawton/CCEW_TRACKS/DAILY_IMERG_FULL_WITH_LATE/padded_daily_mean_CCKW_tracks_2001-2024_full_adj_seam_signal_init_1_cont_0.25.nc'
lon_thresh = int(sys.argv[2])
model_only = False

#### END MANUAL

if model_only == True:
    print('Shorter model lengths being tested')
    ERA5_cut_err = '/glade/work/qlawton/DATA/FORECAST_EVAL/DAILY_MEAN/'+model_in.upper()+'/CUT_TP/'+model_in+'_'+str(lon_thresh)+'lon_model_only_updated_with_raw_ext_error_data.nc'
    ERA5_cut_waves = '/glade/work/qlawton/DATA/FORECAST_EVAL/DAILY_MEAN/'+model_in.upper()+'/CUT_TP/'+model_in+'_'+str(lon_thresh)+'lon_model_only_updated_with_raw_ext_matching_wave_nums.pkl'
else:
    ERA5_cut_err = '/glade/work/qlawton/DATA/FORECAST_EVAL/DAILY_MEAN/'+model_in.upper()+'/CUT_TP/'+model_in+'_'+str(lon_thresh)+'lon_updated_with_raw_ext_error_data.nc'
    ERA5_cut_waves = '/glade/work/qlawton/DATA/FORECAST_EVAL/DAILY_MEAN/'+model_in.upper()+'/CUT_TP/'+model_in+'_'+str(lon_thresh)+'lon_updated_with_raw_ext_matching_wave_nums.pkl'

### Information for data to use for errors
ref_dir = '/glade/work/qlawton/CCEW_TRACKS/TP/FULL_DAILY_MEAN/'+model_in.upper()+'/'
ref_start = 'CCKW_tracks_init_1_cont_0.25_Kelvin_obs_for_'+model_in+'_'
ref_xr = xr.open_dataset(ref_data)

### Information for data to use for "master" tracks
master_tracks_xr = xr.open_dataset(master_ref_tracks)
ERA5_cut_err_xr = xr.open_dataset(ERA5_cut_err)
with open(ERA5_cut_waves, 'rb') as file:
    ERA5_cut_wave_list = pickle.load(file)

### Other miscellaneous data
pad_len = 30
model_len_hr = 15*24

#file_beg = 'CCKW_tracks_init_1_cont_0.25_Kelvin_obs_for_ec_cf_'
file_beg = 'CCKW_tracks_init_1_cont_0.25_Kelvin_'+model_in+'_'


# start_date = datetime(2023, 8, 29, 0, 0) 
# #start_date = datetime(2024, 4, 1, 0, 0)
# end_date = datetime(2024, 9, 1, 0, 0)

start_date = datetime(2018, 6, 6, 0, 0)
end_date = datetime(2019, 6, 4, 0, 0)

delta = timedelta(days=3)

date_list = []
current_date = start_date
while current_date <= end_date:
    formatted_date = current_date.strftime('%Y%m%d%H')
    date_list.append(formatted_date)
    current_date += delta

ref_list = []
ext_list = []
orig_list = []

for date_i in range(len(date_list)):
    date = date_list[date_i]
    date_new = datetime.strftime(datetime.strptime(date, '%Y%m%d%H'), '%Y-%m-%d-%H')
    print(date_new)
    ext_file = save_dir+file_beg+date+'_extended_length_padded.nc'
    orig_file = save_dir+file_beg+date+'_orig_length_padded.nc'

    if model_only == True:
        ref_file = ref_dir+ref_start+date+'_orig_length_padded.nc'
    else:
        ref_file = ref_dir+ref_start+date+'_extended_length_padded.nc'
    ref_list.append(xr.open_dataset(ref_file).isel(time=slice(pad_len, -pad_len)).sel(time=slice(date_new, None)))
    ext_list.append(xr.open_dataset(ext_file).isel(time=slice(pad_len, -pad_len)).sel(time=slice(date_new, None)))
    orig_list.append(xr.open_dataset(orig_file).isel(time=slice(pad_len, -pad_len)).sel(time=slice(date_new, None)))


olr_data_xr = (xr.open_dataset(olr_data).precipitation/np.sqrt(ref_xr.precipitation)).isel(time=slice(pad_len, -pad_len))

olr_hov_xr = olr_data_xr.sel(lat=slice(-10, 10)).mean(dim='lat')
olr_hov_xr.coords['lon'] = (olr_hov_xr.coords['lon'] + 180) % 360 - 180
olr_hov_xr = olr_hov_xr.sortby(olr_hov_xr.lon)

def find_matches(compare_tracks, ref_tracks, lon_thresh):
    final_match_list = []
    final_min_list = []
    final_time_list = []
    final_wave_number_list = []
    ### We first just want to focus on systems in the time range of our "model" run
    st_time = compare_tracks.time.values[0]
    end_time = compare_tracks.time.values[-1]
    ref_tracks = ref_tracks.sel(time=slice(st_time, end_time)).dropna(dim='system', how='all')
    
    for cmp_wv_i in range(len(compare_tracks.system)):
        #print('Testing wave: ', cmp_wv_i)
        wave_cmp = compare_tracks.isel(system = cmp_wv_i)#.dropna(dim='time')

        if len(wave_cmp.CCKW_str.dropna('time').time) == 0:
            #print('Empty slice, continuing...')
            continue
        
        yr_st = wave_cmp.time.dt.year.values[0]
        yr_end = wave_cmp.time.dt.year.values[-1]

        match_list = []
        min_dist_list = []
        first_match_time_list = []
        for ref_wv_i in range(len(ref_tracks.system)):
            #print('Reference wave')
            wave_ref = ref_tracks.isel(system=ref_wv_i)#.dropna(dim='time')
            if len(wave_ref.CCKW_str.dropna('time').time)==0:
                continue
            ref_st = wave_ref.time.dt.year.values[0]
            ref_end = wave_cmp.time.dt.year.values[0]

            if np.isin(np.array(ref_st, ref_end), np.array(yr_st, yr_end)).any() == False:
                continue

            
            overlap = wave_cmp['time'].isin(wave_ref['time'])

            overlap_exists = overlap.any().item()
            if overlap_exists == False:
                continue
            
            abs_diff_lon = np.abs((wave_cmp['CCKW_lon'] - wave_ref['CCKW_lon']))
            abs_diff_lon = abs_diff_lon.where(abs_diff_lon<=lon_thresh, drop=True)
            if len(abs_diff_lon) == 0: #If they aren't...gonn have to skip
                continue
            #print('Match found')
            match_list.append(wave_ref.system.values)
            min_dist_list.append(abs_diff_lon.min())
            first_match_time_list.append(abs_diff_lon.time.values[0])
        if len(match_list) == 0:
            final_match_list.append([np.nan])
            final_min_list.append([np.nan])
            final_time_list.append([np.nan])
        else:
            final_match_list.append(match_list)
            final_min_list.append(min_dist_list)
            final_time_list.append(first_match_time_list)
        final_wave_number_list.append(wave_cmp.system.values)
    return final_wave_number_list, final_match_list, final_min_list, final_time_list


ref_tracks_in = ref_list.copy()

if model_only == True:
    print('Shorter model length input')
    compare_tracks_in = orig_list.copy()
else:
    compare_tracks_in = ext_list.copy()


model_run_wave_nums = []
model_run_matches = [] 
model_run_mins = []
model_run_times = []

for i in range(len(compare_tracks_in)):
    print('Running on Model Number:', i+1)
    wv_out, match_out, mins_out, times_out = find_matches(compare_tracks_in[i], ref_tracks_in[i], lon_thresh)
    model_run_wave_nums.append(wv_out)
    model_run_matches.append(match_out)
    model_run_mins.append(mins_out)
    model_run_times.append(times_out)


final_connect_wave_count = []
final_ghost_wave_count = []
for i in range(len(model_run_matches)):
    d_in = model_run_matches[i]
    ## Count number of nans
    connect_count = 0
    ghost_count = 0
    for j in range(len(d_in)): # Loop over and check if NaN or not
        if any(np.isnan(d_in[j])):
            ghost_count=ghost_count+1
        else:
            connect_count=connect_count+1
    final_connect_wave_count.append(connect_count)
    final_ghost_wave_count.append(ghost_count)
final_connect_wave_count = np.array(final_connect_wave_count)
final_ghost_wave_count = np.array(final_ghost_wave_count)


connect_count_xr = xr.DataArray(final_connect_wave_count, 
                                coords = [date_list],
                                dims=['model_run'])
ghost_count_xr = xr.DataArray(final_ghost_wave_count, 
                                coords = [date_list],
                                dims=['model_run'])

all_dates_connect_model = []
all_refs_num_list = []
for date_i in range(len(date_list)):
    ref_tracks_for_loop = ref_tracks_in[date_i]
    ref_system_num_list = []
    model_system_i_list = []
    full_wave_connect_model = []
    for ref_i in range(len(ref_tracks_for_loop.system)):
        connect_waves_model = []
        system_in = ref_tracks_for_loop.system.isel(system=ref_i).values

        matches_in = model_run_matches[date_i]
        waves_in = model_run_wave_nums[date_i]
        matches_list = []

        for mi in range(len(matches_in)):
            if system_in in matches_in[mi]:
                matches_list.append(waves_in[mi])
        if len(matches_list) != 0:        
            connect_waves_model.append(matches_list)
        else:
            connect_waves_model.append([np.array(np.nan)])
        full_wave_connect_model.append(connect_waves_model)
        ref_system_num_list.append(system_in)
    all_dates_connect_model.append(full_wave_connect_model)
    all_refs_num_list.append(ref_system_num_list)

combined_error_list = []

### Compute the track and strength errors for each dataset
for date_i in range(len(date_list)):
    date_val = date_list[date_i]

    linked_model_waves = all_dates_connect_model[date_i]
    all_ref_waves = all_refs_num_list[date_i]

    ref_data_in = ref_tracks_in[date_i]
    model_data_in = compare_tracks_in[date_i]

    ### New arrays
    track_error_xr = xr.full_like(ref_data_in.CCKW_lon, np.nan).copy()
    str_error_xr = xr.full_like(ref_data_in.CCKW_lon, np.nan).copy()
    model_track_xr = xr.full_like(ref_data_in.CCKW_lon, np.nan).copy()
    model_str_xr = xr.full_like(ref_data_in.CCKW_lon, np.nan).copy()
    ref_track_xr = xr.full_like(ref_data_in.CCKW_lon, np.nan).copy()
    ref_str_xr = xr.full_like(ref_data_in.CCKW_lon, np.nan).copy()
    exist_times_xr = xr.full_like(ref_data_in.CCKW_lon, np.nan).copy()
    
    exist_bool = []
    ref_bool = []
    for ref_sys_i in range(len(all_ref_waves)):
        ref_wave_num = int(all_ref_waves[ref_sys_i])
        waves_in = linked_model_waves[ref_sys_i][0]
        ref_data_xr = ref_data_in.sel(system=ref_wave_num).dropna(dim='time')
        ref_data_full = ref_data_in.sel(system=ref_wave_num)      
        if len(ref_data_xr.time) == 0: #If we don't have comparable reference data
            ref_bool.append(0)
            exist_bool.append(0)
            continue
        ref_bool.append(1)
        if np.isnan(waves_in).any():
            exist_bool.append(0)
            continue
        exist_bool.append(1)
        ref_time_start = ref_data_xr.isel(time=0).time
        ref_time_end = ref_data_xr.isel(time=-1).time
        if len(waves_in) > 1: #If we have more than one wave, let's combine them
            step = 0
            for i in range(len(waves_in)):
                if step == 0:
                    combined_tracks = model_data_in.CCKW_lon.sel(system = waves_in[i])
                    combined_str = model_data_in.CCKW_str.sel(system=waves_in[i])
                else:
                    combined_tracks = combined_tracks.combine_first(model_data_in.CCKW_lon.sel(system = waves_in[i]))
                    combined_str = combined_str.combine_first(model_data_in.CCKW_str.sel(system=waves_in[i]))
                step = step+1
        else: #Otherwise, jsut set this variable
            combined_tracks = model_data_in.CCKW_lon.sel(system = waves_in[0])
            combined_str = model_data_in.CCKW_str.sel(system = waves_in[0])
     ### Next we compute the differences
        track_error = ((combined_tracks - ref_data_xr.CCKW_lon ) + 180) % 360 - 180 ## Compute and normalize
        str_error = combined_str*-1 - ref_data_xr.CCKW_str*-1 

        track_error_xr.loc[{'system': ref_wave_num, 'time': track_error.time}] = track_error.squeeze()
        str_error_xr.loc[{'system': ref_wave_num, 'time': str_error.time}] = str_error.squeeze()

        ### Now we save out the corresponding "raw" data
        model_track_xr.loc[{'system': ref_wave_num, 'time':combined_tracks.time}] = combined_tracks.squeeze()
        model_str_xr.loc[{'system': ref_wave_num, 'time':combined_str.time}] = combined_str.squeeze()*-1
        ref_track_xr.loc[{'system': ref_wave_num, 'time':ref_data_xr.CCKW_lon.time}] = ref_data_xr.CCKW_lon.squeeze()
        ref_str_xr.loc[{'system': ref_wave_num, 'time':(ref_data_xr.CCKW_str*-1).time}] = (ref_data_xr.CCKW_str*-1).squeeze()

        final_array = xr.full_like(combined_tracks, np.nan).copy()
        # Set 2 where combined_tracks exists and CCKW_lon does not (i.e., is NaN)
        final_array[~np.isnan(combined_tracks) & np.isnan(ref_data_full.CCKW_lon)] = 2
        # Set 1 where CCKW_lon exists (i.e., is not NaN) and combined_tracks also exists
        final_array[~np.isnan(ref_data_full.CCKW_lon) & ~np.isnan(combined_tracks)] = 1
        # Set 0 where CCKW_lon exists (i.e., is not NaN) and combined_tracks does not
        final_array[~np.isnan(ref_data_full.CCKW_lon) & np.isnan(combined_tracks)] = 0

        exist_times_xr.loc[{'system': ref_wave_num, 'time':combined_tracks.time}] = final_array

    ### Next, we want to identify the reference waves that the model had a chance of capturing
    ref_bool_xr = xr.DataArray(ref_bool, dims=['system'], coords=[track_error_xr.system]) 
    exist_bool_xr = xr.DataArray(exist_bool, dims=['system'], coords=[track_error_xr.system]) 
    
    combined_xr = xr.merge([track_error_xr.to_dataset(name='track_error'), str_error_xr.to_dataset(name='str_error'), 
                            model_track_xr.to_dataset(name='model_tracks'), model_str_xr.to_dataset(name='model_str'),
                            ref_track_xr.to_dataset(name='ref_tracks'), ref_str_xr.to_dataset(name='ref_str'),
                            ref_bool_xr.to_dataset(name='ref_bool'), exist_bool_xr.to_dataset(name='exist_bool'),
                           exist_times_xr.to_dataset(name='exist_compare')]) 

    combined_xr = combined_xr.where(combined_xr.ref_bool, drop = True)
    success_rate = (np.sum(combined_xr.exist_bool)/len(combined_xr.exist_bool)).values
    combined_xr['model_success_rate'] = success_rate
    combined_xr['model_ghost_count'] = ghost_count_xr.isel(model_run=date_i).values
    combined_error_list.append(combined_xr)

# ## Finally, let's make the connection to the reference dataset
# 1. Loop over all of the reference waves. For each wave, identify where there exists ERA5 cut waves.
# 2. Take these cut waves, pull out the corresponding model wave(s), reference wave(s), error data, and other statistics.
# 3. Save out this data

model_track_error_xr = xr.full_like(ERA5_cut_err_xr['track_error'], np.nan)
model_str_error_xr = xr.full_like(ERA5_cut_err_xr['track_error'], np.nan)
model_exist_compare = xr.full_like(ERA5_cut_err_xr['track_error'], np.nan)
model_success_bool = xr.full_like(ERA5_cut_err_xr['success_bool'], np.nan)

ref_raw_tracks = xr.full_like(ERA5_cut_err_xr['ref_tracks'], np.nan)
cut_raw_tracks = xr.full_like(ERA5_cut_err_xr['ref_tracks'], np.nan)
model_raw_tracks = xr.full_like(ERA5_cut_err_xr['ref_tracks'], np.nan)

ref_raw_str = xr.full_like(ERA5_cut_err_xr['ref_tracks'], np.nan)
cut_raw_str = xr.full_like(ERA5_cut_err_xr['ref_tracks'], np.nan)
model_raw_str = xr.full_like(ERA5_cut_err_xr['ref_tracks'], np.nan)

ref_success_rate =  xr.DataArray(
    np.nan * np.ones_like(ERA5_cut_err_xr['system']),
    dims=['system'],
    coords={'system': ERA5_cut_err_xr['system']}
)

ref_model_capture =  xr.DataArray(
    np.nan * np.ones_like(ERA5_cut_err_xr['system']),
    dims=['system'],
    coords={'system': ERA5_cut_err_xr['system']}
)
ref_total_expected =  xr.DataArray(
    np.nan * np.ones_like(ERA5_cut_err_xr['system']),
    dims=['system'],
    coords={'system': ERA5_cut_err_xr['system']}
)

for master_ref_i in range(len(ERA5_cut_err_xr.system)):
    master_ref_wave = ERA5_cut_err_xr.system.isel(system=master_ref_i).values
    master_ref_xr = ERA5_cut_err_xr.isel(system=master_ref_i)
    print('Working on Wave:', master_ref_wave)

    ## Here are the cooresponding ERA5 run to pull error data from 
    ### Put in an exception for no model run successes
    if (master_ref_xr.success_bool == 1).values.any():
        model_list = master_ref_xr.where(master_ref_xr.success_bool == 1, drop = True).model_run
    else:
        print('No matching waves.')
        continue
    total_expected = len(model_list)
    model_captured = 0
    for model_date_i in range(len(model_list)):
        date_in = model_list.isel(model_run = model_date_i).values
        date_i = date_list.index(date_in)
        cut_wave_list = ERA5_cut_wave_list[master_ref_i][date_i]
        error_data_in = combined_error_list[date_i].sel(system = cut_wave_list) #Pull in other data
        if len(cut_wave_list) > 1:
            step = 0
            for i in range(len(error_data_in.system)):
                if step == 0:
                    combined_track_err = error_data_in.track_error.sel(system = error_data_in.system.values[i])
                    combined_str_err = error_data_in.str_error.sel(system=error_data_in.system.values[i])
                    combined_exist_compare = error_data_in.exist_compare.sel(system=error_data_in.system.values[i])
                    combined_cut_tracks = error_data_in.ref_tracks.sel(system=error_data_in.system.values[i])
                    combined_cut_str = error_data_in.ref_str.sel(system=error_data_in.system.values[i])
                    combined_model_tracks = error_data_in.model_tracks.sel(system=error_data_in.system.values[i])
                    combined_model_str = error_data_in.model_str.sel(system=error_data_in.system.values[i])
                else:
                    combined_track_err = combined_track_err.combine_first(error_data_in.track_error.sel(system = error_data_in.system.values[i]))
                    combined_str_err = combined_str_err.combine_first(error_data_in.str_error.sel(system=error_data_in.system.values[i]))
                    combined_exist_compare = combined_exist_compare.combine_first(error_data_in.exist_compare.sel(system=error_data_in.system.values[i]))
                    combined_cut_tracks = combined_cut_tracks.combine_first(error_data_in.ref_tracks.sel(system=error_data_in.system.values[i]))
                    combined_cut_str = combined_cut_str.combine_first(error_data_in.ref_str.sel(system=error_data_in.system.values[i]))
                    combined_model_tracks = combined_model_tracks.combine_first(error_data_in.model_tracks.sel(system=error_data_in.system.values[i]))
                    combined_model_str = combined_model_str.combine_first(error_data_in.model_str.sel(system=error_data_in.system.values[i]))
                step = step+1
        else:
            combined_track_err = error_data_in.track_error.sel(system = error_data_in.system.values)
            combined_str_err = error_data_in.str_error.sel(system = error_data_in.system.values)
            combined_exist_compare = error_data_in.exist_compare.sel(system = error_data_in.system.values)
            combined_cut_tracks = error_data_in.ref_tracks.sel(system=error_data_in.system.values)
            combined_cut_str = error_data_in.ref_str.sel(system=error_data_in.system.values)
            combined_model_tracks = error_data_in.model_tracks.sel(system=error_data_in.system.values)
            combined_model_str = error_data_in.model_str.sel(system=error_data_in.system.values)
        if error_data_in.exist_bool.any() == 1:
            exist_bool_combined = 1
            model_captured = model_captured+1
        else:
            exist_bool_combined = 0

        ### Model success metrics
        model_success_bool.loc[{'system':master_ref_wave, 'model_run':date_in}] = exist_bool_combined
        model_track_error_xr.loc[{'system':master_ref_wave, 'model_run':date_in, 'time':combined_track_err.time}] = combined_track_err.squeeze().values
        model_str_error_xr.loc[{'system':master_ref_wave, 'model_run':date_in, 'time':combined_str_err.time}] = combined_str_err.squeeze().values
        model_exist_compare.loc[{'system':master_ref_wave, 'model_run':date_in, 'time':combined_exist_compare.time}] = combined_exist_compare.squeeze().values

        ### Include some of the raw tracks, renamed
        ref_raw_tracks.loc[{'system':master_ref_wave}] = ERA5_cut_err_xr['ref_tracks'].sel(system = master_ref_wave)
        cut_raw_tracks.loc[{'system':master_ref_wave, 'model_run':date_in, 'time':combined_cut_tracks.time}] = combined_cut_tracks.values.squeeze()
        model_raw_tracks.loc[{'system':master_ref_wave, 'model_run':date_in, 'time':combined_model_tracks.time}] = combined_model_tracks.values.squeeze()

        ref_raw_str.loc[{'system':master_ref_wave}] = ERA5_cut_err_xr['ref_str'].sel(system = master_ref_wave)
        cut_raw_str.loc[{'system':master_ref_wave, 'model_run':date_in, 'time':combined_cut_str.time}] = combined_cut_str.values.squeeze()
        model_raw_str.loc[{'system':master_ref_wave, 'model_run':date_in, 'time':combined_model_str.time}] = combined_model_str.values.squeeze()
        
    ref_wave_success_rate = model_captured/total_expected
    ref_success_rate.loc[{'system':master_ref_wave}] = ref_wave_success_rate
    ref_model_capture.loc[{'system':master_ref_wave}] = model_captured
    ref_total_expected.loc[{'system':master_ref_wave}] = total_expected


model_track_RMSE = np.sqrt(np.power(model_track_error_xr, 2))
model_str_RMSE = np.sqrt(np.power(model_str_error_xr, 2))


model_success_rate_xr =  xr.DataArray(
    np.nan * np.ones_like(ERA5_cut_err_xr['model_run']),
    dims=['model_run'],
    coords={'model_run': ERA5_cut_err_xr['model_run']}
)

for ds_i in range(len(combined_error_list)):
    date_in = date_list[ds_i]
    model_success_rate_xr.loc[{'model_run':date_in}] = combined_error_list[ds_i].model_success_rate

final_output_combined_xr = xr.merge([model_success_bool.to_dataset(name='model_success_bool'), 
                                    model_track_error_xr.to_dataset(name='model_track_error'),
                                    model_str_error_xr.to_dataset(name='model_str_error'), 
                                    model_track_RMSE.to_dataset(name='model_track_RMSE'), 
                                    model_str_RMSE.to_dataset(name='model_str_RMSE'),
                                    model_exist_compare.to_dataset(name='model_exist_compare'), 
                                    ref_raw_tracks.to_dataset(name='ref_raw_tracks'),
                                    cut_raw_tracks.to_dataset(name='cut_raw_tracks'),
                                    model_raw_tracks.to_dataset(name='model_raw_tracks'),
                                    ref_raw_str.to_dataset(name='ref_raw_str'),
                                    cut_raw_str.to_dataset(name='cut_raw_str'),
                                    model_raw_str.to_dataset(name='model_raw_str'),
                                    ref_success_rate.to_dataset(name='success_rate'),
                                    ref_model_capture.to_dataset(name='model_captured'),
                                    ref_total_expected.to_dataset(name='total_expected'),
                                    ghost_count_xr.to_dataset(name='model_ghost_count'), 
                                    connect_count_xr.to_dataset(name='model_connect_count'),
                                    model_success_rate_xr.to_dataset(name='model_success_rate')]
                                   )

adjusted_raw_tracks = final_output_combined_xr.ref_raw_tracks.mean(dim='model_run')
adjusted_raw_str = final_output_combined_xr.ref_raw_str.mean(dim='model_run')

max_values_xr = adjusted_raw_str.max(dim='time', skipna=True)
max_times_xr = adjusted_raw_str.idxmax(dim='time', skipna = True)

mean_values_xr = adjusted_raw_str.mean(dim='time', skipna=True)

## Get the starting longitude (first non-NaN longitude in ref_raw_tracks)
def first_non_nan_longitude(ref_str, ref_tracks):
    """Finds the first longitude where ref_str is non-NaN."""
    mask = ~np.isnan(ref_str)
    first_valid_idx = mask.argmax(axis=1)
    return ref_tracks.isel(time=first_valid_idx)

## Get the last longitude where the system is valid (non-NaN)
def last_non_nan_longitude(ref_str, ref_tracks):
    """Finds the last longitude where ref_str is non-NaN."""
    mask = ~np.isnan(ref_str)
    # Reverse the time index to find the last non-NaN value
    last_valid_idx = mask.shape[1] - 1 - mask[:, ::-1].argmax(dim='time')
    return ref_tracks.isel(time=last_valid_idx)


start_longitudes_xr = first_non_nan_longitude(adjusted_raw_str, adjusted_raw_tracks).drop_vars('time')
end_longitudes_xr = last_non_nan_longitude(adjusted_raw_str, adjusted_raw_tracks).drop_vars('time')
max_longitudes_xr = adjusted_raw_tracks.where(max_times_xr.dropna(dim='system')).sel(time=max_times_xr.dropna(dim='system')).drop_vars('time')
max_longitudes_xr = max_longitudes_xr.reindex(system=start_longitudes_xr.system, fill_value=np.nan)

final_output_combined_xr = xr.merge([final_output_combined_xr,
                                        max_values_xr.to_dataset(name='ref_max_str'), 
                                        max_times_xr.to_dataset(name='ref_time_max_str'), 
                                        mean_values_xr.to_dataset(name='ref_mean_str'), 
                                        start_longitudes_xr.to_dataset(name='ref_start_lon'), 
                                        end_longitudes_xr.to_dataset(name='ref_end_lon'), 
                                        max_longitudes_xr.to_dataset(name='ref_lon_max_str')])
 
output_dir = '/glade/work/qlawton/DATA/FORECAST_EVAL/DAILY_MEAN/'+model_in.upper()+'/TP/'

if model_only == True:
    print('model only outputs')
    output_save_name = output_dir+model_in+'_'+str(lon_thresh)+'lon_model_only_final_matched_CCKW_errors_and_tracks.nc'
    dump_ref_name = output_dir+model_in+'_'+str(lon_thresh)+'lon_model_only_model_matched_with_cut_ERA5_data_wave_nums.pkl'
else:
    output_save_name = output_dir+model_in+'_'+str(lon_thresh)+'lon_final_matched_CCKW_errors_and_tracks.nc'
    dump_ref_name = output_dir+model_in+'_'+str(lon_thresh)+'lon_model_matched_with_cut_ERA5_data_wave_nums.pkl'

file = open(dump_ref_name, 'wb')
pickle.dump(full_wave_connect_model, file)
file.close()
final_output_combined_xr.to_netcdf(output_save_name)
