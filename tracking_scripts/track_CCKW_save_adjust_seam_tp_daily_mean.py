import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from scipy import signal, stats
import pandas as pd
from matplotlib.lines import Line2D
import sys

model_in = sys.argv[3]
test_save_path = '/glade/work/qlawton/CCEW_TRACKS/TP/FULL_DAILY_MEAN/'+str(model_in)+'/'


##### FUNCTION DEFINITIONS
def run_CCKW_tracker(data_in, lon_limit_in, prominence=1, cont_threshold = 0.5, init_threshold = 0.5, hgt_in=0, extend_data = True, extension_size = 10):
    thr_in = prominence
    lon_limit = lon_limit_in
    time_in_dt = pd.to_datetime(data_in.time)
    empty_stack = np.ones((1, np.shape(time_in_dt)[0]))*np.nan

    for tm_i in range(len(time_in_dt)): #Loop over the data
        tm_raw_in = pd.to_datetime(data_in.time)[tm_i]
        #tm_in = datetime.strftime(tm_raw_in, '%Y-%m-%d-%H')
        
        data_test = data_in.isel(time=tm_i)
        
        if extend_data == True:
            data_new = np.concatenate((data_test[-extension_size:], data_test, data_test[:extension_size]))
        #### First, we will call the main ID script
            peaks_i = signal.find_peaks(data_new, height = hgt_in, prominence = thr_in)[0]
            valleys_i = signal.find_peaks(-data_new, height = hgt_in, prominence = thr_in)[0]
        else:
            peaks_i = signal.find_peaks(data_test, height = hgt_in, prominence = thr_in)[0]
            valleys_i = signal.find_peaks(-data_test, height = hgt_in, prominence = thr_in)[0]        
        
        if extend_data == True:
            peaks_i = [peak - extension_size for peak in peaks_i if extension_size<= peak < len(data_test)+extension_size]
            valleys_i = [valley - extension_size for valley in valleys_i if extension_size<= valley < len(data_test)+extension_size]
        
        lon_peak = data_test['longitude'].values[peaks_i]
        lon_valley = data_test['longitude'].values[valleys_i]
        data_peak = data_test.values[peaks_i]
        data_valley = data_test.values[valleys_i]
        
        #print(data_peak, data_valley)
        #### Get the longitudes and the VP values at these points
        
        if tm_i == 0: ## If we have our first step, we will simply just initiate our arrays

            #### These are our initial arrays
            act_lon = np.zeros((len(lon_valley), np.size(time_in_dt)))*np.nan
            sup_lon = np.zeros((len(lon_peak), np.size(time_in_dt)))*np.nan
            act_avgval = np.zeros((len(data_valley), np.size(time_in_dt)))*np.nan 
            sup_avgval = np.zeros((len(data_peak), np.size(time_in_dt)))*np.nan
            
                #### Now we actually put the values in the arrays (initialize everything in first time step)
            for wv in range(len(lon_valley)):
                act_lon[wv, tm_i] = lon_valley[wv]
                act_avgval[wv, tm_i] = data_valley[wv]
            # And that's it at first
        else: #But if we aren't just starting out, we have a little more work to do...
            used_prev_wv = []
            used_wv = []
            for prev_wv in range(act_lon.shape[0]):
                prev_wv_lon = act_lon[prev_wv, tm_i-1]
                if np.isnan(prev_wv_lon): #Obviously only want waves that exist at last time step
                    continue

                did_append = False
                #### Here's another special case: if within "lon_limits" of the right seam (the +180 mark), and we don't actually have a close match, we check otherside
                if (180 - prev_wv_lon <= lon_limit):
                    #print('Trying to append across')
                    if not any(lon_valley >= prev_wv_lon): #if there are seemingly none to attach...
                        dist_from_seam = 180 - prev_wv_lon
                        adjusted_dist = (180+np.min(lon_valley))+dist_from_seam
                        #print(adjusted_dist)
                        if adjusted_dist <= lon_limit: #and if the closest one on left side is within bounds...
                            nearest_lon = np.min(lon_valley)
                            nearest_i = np.where(lon_valley == nearest_lon)[0][0]
                            nearest_dist = nearest_lon - prev_wv_lon
                            nearest_avgval = data_valley[nearest_i]
                            if nearest_i in (used_wv):
                                #### We will check to see if we appended the closest one. If not, we want to 
                                #### replaced the current one with the closest one, and previous the previous CCKW attachment
                                used_wv_loci = np.where(used_wv == nearest_i)[0][0]
                                other_wv_to_check = used_prev_wv[used_wv_loci]
                                if np.isnan(other_wv_to_check):
                                    continue
                                    
                                if  act_lon[other_wv_to_check, tm_i]<0 and act_lon[other_wv_to_check, tm_i-1]>0: #if crossing discontinuity
                                    seam_dist = 180 - act_lon[other_wv_to_check, tm_i-1]
                                    new_dist =  act_lon[other_wv_to_check, tm_i] +180
                                    previous_dist = seam_dist+new_dist
                                else:
                                    previous_dist = act_lon[other_wv_to_check, tm_i] - act_lon[other_wv_to_check, tm_i-1]
                                if np.abs(nearest_dist) < np.abs(previous_dist):
                                    act_lon[other_wv_to_check, tm_i] = np.nan
                                    act_avgval[other_wv_to_check, tm_i] = np.nan
                                    used_prev_wv[used_wv_loci] = np.nan
                                else:
                                    continue

                            if np.abs(nearest_avgval) < cont_threshold:
                                continue
                            used_prev_wv.append(prev_wv)
                            used_wv.append(nearest_i)

                            ### And of course, update the data
                            act_lon[prev_wv, tm_i] = nearest_lon
                            act_avgval[prev_wv, tm_i] = nearest_avgval
                            did_append = True                        
                            
                if any(lon_valley >= prev_wv_lon) and did_append == False:
                    ## Like before, will will find the closest longitude ~to east~ of previous and then check distance

                    nearest_lon = np.array(lon_valley)[lon_valley >= prev_wv_lon].min()
                    nearest_i = np.where(lon_valley == nearest_lon)[0][0]
                    nearest_dist = nearest_lon - prev_wv_lon
                    
                    if nearest_i in (used_wv):
                        #### We will check to see if we appended the closest one. If not, we want to 
                        #### replace the current one with the closest one, and previous the previous CCKW attachment
                        used_wv_loci = np.where(used_wv == nearest_i)[0][0]
                        other_wv_to_check = used_prev_wv[used_wv_loci]
                        if np.isnan(other_wv_to_check):
                            continue
                        if  act_lon[other_wv_to_check, tm_i]<0 and act_lon[other_wv_to_check, tm_i-1]>0: #if crossing discontinuity
                            seam_dist = 180 - act_lon[other_wv_to_check, tm_i-1]
                            new_dist =  act_lon[other_wv_to_check, tm_i] +180
                            previous_dist = seam_dist+new_dist
                        else:
                            previous_dist = act_lon[other_wv_to_check, tm_i] - act_lon[other_wv_to_check, tm_i-1]
                        if np.abs(nearest_dist < previous_dist):
                            act_lon[other_wv_to_check, tm_i] = np.nan
                            act_avgval[other_wv_to_check, tm_i] = np.nan
                            used_prev_wv[used_wv_loci] = np.nan
                        else:
                            continue
                    #nearest_i is the element with a lon equal to the nearest lon...
                    nearest_avgval = data_valley[nearest_i]

                    
                    
                    if nearest_dist <= lon_limit:

                        if np.abs(nearest_avgval) < cont_threshold:
                            continue
                        ### Add to the list of used waves
                        used_prev_wv.append(prev_wv)
                        used_wv.append(nearest_i)

                        ### And of course, update the data
                        act_lon[prev_wv, tm_i] = nearest_lon
                        act_avgval[prev_wv, tm_i] = nearest_avgval

            #### Okay now that we are done, we now consider the waves we have left
            for wv in range(len(lon_valley)):
                #print(data_valley[wv])
                ### First, check the most recent
                if wv in used_wv: #If wave already exists, continue
                    continue
                else:
                    if np.abs(data_valley[wv]) < init_threshold: #If not strong enough to initiate...
                        continue
                    act_lon = np.vstack([act_lon, empty_stack])
                    act_avgval = np.vstack([act_avgval, empty_stack])

                    act_lon[-1, tm_i] = lon_valley[wv]
                    act_avgval[-1, tm_i] = data_valley[wv]
                ### Repeat for any crests remaining, now checking the timestep prior to this one 

                ### And finally, repeat a third time
    return act_lon, act_avgval  

def clean_up(act_lon, act_avgval, day_cut, t_res):
    step_len = int(day_cut*(24/t_res))
    new_act_lon=[]
    new_act_avgval=[]
    for row in range(act_lon.shape[0]):
        test_in = act_lon[row, :][~np.isnan(act_lon[row,:])]

        if len(test_in)>=step_len: #If we have enough for a CCKW, we will preserve it
            if len(new_act_lon) == 0:
                new_act_lon = act_lon[row, :]
                new_act_avgval = act_avgval[row, :]
            else:
                new_act_lon = np.vstack([new_act_lon, act_lon[row,:]])
                new_act_avgval = np.vstack([new_act_avgval, act_avgval[row,:]])
    return new_act_lon, new_act_avgval

def connect_tracks(AEW_lon_enter, AEW_avgval_enter):
    AEW_lon_in = AEW_lon_enter.copy()
    AEW_avgval_in = AEW_avgval_enter.copy()
    st_list = []
    end_list = []
    changed_wv = []
    for row in range(AEW_lon_in.shape[0]):
        st_idx = np.where(~np.isnan(AEW_lon_in[row,:]))[0][0]        
        end_idx = np.where(~np.isnan(AEW_lon_in[row,:]))[0][-1]
        st_list.append(st_idx)
        end_list.append(end_idx)
    for i in range(len(st_list)):
        if end_list[i] in st_list: 
            paired_wv = np.where(end_list[i] == st_list)[0][0]
            lon_diff = AEW_lon_in[paired_wv, end_list[i]] - AEW_lon_in[i, end_list[i]] # end_list[i] is the time for both. i is the time for wave we look at , paired_wv the compare one
            if np.abs(lon_diff) <= lon_limit and lon_diff>=-back_allow: #If this is truely a connecting wave case...
                averaged_point = np.mean([AEW_lon_in[paired_wv, end_list[i]],  AEW_lon_in[i, end_list[i]]])
                
                ### Change first wave
                AEW_lon_in[i, end_list[i]] = averaged_point
                AEW_lon_in[i, (end_list[i]+1):] = AEW_lon_in[paired_wv, (end_list[i]+1):] 
                
                ## Change second wave
                AEW_lon_in[paired_wv, end_list[i]] = averaged_point
                AEW_lon_in[paired_wv, :(end_list[i]+1)] = AEW_lon_in[i, :(end_list[i]+1)] 
                changed_wv.append([i, paired_wv])
    del_rows = []
    for i in range(len(changed_wv)):
        del_rows.append(changed_wv[i][0])
    AEW_lon_in = np.delete(AEW_lon_in, obj=del_rows, axis = 0)
    AEW_avgval_in = np.delete(AEW_avgval_in, obj=del_rows, axis = 0)
    return AEW_lon_in, AEW_avgval_in, changed_wv
                ### now we are going to modify input data so that we can cascade down changes if necessary

def best_fit_line(act_lon_in):
    out_arr = np.ones(np.shape(act_lon_in))*np.nan
    for sys in range(act_lon_in.shape[0]):
        sys_in_slc = act_lon_in[sys,:] #This is our slice
        ### But! We only want to pull out the non-nan ones
        sys_in_slc_non_nan = sys_in_slc[~np.isnan(sys_in_slc)]
        
        if sys_in_slc_non_nan[0]>0 and sys_in_slc_non_nan[-1]<0: #If we cross the discontinuity...
            ##### NOTE TO SELF: NANS GET INCLUDED ACROSS THE LINE. Will need to add special logic to account for this
            #print('Special...')
            sys_in_special = sys_in_slc_non_nan.copy()
            sys_in_special[sys_in_special<0] = sys_in_special[sys_in_special<0]+360
            result = stats.linregress(np.arange(len(sys_in_special)), sys_in_special)
            new_line = (np.arange(len(sys_in_special))*result.slope + result.intercept) 
            new_line[new_line>180] = new_line[new_line>180]-360
            #print(new_line)
            sys_in_final = sys_in_slc.copy()
            sys_in_final[~np.isnan(sys_in_slc)] = new_line
        else:
            result = stats.linregress(np.arange(len(sys_in_slc_non_nan)), sys_in_slc_non_nan)
            new_line = np.arange(len(sys_in_slc_non_nan))*result.slope + result.intercept

            sys_in_final = sys_in_slc.copy()
            sys_in_final[~np.isnan(sys_in_slc)] = new_line
        
        out_arr[sys,:] = sys_in_final
    return out_arr

# #### END OF FUNCTION DEFINITIONS

# #### START OF ACTUAL CODE

### DEFINITIONS
lat_bin = [-10, 10]
wave_type = 'Kelvin'
t_res = 24 #hours, resolution of dataset
if wave_type == 'Kelvin':
    speed_thresh = 30 #m/s
    day_cut = 3

#### Other definitions
lon_m = 111*1000 #m in a degree longitude at the equator
m_limit = t_res*3600*speed_thresh #meters a wave can possibly travel west
lon_limit = m_limit/lon_m
back_allow = 5 #How many degrees backwards we allow connections...

#### COMPILE FULL PATH
data_in = sys.argv[1]
var_in = sys.argv[2]

test_save_name = test_save_path+'CCKW_tracks_init_1_cont_0.25_'+data_in

### LOAD IN THE FULL DATASET
print('Loading in data...')
raw_data_xr = xr.open_dataset(data_in)
var_data_xr = xr.open_dataset(var_in)

data_in_xr = raw_data_xr['precipitation']/np.sqrt(var_data_xr['precipitation'])

data_in_xr.coords['lon'] = (data_in_xr.coords['lon'] + 180) % 360 - 180
data_in_xr = data_in_xr.sortby(data_in_xr.lon)

### Turn into a Hovmoller
print('Averaging...')
hov_data_in = data_in_xr.sel(lat = slice(lat_bin[0], lat_bin[-1])).mean(dim='lat')

### ADJUST INPUT DATA
data_in_xr = data_in_xr.rename({'lon':'longitude','lat':'latitude'})
hov_data_in = hov_data_in.rename({'lon':'longitude'})
hov_data_test = hov_data_in#.sel(time = slice('1980-01-01', '1980-12-31'))

longitude_arr = hov_data_test.longitude.values
longitude_arr = longitude_arr - 240

longitude_arr[longitude_arr < -180] = longitude_arr[longitude_arr < -180] + 360
hov_data_test['longitude'] = longitude_arr
hov_data_test = hov_data_test.sortby(hov_data_test.longitude)

### RUN CCKW IDENTIFICATION
print('Running initial tracker')
raw_act_lon, raw_act_avgval = run_CCKW_tracker(hov_data_test*-1, lon_limit, prominence = 0, cont_threshold = 0.25, init_threshold = 1)
print('connecting tracks')
int_act_lon, int_act_avgval, changed_wv_out = connect_tracks(raw_act_lon, raw_act_avgval)
print('cleaning up tracks')
final_act_lon, final_act_avgval = clean_up(int_act_lon, int_act_avgval, day_cut = day_cut, t_res = t_res)

if len(final_act_lon) == 0: #If we have no waves
    print('No CCKWS identified!')
    final_act_lon = np.zeros((1, np.shape(hov_data_test.values)[0]))*np.NaN
    final_act_avgval = final_act_lon.copy()
    print(final_act_lon)


new_act_lon = final_act_lon.copy()
new_act_lon[final_act_lon<=-60] = final_act_lon[final_act_lon<=-60] + 240
new_act_lon[final_act_lon>-60] = final_act_lon[final_act_lon>-60] - 120

final_act_lon = new_act_lon

if final_act_lon.ndim == 1: #Edge case, one only dimension!
    print('Edge case! Only one dimension. Reshaping')
    final_act_lon = final_act_lon.reshape(1, len(final_act_lon))
    final_act_avgval = final_act_avgval.reshape(1, len(final_act_avgval))
### TURN INTO AN XARRAY OBJECT
time_array = hov_data_test.time
system_array = np.arange(0, final_act_lon.shape[0])+1

CCKW_track_xr = xr.DataArray(final_act_lon, 
                             dims = ["system", "time"], 
                             coords = dict(
                                 system=(["system"], system_array),
                                 time=time_array),
                                 attrs=dict(
                                     description="Longitude of Tracked CCKWs",
                                     units="deg")
                                 )
CCKW_str_xr = xr.DataArray(final_act_avgval, 
                             dims = ["system", "time"], 
                             coords = dict(
                                 system=(["system"], system_array),
                                 time=time_array),
                                 attrs=dict(
                                     description="Strength of Tracked CCKWs",
                                     units="standard deviation")
                                 )

CCKW_combined_xr = xr.merge([CCKW_track_xr.to_dataset(name='CCKW_lon'), CCKW_str_xr.to_dataset(name='CCKW_str')])

### SAVE OUT
CCKW_combined_xr.to_netcdf(test_save_name)
