#!/bin/bash
module load cdo
module load nco
ml conda
conda activate my-npl-ml

model_name="MPAS"
model_in=$1
DIR=/glade/derecho/scratch/qlawton/model_data_for_filter_daily_mean/TP/FILTERED/
t_res=24
ref_file=/glade/work/qlawton/DATA/IMERG/DAILY_FILTERED_WITH_LATE/VAR/daily_mean_var-daily-Kelvin.nc
cd $DIR
for file in Kelvin*"${model_name,,}"_${model_in}*.nc; do
    echo $file
    python ~/SCRIPTS/Model_CCEW_Skill/scripts/track_CCKW_save_adjust_seam_tp_daily_mean.py $file $ref_file $model_name
    #python ~/SCRIPTS/Model_CCEW_Skill/scripts/preprocess_FFT.py $file ${model_name_in} $t_res $just_obs
done
    ### Now combine them and rename time



