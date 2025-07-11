#!/bin/bash
module load cdo
module load nco
module load conda
conda activate my-npl-ml

DIR=/glade/derecho/scratch/qlawton/model_data_for_filter_daily_mean/TP/
t_res=24
data_var="precipitation"
model_name=mpas
model_in=$1
cd $DIR
for file in *${model_name}_${model_in}*.nc; do
    echo $file
    python ~/SCRIPTS/Model_CCEW_Skill/scripts/FFT_filter_daily_mean.py $file $t_res $DIR $data_var
    #python ~/SCRIPTS/Model_CCEW_Skill/scripts/preprocess_FFT.py $file ${model_name_in} $t_res $just_obs
done
    ### Now combine them and rename time



