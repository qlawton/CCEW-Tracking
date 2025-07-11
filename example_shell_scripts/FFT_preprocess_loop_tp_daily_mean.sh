#!/bin/bash
module load cdo
module load nco
module load conda
conda activate my-npl-ml
model_cat=mpas
model_in=$1
just_obs='False'
t_res=24

DIR=/glade/derecho/scratch/qlawton/mpas_runs/CCKW_MODELS/${model_in}/
model_name_in=${model_cat}
echo $model_name_in ${model_in}
file=${DIR}post_process/merged_deg_1_daily_mean_rainrate_for_FFT.nc
python ~/SCRIPTS/Model_CCEW_Skill/scripts/preprocess_FFT_tp_daily_mean_MPAS.py $file ${model_name_in} $t_res $just_obs


