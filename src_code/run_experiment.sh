#! /bin/bash

source activate mytorch
date=`date +"%Y%m%d"`


dataset=${1:-mhealth_timestep_same_subject_random_view}
sample=${2:-1}
window_sz=${3:-25}
nb_views=${4:-3}

ar=5
max_epochs=100
batch_sz=256 
lr=1e-3
latent_size=20
rnn_hidden_size=200
emission_hidden_size=100
transition_hidden_size=100


data_dir="../data_preparation/preprocessed_datasets/$dataset/sample$sample/anomaly_rate_${ar}_views_${nb_views}"
echo $data_dir
output_dir="./experiments/date_$date/$dataset-$sample-$ar"
mkdir -p $output_dir
python3 main.py \
    --data_dir=$data_dir \
    --output_dir=$output_dir \
    --anomaly_rate=$ar \
    --window_size=$window_sz \
    --max_epochs=$max_epochs \
    --batch_sz=$batch_sz \
    --lr=$lr \
    --latent_size=$latent_size \
    --rnn_hidden_size=$rnn_hidden_size \
    --emission_hidden_size=$emission_hidden_size \
    --transition_hidden_size=$transition_hidden_size