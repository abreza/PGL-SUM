#!/bin/bash

# Variable Declaration
n_epochs=200
split_indexes=(0 1 2 3 4)
pretrained_models=("pretrain-epoch-10.pkl" "pretrain-epoch-20.pkl" "pretrain-epoch-40.pkl")
pretrain_path="/media/external_10TB/10TB/p_haghighi/saved_models/pretrain-summarization-models-on-our-dataset"

# Function for training
train_model () {
    local video_type=$1
    local batch_size=$2
    local n_pretrain_epochs=$3
    local eval_method=$4

    for split_index in "${split_indexes[@]}"; do
        save_dir="../PGL-SUM/Summaries/from_scratch"
        python model/main.py --split_index $split_index --n_epochs $n_epochs --batch_size $batch_size --video_type $video_type --zero_shot --mode train --save_dir $save_dir
        python evaluation/compute_fscores.py --path ${save_dir}/${video_type}/results/split${split_index} --dataset ${video_type} --eval ${eval_method}


        if [ "$n_pretrain_epochs" != "" ]; then
            python model/main.py --split_index $split_index --n_epochs $n_epochs --batch_size $batch_size --video_type $video_type --zero_shot --n_pretrain_epochs $n_pretrain_epochs --mode train --save_dir $save_dir
        fi

        for model in ${pretrained_models[@]}; do
            pretrain_model_path=$pretrain_path/$model
            save_dir="../PGL-SUM/Summaries/from_pretrain_${model:15:-4}"
            python model/main.py --split_index $split_index --n_epochs $n_epochs --batch_size $batch_size --video_type $video_type --zero_shot --from_pretrain $pretrain_model_path --mode train --save_dir $save_dir
            python evaluation/compute_fscores.py --path ${save_dir}/${video_type}/results/split${split_index} --dataset ${video_type} --eval ${eval_method}
        done
    done
}

# SumMe
train_model 'SumMe' 20 100 'max'

# TVSum
train_model 'TVSum' 40 100 'avg'
