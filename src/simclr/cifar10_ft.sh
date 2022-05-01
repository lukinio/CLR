#/bin/bash

dataset=cifar10
data_dir="/shared/sets/datasets/vision/cifar10"
batch_size=64
max_epochs=100
learning_rate=( 0.3 )
exp_names=( "simclr_big_batch_1024" "simclr_small_batch_256" )



OUTDIR=outputs/${dataset}/finetune/console
mkdir -p $OUTDIR



for exp_name in "${exp_names[@]}"
do
    for lr in "${learning_rate[@]}"
    do
        python -u simclr_finetune.py --gpus 1 --dataset ${dataset} --data_dir ${data_dir} --batch_size ${batch_size} \
                                     --learning_rate ${lr} --num_epochs ${max_epochs} --exp_name ${exp_name} \
                                     | tee ${OUTDIR}/${lr}_${batch_size}.log

    done
done