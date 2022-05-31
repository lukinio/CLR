#/bin/bash

dataset=imagenet
data_dir="/shared/sets/datasets/vision/ImageNet"
batch_size=256
optimizer=lars
max_epochs=800
learning_rate=( 4.8 )

prefix=simclr_imagenet_batch

OUTDIR=outputs/${dataset}/${prefix}/console
mkdir -p $OUTDIR



for lr in "${learning_rate[@]}"
do
    exp_name=${prefix}_${batch_size}
    python -u simclr_pretrain.py --gpus 1 --dataset ${dataset} --data_dir ${data_dir} --batch_size ${batch_size} --optimizer ${optimizer} \
                                 --learning_rate ${lr} --exclude_bn_bias --max_epochs ${max_epochs} --exp_name ${exp_name} \
                                 --online_ft --devices 1 | tee ${OUTDIR}/${lr}_${batch_size}.log
done

