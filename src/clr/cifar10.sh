#/bin/bash

dataset=cifar10
data_dir="/shared/sets/datasets/vision/cifar10"
batch_size=256
memory_lengths=( 128 256 512 1024 2048 )
memory_lengths=( 2048 1024 512 256 128 )
optimizer=lars
grad_clip=1
max_epochs=800
learning_rate=( 0.1 )
coeffs=( 1 )

prefix=one_small_memory


OUTDIR=outputs/${dataset}/console/${prefix}
mkdir -p $OUTDIR

for reg_coeff in "${coeffs[@]}"
do
    for memory_length in "${memory_lengths[@]}"
    do
        for lr in "${learning_rate[@]}"
        do
              exp_name=${prefix}_${memory_length}_coeff_${reg_coeff}_lr_${lr}
              python -u clr_pretrain.py --gpus 1 --dataset ${dataset} --data_dir ${data_dir} --batch_size ${batch_size} --optimizer ${optimizer} \
                                        --learning_rate ${lr} --exclude_bn_bias --max_epochs ${max_epochs} --exp_name ${exp_name} \
                                        --reg_coeff ${reg_coeff} --memory_length ${memory_length} --mode one --online_ft \
                                        --grad_clip 1 | tee ${OUTDIR}/${lr}_${reg_coeff}.log
        done
    done
done


