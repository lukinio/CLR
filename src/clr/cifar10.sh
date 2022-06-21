#/bin/bash

dataset=cifar10
data_dir="/shared/sets/datasets/vision/cifar10"
batch_size=256
optimizer=lars
max_epochs=800
learning_rate=( 1.5 )
coeffs=( 1 10 100 )

prefix=one_small_pass


OUTDIR=outputs/${dataset}/console/${prefix}
mkdir -p $OUTDIR

for reg_coeff in "${coeffs[@]}"
do
    for lr in "${learning_rate[@]}"
    do
          exp_name=${prefix}_coeff_${reg_coeff}
          python -u clr_pretrain.py --gpus 1 --dataset ${dataset} --data_dir ${data_dir} --batch_size ${batch_size} --optimizer ${optimizer} \
                                    --learning_rate ${lr} --exclude_bn_bias --max_epochs ${max_epochs} --exp_name ${exp_name} \
                                    --reg_coeff ${reg_coeff} --mode both --online_ft | tee ${OUTDIR}/${lr}_${reg_coeff}.log
    done
done


