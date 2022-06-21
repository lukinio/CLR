#/bin/bash

dataset=cifar10
data_dir="/shared/sets/datasets/vision/cifar10"
batch_size=128
max_epochs=100
learning_rate=( 0.1 0.3 )
exp_names=( "one_small_pass" )
coeffs=( 10 )

OUTDIR=outputs/${dataset}/console/finetune
mkdir -p $OUTDIR


for exp_name in "${exp_names[@]}"
do
    for lr in "${learning_rate[@]}"
    do
          for reg_coeff in "${coeffs[@]}"
          do
              full_exp_name=${exp_name}_coeff_${reg_coeff}
              python -u clr_finetune.py --gpus 1 --dataset ${dataset} --data_dir ${data_dir} --batch_size ${batch_size} \
                                        --learning_rate ${lr} --num_epochs ${max_epochs} --exp_name ${full_exp_name} \
                                        --reg_coeff ${reg_coeff} | tee ${OUTDIR}/${lr}_${reg_coeff}.log
          done
    done
done
