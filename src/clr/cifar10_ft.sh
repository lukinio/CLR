#/bin/bash

dataset=cifar10
data_dir="/shared/sets/datasets/vision/cifar10"
batch_size=128
max_epochs=100
learning_rate=( 0.001 )
optimizer=adam
grad_clip=1
memory_lengths=( 2048 )
#exp_names=( "one_small_coeff" "both_small_coeff" "both_big_coeff" "one_big_coeff" )
exp_names=( "one_small_memory" )
coeffs=( 1 )

OUTDIR=outputs/${dataset}/console/finetune
mkdir -p $OUTDIR



reg_coeff=1
pre_train_lr=0.1
for exp_name in "${exp_names[@]}"
do
    for memory_length in "${memory_lengths[@]}"
    do
        for lr in "${learning_rate[@]}"
        do
            full_exp_name=${exp_name}_${memory_length}_coeff_${reg_coeff}_lr_${pre_train_lr}
#            full_exp_name=one_small_coeff_10
            echo ${full_exp_name}
            python -u clr_finetune.py --gpus 1 --dataset ${dataset} --data_dir ${data_dir} --batch_size ${batch_size} \
                                      --learning_rate ${lr} --num_epochs ${max_epochs} --exp_name ${full_exp_name} \
                                      --reg_coeff ${reg_coeff} --memory_length 0 --optimizer ${optimizer} \
                                      | tee ${OUTDIR}/${lr}_${reg_coeff}.log
        done
    done
done


#for exp_name in "${exp_names[@]}"
#do
#    for lr in "${learning_rate[@]}"
#    do
#          for reg_coeff in "${coeffs[@]}"
#          do
#              full_exp_name=${exp_name}_${reg_coeff}
#              python -u clr_finetune.py --gpus 1 --dataset ${dataset} --data_dir ${data_dir} --batch_size ${batch_size} \
#                                        --learning_rate ${lr} --num_epochs ${max_epochs} --exp_name ${full_exp_name} \
#                                        --reg_coeff ${reg_coeff} --memory_length 0 | tee ${OUTDIR}/${lr}_${reg_coeff}.log
#          done
#    done
#done
