#/bin/bash

dataset=imagenet
data_dir="/shared/sets/datasets/vision/ImageNet"
batch_size=200
optimizer=lars
max_epochs=800
learning_rate=( 4.8 )
coeffs=( 1 )
model_weights="../simclr_imagenet.ckpt"

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
                                    --reg_coeff ${reg_coeff} --mode both --online_ft \
                                    --model_weights ${model_weights} | tee ${OUTDIR}/${lr}_${reg_coeff}.log
    done
done


