#!/bin/bash

if [ -z "$1" ]
  then
    gpu="0"
  else
    gpu="${1}"
fi

alphabet_set="A" 
lr_set="0.003 0 0.001"
lr2_set="0.003 0.001"
weight_decay_set="0.00005 0.00001"
start_epoch_set="3 0 1 2"
postfix="--n_epochs 50 --batch_size 3072 --weight_decay 5e-4"

# 200320
# lr_set="0.003 0.001 0.01 0.03 0.06 0.09 0.1 0.3 0.6"
# for lr in ${lr_set}
# do
#   python main3_entropy.py --prefix adEntMinus --batch_size 3000 --gpu ${gpu} --lr ${lr}
# done

if [ ${gpu} -eq "0" ]
then
  for lr in ${lr_set}
  do
    for lr2 in ${lr2_set}
      for weight_decay in ${weight_decay_set}
        for start_epoch in ${start_epoch_set}
          python main3_entropy.py --prefix adEntPlus ${postfix} --gpu ${gpu} --start_epoch ${start_epoch} --weight_decay ${weight_decay} --lr ${lr} --lr2 ${lr2}
        done
      done
    done
  done
elif [ ${gpu} -eq "3" ]
  for lr in ${lr_set}
  do
    for lr2 in ${lr2_set}
      for weight_decay in ${weight_decay_set}
        for start_epoch in ${start_epoch_set}
          python main3_entropy.py --prefix adEntPlus ${postfix} --gpu ${gpu} --start_epoch ${start_epoch} --weight_decay ${weight_decay} --lr ${lr} --lr2 ${lr2}
        done
      done
    done
  done
fi

