#!/bin/bash

if [ -z "$1" ]
  then
    gpu="0"
  else
    gpu="${1}"
fi

postfix="--n_epochs 200 --batch_size 3072"
if [ ${gpu} -eq "0" ]
then
  lr_set="0.01 0.05 0.03 0.5 0.1 0.005 0.001"
  lr2_set="0.01 0.05 0.03 0.5 0.1 0.005 0.001"
  weight_decay_set="0.0005"
  start_epoch_set="1"
  start_acc_set="10"
  for lr in ${lr_set}
  do
    for lr2 in ${lr2_set}
    do
      for weight_decay in ${weight_decay_set}
      do
        for start_acc in ${start_acc_set}
        do
          python main3_entropy.py --prefix "adEntPlus" --gpu ${gpu} --start_acc ${start_acc} --weight_decay ${weight_decay} --lr ${lr} --lr2 ${lr2} ${postfix}
        done
      done
    done
  done
elif [ ${gpu} -eq "3" ]
then
  lr_set="0.01"
  lr2_set="0.01"
  weight_decay_set="0.0005"
  start_acc_set="60 70 80"
  for lr in ${lr_set}
  do
    for lr2 in ${lr2_set}
    do
      for weight_decay in ${weight_decay_set}
      do
        for start_acc in ${start_acc_set}
        do
          python main3_entropy.py --prefix "adEntPlus" --gpu ${gpu} --start_acc ${start_acc} --weight_decay ${weight_decay} --lr ${lr} --lr2 ${lr2} ${postfix}
        done
      done
    done
  done
  
  weight_decay_set="0.0001 0.001 0.005 0.01 0.05 0.1 0.5"
  start_acc_set="50 60 70 80"
  for lr in ${lr_set}
  do
    for lr2 in ${lr2_set}
    do
      for weight_decay in ${weight_decay_set}
      do
        for start_acc in ${start_acc_set}
        do
          python main3_entropy.py --prefix "adEntPlus" --gpu ${gpu} --start_acc ${start_acc} --weight_decay ${weight_decay} --lr ${lr} --lr2 ${lr2} ${postfix}
        done
      done
    done
  done
fi

