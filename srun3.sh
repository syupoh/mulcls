#!/bin/bash

if [ -z "$1" ]
  then
    gpu="0"
  else
    gpu="${1}"
fi

postfix="--n_epochs 200 --batch_size 4096"
if [ ${gpu} -eq "0" ]
then
  lr_set="0.01"
  lr2_set="0.01"
  weight_decay_set="0.0005"
  start_acc_set="50 50 50 50"
  start_acc_set2="50 50 50 50"
  # prefix_set="adEntPlus_2cls_F adEntPlus_2cls_G adEntPlus_2cls_K adEntPlus_2cls_L adEntPlus_2cls_M adEntPlus_2cls_A adEntPlus_2cls_B adEntPlus_2cls_C adEntPlus_2cls_H adEntPlus_2cls_I adEntPlus_2cls_J adEntPlus_2cls_N"
  prefix_set="adEntPlus_2cls_D adEntPlus_2cls_E"
  postfix="--seedfix ""False"" --n_epochs 150 --batch_size 4096"
  for lr in ${lr_set}
  do
    for lr2 in ${lr2_set}
    do
      for weight_decay in ${weight_decay_set}
      do
        for start_acc in ${start_acc_set}
        do
          for start_acc2 in ${start_acc_set2}
          do
            for prefix in ${prefix_set}
            do
              python main3_entropy.py --prefix ${prefix} --gpu ${gpu} --start_acc2 ${start_acc2} --start_acc ${start_acc} --weight_decay ${weight_decay} --lr ${lr} --lr2 ${lr2} ${postfix}
            done
          done
        done
      done
    done
  done
elif [ ${gpu} -eq "3" ]
then
  lr_set="0.01"
  lr2_set="0.01"
  weight_decay_set="0.0005"
  start_acc_set="50 50 50 50"
  start_acc_set2="50 50 50 50"
  prefix_set="adEntPlus_2clsPlus_F adEntPlus_2clsPlus_G adEntPlus_2clsPlus_K adEntPlus_2clsPlus_L adEntPlus_2clsPlus_M adEntPlus_2clsPlus_A adEntPlus_2clsPlus_B adEntPlus_2clsPlus_C adEntPlus_2clsPlus_H adEntPlus_2clsPlus_I adEntPlus_2clsPlus_J adEntPlus_2clsPlus_N"
  # prefix_set="adEntPlus_2clsPlus_D adEntPlus_2clsPlus_E"
  postfix="--seedfix ""False"" --n_epochs 150 --batch_size 4096"
  for lr in ${lr_set}
  do
    for lr2 in ${lr2_set}
    do
      for weight_decay in ${weight_decay_set}
      do
        for start_acc in ${start_acc_set}
        do
          for start_acc2 in ${start_acc_set2}
          do
            for prefix in ${prefix_set}
            do
              python main3_entropy.py --prefix ${prefix} --gpu ${gpu} --start_acc2 ${start_acc2} --start_acc ${start_acc} --weight_decay ${weight_decay} --lr ${lr} --lr2 ${lr2} ${postfix}
            done
          done
        done
      done
    done
  done
fi