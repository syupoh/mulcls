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
          python main3_entropy.py --prefix "adEntPlus_lrvariation" --gpu ${gpu} --start_acc ${start_acc} --weight_decay ${weight_decay} --lr ${lr} --lr2 ${lr2} ${postfix}
        done
      done
    done
  done
elif [ ${gpu} -eq "3" ]
then
  lr_set="0.01"
  lr2_set="0.01"
  weight_decay_set="0.0005"
  
  start_acc_set="70"
  start_acc_set2="10 20 30 40 50 60 70 80"
  postfix="--seedfix ""True"" --start_epoch 0 --n_epochs 150 --batch_size 4096"
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
            python main3_entropy.py --prefix "adEntPlus_2cls" --gpu ${gpu} --start_acc2 ${start_acc2} --start_acc ${start_acc} --weight_decay ${weight_decay} --lr ${lr} --lr2 ${lr2} ${postfix}
          done
        done
      done
    done
  done
  
  start_acc_set="80 90"
  start_acc_set2="0 10 20 30 40 50 60 70 80"
  postfix="--seedfix ""True"" --start_epoch 0 --n_epochs 150 --batch_size 4096"
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
            python main3_entropy.py --prefix "adEntPlus_2cls" --gpu ${gpu} --start_acc2 ${start_acc2} --start_acc ${start_acc} --weight_decay ${weight_decay} --lr ${lr} --lr2 ${lr2} ${postfix}
          done
        done
      done
    done
  done

fi