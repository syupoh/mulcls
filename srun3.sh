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
  # prefix_set="adEntPlus_2clsF adEntPlus_2clsG adEntPlus_2clsK adEntPlus_2clsL adEntPlus_2clsM adEntPlus_2clsA adEntPlus_2clsB adEntPlus_2clsC adEntPlus_2clsH adEntPlus_2clsI adEntPlus_2clsJ adEntPlus_2clsN"
  prefix_set="adEntPlus_2clsD adEntPlus_2clsE"
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
  # prefix_set="adEntPlus_2clsF adEntPlus_2clsG adEntPlus_2clsK adEntPlus_2clsL adEntPlus_2clsM adEntPlus_2clsA adEntPlus_2clsB adEntPlus_2clsC adEntPlus_2clsH adEntPlus_2clsI adEntPlus_2clsJ adEntPlus_2clsN"
  prefix_set="adEntPlus_2clsD adEntPlus_2clsE"
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