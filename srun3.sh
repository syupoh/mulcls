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
  start_acc_set2="60 70 80"
  # prefix_set="adEntPlus_2clscossim_F adEntPlus_2clscossim_G adEntPlus_2clscossim_K adEntPlus_2clscossim_L adEntPlus_2clscossim_M adEntPlus_2clscossim_A adEntPlus_2clscossim_B adEntPlus_2clscossim_C adEntPlus_2clscossim_H adEntPlus_2clscossim_I adEntPlus_2clscossim_J adEntPlus_2clscossim_N"
  # prefix_set="adEntPlus_2clscossim_D adEntPlus_2clscossim_E adEntPlus_2clscossim_O"
  prefix_set="adEntPlus_2clscossim"
  postfix="--n_epochs 150 --batch_size 4096" # automatically seedfix false
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
              prompt="python main3_entropy.py --prefix ${prefix} --gpu ${gpu} --start_acc2 ${start_acc2} --start_acc ${start_acc} --weight_decay ${weight_decay} --lr ${lr} --lr2 ${lr2} ${postfix}"
              echo ${prompt}
              ${prompt}
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
  start_acc_set2="-1 10 20 30 40 50"
  # prefix_set="adEntPlus_2clscossim_F adEntPlus_2clscossim_G adEntPlus_2clscossim_K adEntPlus_2clscossim_L adEntPlus_2clscossim_M adEntPlus_2clscossim_A adEntPlus_2clscossim_B adEntPlus_2clscossim_C adEntPlus_2clscossim_H adEntPlus_2clscossim_I adEntPlus_2clscossim_J adEntPlus_2clscossim_N"
  # prefix_set="adEntPlus_2clscossim_D adEntPlus_2clscossim_E adEntPlus_2clscossim_O"
  prefix_set="adEntPlus_2clscossim"
  postfix="--n_epochs 150 --batch_size 4096" # automatically seedfix false
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
              prompt="python main3_entropy.py --prefix ${prefix} --gpu ${gpu} --start_acc2 ${start_acc2} --start_acc ${start_acc} --weight_decay ${weight_decay} --lr ${lr} --lr2 ${lr2} ${postfix}"
              echo ${prompt}
              ${prompt}
            done
          done
        done
      done
    done
  done
fi