
if [ -z "$1" ]
  then
    gpu="0"
  else
    gpu="${1}"
fi

postfix="--n_epochs 50 --batch_size 3072"
alphabet_set="A" 
lr_set="0.003 0.001 0.03 0.06 0.09 0.1 0.3 0.6"
start_epoch_set="3 0 1 2"

for lr in ${lr_set}
do
  for start_epoch in ${start_epoch_set}
  do
    for alphabet in ${alphabet_set}
    do
      printprom="python main_bitranslation_2.py --model svhn_mnist --lr ${lr} --prefix vat_entropy_${alphabet} ${postfix} --gpu ${gpu} --start_epoch ${start_epoch}"
      echo ${printprom}
      ${printprom}
    done
  done
done

lr_set="0.003"
start_epoch_set="0 1 2"

for lr in ${lr_set}
do
  for start_epoch in ${start_epoch_set}
  do
    for alphabet in ${alphabet_set}
    do
      printprom="python main_bitranslation_2.py --model svhn_mnist --lr ${lr} --prefix vat_entropy_${alphabet} ${postfix} --gpu ${gpu} --start_epoch ${start_epoch}"
      echo ${printprom}
      ${printprom}
    done
  done
done

lr_set="0.001 0.03 0.06 0.09 0.1 0.3 0.6"
start_epoch_set="3 0 1 2"

for lr in ${lr_set}
do
  for start_epoch in ${start_epoch_set}
  do
    for alphabet in ${alphabet_set}
    do
      printprom="python main_bitranslation_2.py --model svhn_mnist --lr ${lr} --prefix vat_entropy_${alphabet} ${postfix} --gpu ${gpu} --start_epoch ${start_epoch}"
      echo ${printprom}
      ${printprom}
    done
  done
done