# python main_dual.py --gpu 0 --prefix dual --model mnist_mnist --batch_size 2048 --num_epochs 100
# python main_dual.py --gpu 0 --prefix dual --model usps_usps --batch_size 512 --num_epochs 50
# python main_dual.py --gpu 0 --prefix dual --model mnist_mnist --batch_size 4096 --num_epochs 50

# python main_div.py --gpu 0 --prefix div --model mnist_mnist --batch_size 2048 --num_epochs 200
# python main_div.py --gpu 1 --prefix div --model svhn_svhn --batch_size 1024 --num_epochs 200
# python main_div.py --gpu 0 --prefix div --model mnist_mnist  --batch_size 1024 --num_epochs 200 --dropout_probability 0.5
# python main_div3.py --gpu 0 --prefix div3 --model mnist_mnist  --batch_size 2048 --num_epochs 200

if [ -z "$1" ]
  then
    gpu="0"
  else
    gpu="${1}"
fi

postfix="--n_epochs 50 --start_epoch 3 --lr 0.03 --batch_size 1536"
alphabet_set="A B C D E G H" 
lr_set="0.01 0.06 0.09 0.1 0.3 0.6"
weight_in_loss_g_set="1,0.01,0.1,0.1" 
# cyc_loss_weight_set="0.01 0.05 0.001 0.005" 
cyc_loss_weight_set="0.01" 
# cla_plus_weight_set="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"
cla_plus_weight_set="0.3"

for alphabet in ${alphabet_set}
do
  for weight_in_loss_g in ${weight_in_loss_g_set}
  do
    for cyc_loss_weight in ${cyc_loss_weight_set}
    do
      for cla_plus_weight in ${cla_plus_weight_set}
      do  
        for lr in ${lr_set}
        do
          printprom="python main_bitranslation.py --prefix bitranslation_${alphabet} ${postfix} --lr ${lr} --model svhn_mnist --gpu ${gpu} --cyc_loss_weight ${cyc_loss_weight} --weight_in_loss_g ${weight_in_loss_g} --cla_plus_weight ${cla_plus_weight}"
          echo ${printprom}
          ${printprom}
        done
      done
    done
  done
done
