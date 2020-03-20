
if [ -z "$1" ]
  then
    gpu="0"
  else
    gpu="${1}"
fi

postfix="--n_epochs 50 --batch_size 3072"
alphabet_set="A" 
lr_set="0.003 0.001 0.01 0.03 0.06 0.09 0.1 0.3 0.6"
start_epoch_set="3 0 1 2"

# 200320
# lr_set="0.003 0.001 0.01 0.03 0.06 0.09 0.1 0.3 0.6"
# for lr in ${lr_set}
# do
#   python main_bitranslation_3.py --gpu ${gpu} --batch_size 3000 --lr ${lr}
# done


for lr in ${lr_set}
do
  python main_bitranslation_3.py --gpu ${gpu} --batch_size 3000 --lr ${lr}
done

