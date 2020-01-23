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
    gpu=${1}
fi

if [ ${gpu} = 0 ]
then
    # loss4rate_set="1e-4 5e-4 1e-5 5e-5 1e-6 5e-6" 
    # # loss4rate_set="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.001 0.05 2 5 10 20 30" 
    # for loss4rate in ${loss4rate_set}
    # do
    #     printprom="python main_div3.py --gpu ${gpu} --prefix div3 --model mnist_mnist --batch_size 2048 --num_epochs 200 --loss4_KLD_dis_rate ${loss4rate}"
    #     echo ${printprom}
    #     ${printprom}
    # done
    beta_set="5 7 10 15" 
    mu_set="5 7 10 15" 
    for beta in ${beta_set}
    do
      for mu in ${mu_set}
      do
        printprom="python main_translation.py --model mnist_mnist --beta ${beta} --mu ${mu} --batch_size 6 --gpu ${gpu} --prefix translation"
        echo ${printprom}
        ${printprom}
      done
    done
elif [ ${gpu} = 1 ]
then
    # model_set="svhn_svhn"
    # loss4rate_set="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.001 0.05 2 5 10 20 30" 
    # dropout_set="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"
    # for model in ${model_set}
    # do
    #     for loss4rate in ${loss4rate_set}
    #     do
    #         printprom="python main_div3.py --gpu ${gpu} --prefix div3 --model ${model} --batch_size 256 --num_epochs 200 --loss4_KLD_dis_rate ${loss4rate}"
    #         echo ${printprom}
    #         ${printprom}
    #     done
    # done
    printprom="python main_translation.py --model svhn_svhn --batch_size 4 --gpu ${gpu} --prefix translation_noCE "
    echo ${printprom}
    ${printprom}
elif [ ${gpu} = 2 ]
then
    # model_set="usps_usps"
    # loss4rate_set="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.001 0.05 2 5 10 20 30" 
    # dropout_set="0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"
    # for model in ${model_set}
    # do
    #     for loss4rate in ${loss4rate_set}
    #     do
    #         printprom="python main_div3.py --gpu ${gpu} --prefix div3 --model ${model} --batch_size 256 --num_epochs 200 --loss4_KLD_dis_rate ${loss4rate}"
    #         echo ${printprom}
    #         ${printprom}
    #     done
    # done
    beta_set="5 7 10 15" 
    mu_set="5 7 10 15" 
    for beta in ${beta_set}
    do
      for mu in ${mu_set}
      do
        printprom="python main_translation.py --batch_size 4 --model svhn_svhn --beta ${beta} --mu ${mu} --batch_size 4 --gpu ${gpu} --prefix translation --norm True"
        echo ${printprom}
        ${printprom}
      done
    done
fi

