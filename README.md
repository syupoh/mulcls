# Multi Classifier

python dual_dropout.py --gpu 0 --model svhn_svhn --batch_size 512 --num_epochs 100
python dual_dropout.py --gpu 0 --model usps_usps --batch_size 512 --num_epochs 50
python dual_dropout.py --gpu 0 --model mnist_mnist --batch_size 4096 --num_epochs 50