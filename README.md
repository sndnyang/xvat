# xAT/xVAT

This reposityory contains the PyTorch implementation for [Learning with Multiplicative Perturbations](https://arxiv.org/abs/1912.01810)



## Demo

The evolutions of decision boundaries of the learned networks on synthetic "Moons" dataset:





## Requirements

    PyTorch >= 0.4.0
    tensorboardX <= 1.6.0
    numpy


​    
# Download Datasets

1. git clone https://github.com/sndnyang/L0VAT 

2. Download the Datasets(MNIST, SVHN, CIFAR10):
   
   On CIFAR-10
   
   ```
   python dataset/cifar10.py --data_dir=./data/cifar10/
   ```
   
   On SVHN
   
   ```
   python dataset/svhn.py --data_dir=./data/svhn/
   ```
# Usage

## Supervised Learning

### mnist:

python l0_vat_sup_inductive.py --trainer=at --dataset=mnist --arch=MLPSup --data-dir=data --vis --lr=0.001 --lr-decay=0.95 --lr-a=0.000001 --epoch-decay-start=100 --num-epochs=100 --lamb=1  --alpha=1 --k=1 --layer=1 --batch-size=100 --num-batch-it=500 --eps=2 --debug --log-arg=trainer-data_dir-arch-lr-lr_a-eps-lamb-top_bn-layer-debug-seed-fig --seed=1 --gpu-id=4 --alpha=0.5



## Semi-Supervised Learning


### transductive way

    python l0_vat_semi.py

### inductive way

    python l0_vat_semi_inductive.py



## Citation

If you found this code useful, please cite our paper.

```latex
@article{xvat2019,
	title={Learning with Multiplicative Perturbations},
	author={Xiulong Yang and Shihao Ji},
	journal={arXiv preprint arXiv:1912.01810},
	year={2019}
}
```

