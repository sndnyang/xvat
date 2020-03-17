#!/usr/bin/env bash

# l0 -> L0, not 10 ten

python l0_vat_semi_trans.py \
  --gpu-id=7 --num-epochs=100 \
  --dataset=mnist \
  --trainer=l0  \
  --arch=MLPSemi  \
  --zeta=1.1      \
  --gamma=-0.1    \
  --beta=0.66     \
  --size=100      \
  --lamb=1    \
  --kl=1      \
  --k=1       \
  --lamb2=0   \
  --lr=0.001  \
  --lr-a=0.001 \
  --num-batch-it=500  \
  --lr-decay=0.95     \
  --batch-size=100    \
  --ul-batch-size=250 \
  --debug             \
  --vis --log-dir=
