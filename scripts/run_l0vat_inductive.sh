#!/usr/bin/env bash

# l0 -> L0, not 10 ten

function cifar10() {
    python l0_vat_semi_inductive.py \
      --gpu-id=7 --num-epochs=500 \
      --epoch-decay-start=450     \
      --dataset=cifar10 \
      --data-dir=dataset \
      --trainer=inl0  \
      --arch=CNN9  \
      --max-lamb=0.1  \
      --layer=3       \
      --zeta=1.1      \
      --gamma=-0.1    \
      --beta=0.66     \
      --lamb=3    \
      --lamb2=1   \
      --lr-a=0.00001      \
      --lr=0.003          \
      --debug           \
      --vis --log-dir=

}

function svhn() {
    python l0_vat_semi_inductive.py \
      --gpu-id=7 --num-epochs=120 \
      --epoch-decay-start=80     \
      --dataset=svhn  \
      --data-dir=dataset \
      --trainer=inl0  \
      --arch=CNN9  \
      --max-lamb=0.1  \
      --layer=3       \
      --zeta=1.1      \
      --gamma=-0.1    \
      --beta=0.66     \
      --lamb=3    \
      --lamb2=1   \
      --lr-a=0.00001      \
      --lr=0.003          \
      --debug           \
      --vis --log-dir=

}

cifar10
# --two-stage: default use one stage way
