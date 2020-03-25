
function cifar10_xat() {

  python l0_vat_sup_inductive.py --trainer=l0at \
      --data-dir=dataset \
      --lr=0.003 \
      --gpu-id=2 \
      --alpha=0.5 \
      --dataset=cifar10  \
      --num-epochs=300 \
      --layer=1 \
      --lr-a=0.000001 \
      --epoch-decay-start 200 \
      --aug-trans \
      --aug-flip  \
      --log-dir=./runs
}

function cifar10_xvat() {

  # inl0-data-lr=0.003-lr_a=1e-06-eps=1.0-lamb=3.0-top_bn=True-1-aug_trans=True
  python l0_vat_sup_inductive.py --trainer=inl0 \
      --data-dir=dataset \
      --gpu-id=2 \
      --lr=0.003 \
      --lamb=3.0 \
      --dataset=cifar10  \
      --num-epochs=300 \
      --layer=1 \
      --lr-a=0.000001 \
      --top-bn    \
      --aug-trans \
      --aug-flip \
      --epoch-decay-start 200\
      --log-dir=./runs
}

function svhn() {

  python l0_vat_sup_inductive.py --trainer=l0at \
      --data-dir=dataset \
      --gpu-id=2 \
      --alpha=0.5 \
      --dataset=svhn  \
      --epoch-decay-start=180     \
      --num-epochs=120 \
      --layer=1 \
      --lr-a=0.000001 \
      --epoch-decay-start 80 \
      --aug-flip\
      --log-dir=./runs
}

cifar10_xvat
