
function cifar10() {

  python l0_vat_sup_inductive.py --trainer=l0at \
      --data-dir=dataset \
      --gpu-id=2 \
      --alpha=0.5 \
      --dataset=cifar10  \
      --num-epochs=300 \
      --aug-trans \
      --aug-flip
}

function svhn() {

  python l0_vat_sup_inductive.py --trainer=l0at \
      --data-dir=dataset \
      --gpu-id=2 \
      --alpha=0.5 \
      --dataset=svhn  \
      --num-epochs=120 \
      --aug-flip
}


