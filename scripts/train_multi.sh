#!/bin/bash

nohup python scripts/train.py configs/base.yaml --where modal \
    --update configs/regression/base.yaml \
    --update configs/regression/modal.yaml \
    --update configs/regression/mse-pixel-lpips.yaml \
    > logs/mse-pixel-lpips.log 2>&1 &

nohup python scripts/train.py configs/base.yaml --where modal \
    --update configs/regression/base.yaml \
    --update configs/regression/modal.yaml \
    --update configs/regression/mse-pixel-lpips.yaml \
    --update configs/optim/accum-4.yaml \
    > logs/mse-pixel-lpips-accum4.log 2>&1 &

nohup python scripts/train.py configs/base.yaml --where modal \
    --update configs/regression/base.yaml \
    --update configs/regression/modal.yaml \
    --update configs/regression/mse-pixel-lpips.yaml \
    --update configs/optim/cosine.yaml \
    > logs/mse-pixel-lpips-cosine.log 2>&1 &
