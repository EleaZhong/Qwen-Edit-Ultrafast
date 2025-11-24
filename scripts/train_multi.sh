#!/bin/bash

# nohup python scripts/train.py configs/base.yaml --where modal \
#     --update configs/regression/base.yaml \
#     --update configs/regression/modal-datadirs.yaml \
#     --update configs/regression/mse.yaml \
#     --update configs/regression/val_metrics.yaml \
#     --update configs/compare/5k_steps.yaml \
#     --update configs/optim/cosine.yaml \
#     --update configs/regression/lo_mse.yaml \
#     > logs/mse.log 2>&1 &

nohup python scripts/train.py configs/base.yaml --where modal \
    --update configs/regression/base.yaml \
    --update configs/regression/modal-datadirs.yaml \
    --update configs/regression/mse-triplet.yaml \
    --update configs/regression/val_metrics.yaml \
    --update configs/compare/5k_steps.yaml \
    --update configs/optim/cosine.yaml \
    --update configs/regression/lo_mse.yaml \
    > logs/mse-triplet.log 2>&1 &

# nohup python scripts/train.py configs/base.yaml --where modal \
#     --update configs/regression/base.yaml \
#     --update configs/regression/modal-datadirs.yaml \
#     --update configs/regression/mse-neg-mse.yaml \
#     --update configs/regression/val_metrics.yaml \
#     --update configs/compare/5k_steps.yaml \
#     --update configs/optim/cosine.yaml \
#     --update configs/regression/lo_mse.yaml \
#     > logs/mse-neg-mse.log 2>&1 &

nohup python scripts/train.py configs/base.yaml --where modal \
    --update configs/regression/base.yaml \
    --update configs/regression/modal-datadirs.yaml \
    --update configs/regression/mse-pixel-mse.yaml \
    --update configs/regression/val_metrics.yaml \
    --update configs/compare/5k_steps.yaml \
    --update configs/optim/cosine.yaml \
    --update configs/regression/lo_mse.yaml \
    > logs/mse-pixel-mse.log 2>&1 &

nohup python scripts/train.py configs/base.yaml --where modal \
    --update configs/regression/base.yaml \
    --update configs/regression/modal-datadirs.yaml \
    --update configs/regression/mse-pixel-lpips.yaml \
    --update configs/regression/val_metrics.yaml \
    --update configs/compare/5k_steps.yaml \
    --update configs/optim/cosine.yaml \
    --update configs/regression/lo_mse.yaml \
    > logs/mse-pixel-lpips.log 2>&1 &

nohup python scripts/train.py configs/base.yaml --where modal \
    --update configs/regression/base.yaml \
    --update configs/regression/modal-datadirs.yaml \
    --update configs/regression/mse-dm.yaml \
    --update configs/regression/val_metrics.yaml \
    --update configs/compare/5k_steps.yaml \
    --update configs/optim/cosine.yaml \
    --update configs/regression/lo_mse.yaml \
    > logs/mse-pixel-lpips.log 2>&1 &