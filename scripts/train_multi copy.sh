#!/bin/bash


nohup python scripts/train.py configs/base.yaml --where modal \
    --update configs/regression/base.yaml \
    --update configs/regression/modal.yaml \
    --update configs/regression/dm/mse-dm-a.yaml \
    --update configs/compare/5k_steps.yaml \
    > logs/mse-dm-a.log 2>&1 &

nohup python scripts/train.py configs/base.yaml --where modal \
    --update configs/regression/base.yaml \
    --update configs/regression/modal.yaml \
    --update configs/regression/dm/mse-dm-b.yaml \
    --update configs/compare/5k_steps.yaml \
    > logs/mse-dm-b.log 2>&1 &

nohup python scripts/train.py configs/base.yaml --where modal \
    --update configs/regression/base.yaml \
    --update configs/regression/modal.yaml \
    --update configs/regression/dm/mse-dm-c.yaml \
    --update configs/compare/5k_steps.yaml \
    > logs/mse-dm-c.log 2>&1 &

nohup python scripts/train.py configs/base.yaml --where modal \
    --update configs/regression/base.yaml \
    --update configs/regression/modal.yaml \
    --update configs/regression/dm/mse-dm-d.yaml \
    --update configs/compare/5k_steps.yaml \
    > logs/mse-dm-d.log 2>&1 &
