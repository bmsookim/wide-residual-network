#!/bin/bash
export netType='wide-resnet'
export dataset='cifar10'
export mode='min'
export save=logs/${dataset}/${netType}-ensemble
export experiment_number=1
mkdir -p $save

th ensemble.lua \
    -ensembleMode ${mode} \
    -dataset ${dataset} \
    -netType ${netType} \
    -batchSize 16 \
    -dropout 0 \
    -optnet false \
    | tee $save/log_ensemble_${experiment_number}.txt
