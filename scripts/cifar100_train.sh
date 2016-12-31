#!/bin/bash
export netType='wide-resnet'
export depth=10
export width=1
export dataset='cifar100'
export save=logs/${dataset}/${netType}-${depth}x${width}
export experiment_number=1
mkdir -p $save
mkdir -p modelState

th main.lua \
    -dataset ${dataset} \
    -netType ${netType} \
    -nGPU 1 \
    -top5_display true \
    -testOnly false \
    -dropout 0.3 \
    -batchSize 128 \
    -depth ${depth} \
    -widen_factor ${width} \
    -nEpochs 2 \
