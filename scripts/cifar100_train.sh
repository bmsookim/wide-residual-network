#!/bin/bash
export netType='wide-resnet'
export depth=28
export width=10
export dataset='cifar100'
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
