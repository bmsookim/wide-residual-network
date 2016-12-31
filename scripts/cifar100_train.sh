#!/bin/bash
export netType='wide-resnet'
export depth=10
export width=1
export dataset='cifar100'
mkdir -p modelState

th main.lua \
    -dataset ${dataset} \
    -netType ${netType} \
    -top5_display true \
    -testOnly false \
    -dropout 0.3 \
    -batchSize 128 \
    -depth ${depth} \
    -widen_factor ${width} \
    -nEpochs 2 \
