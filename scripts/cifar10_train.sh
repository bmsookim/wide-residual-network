#!/bin/bash
export netType='wide-resnet'
export depth=10
export width=1
export dataset='cifar10'
mkdir -p modelState

th main.lua \
    -dataset ${dataset} \
    -netType ${netType} \
    -batchSize 128 \
    -dropout 0.3 \
    -top5_display false \
    -testOnly false \
    -depth ${depth} \
    -widen_factor ${width} \
    -nEpochs 2 \
