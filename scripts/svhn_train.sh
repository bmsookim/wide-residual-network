#!/bin/bash
export netType='wide-resnet'
export depth=16
export width=8
export experiment_number=1
export dataset='svhn'
mkdir -p modelState

th main.lua \
    -dataset ${dataset} \
    -netType ${netType} \
    -batchSize 128 \
    -dropout 0.4 \
    -LR 0.01 \
    -top5_display false \
    -testOnly false \
    -depth ${depth} \
    -widen_factor ${width} \
    -nExperiment ${experiment_number} \
