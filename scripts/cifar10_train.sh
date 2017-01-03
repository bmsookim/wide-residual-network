#!/bin/bash
export netType='wide-resnet'
export depth=28
export width=10
export dataset='cifar10'
export experiment_num=2
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
    -nExperiment ${experiment_num} \
