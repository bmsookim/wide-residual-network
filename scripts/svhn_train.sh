#!/bin/bash
export netType='wide-resnet'
export depth=16
export width=8
export dataset='svhn'
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
