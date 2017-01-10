#!/bin/bash
export netType='wide-resnet'
export depth=16
export width=8
export dataset='svhn'
export save=logs/${dataset}/${netType}-${depth}x${width}
export experiment_number=1
mkdir -p ${save}
mkdir -p modelState

th main.lua \
    -dataset ${dataset} \
    -netType ${netType} \
    -resume modelState \
    -batchSize 128 \
    -dropout 0.4 \
    -top5_display false \
    -testOnly true \
    -depth ${depth} \
    -widen_factor ${width} \
    -nExperiment ${experiment_number} \
    | tee $save/log_${experiment_number}.txt
