-- ************************************************************
-- Author : Bumsoo Kim, 2016
-- Github : https://github.com/meliketoy/wide-residual-network
--
-- Korea University, Data-Mining Lab
-- wide-residual-networks Torch implementation
--
-- Description : svhn.lua
-- SVHN dataset loader
-- ***********************************************************

local t = require 'datasets/transforms'

local M = {}
local SVHNDataset = torch.class('resnet.SVHNDataset', M)

function SVHNDataset:__init(imageInfo, opt, split)
    assert(imageInfo[split], split)
    self.imageInfo = imageInfo[split]
    self.split = split
end

function SVHNDataset:get(i)
    local image = self.imageInfo.data[i]:float()
    local label = self.imageInfo.labels[i]

    return {
        input = image,
        target = label,
    }
end

function SVHNDataset:size()
    return self.imageInfo.data:size(1)
end

-- Computed from entire SVHN training set
local meanstd = {
    mean = {109.9, 109.7, 113.8},
    std = {50.1, 50.6, 50.9},
}

function SVHNDataset:preprocess()
    if self.split == 'train' then
        return t.Compose{
            t.ColorNormalize(meanstd),
            t.HorizontalFlip(0.5),
            t.RandomCrop(32, 4),
        }
    elseif self.split == 'val' then
        return t.ColorNormalize(meanstd)
    else
        error('invalid split: ' .. self.split)
    end
end

return M.SVHNDataset
