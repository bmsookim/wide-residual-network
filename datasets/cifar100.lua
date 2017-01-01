-- ************************************************************
-- Author : Bumsoo Kim, 2016
-- Github : https://github.com/meliketoy/wide-residual-network
--
-- Korea University, Data-Mining Lab
-- wide-residual-networks Torch implementation
--
-- Description : cifar100.lua
-- CIFAR-100 dataset loader
-- ***********************************************************

local t = require 'datasets/transforms'

local M = {}
local CifarDataset = torch.class('resnet.CifarDataset', M)

function CifarDataset:__init(imageInfo, opt, split)
    assert(imageInfo[split], split)
    self.imageInfo = imageInfo[split]
    self.split = split
end

function CifarDataset:get(i)
    local image = self.imageInfo.data[i]:float()
    local label = self.imageInfo.labels[i]

    return {
        input = image,
        target = label,
    }
end

function CifarDataset:size()
    return self.imageInfo.data:size(1)
end


-- Computed from entire CIFAR-100 training set with this code:
--      dataset = torch.load('cifar100.t7')
--      tt = dataset.train.data:double();
--      tt = tt:transpose(2,4);
--      tt = tt:reshape(50000*32*32, 3);
--      tt:mean(1)
--      tt:std(1)
local meanstd = {
    mean = {129.3, 124.1, 112.4},
    std  = {68.2,  65.4,  70.4},
}

function CifarDataset:preprocess()
    if self.split == 'train' then
        return t.Compose{
            t.Scale(36),
            t.ColorNormalize(meanstd),
            t.HorizontalFlip(0.5),
            t.RandomCrop(32, 0),
        }
    elseif self.split == 'val' then
        return t.ColorNormalize(meanstd)
    else
        error('invalid split: ' .. self.split)
    end
end

return M.CifarDataset
