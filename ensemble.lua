-- ************************************************************
-- Author : Bumsoo Kim, 2016
-- Github : https://github.com/meliketoy/wide-residual-network
--
-- Korea University, Data-Mining Lab
-- wide-residual-networks Torch implementation
--
-- Description : ensemble.lua
-- The Ensemble model testing code.
-- ***********************************************************

require 'torch'
require 'paths'
require 'optim'
require 'nn'

local DataLoader = require 'dataloader'
local models = require 'networks/init'
local Tester = require 'test'
local opts = require 'opts'
local checkpoints = require 'checkpoints'

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

local opt = opts.parse(arg)
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

local checkpoint, optimState = checkpoints.best(opt)

opt.depth = 10
opt.widen_factor = 1
local model1, criterion = models.setup(opt, checkpoint)

opt.widen_factor = 1
local model2, criterion = models.setup(opt, checkpoint)

local _, valLoader = DataLoader.create(opt)

local tester = Tester(model1, model2, criterion, opt, optimState)

local top1, top5 = tester:test(opt.nEpochs, valLoader)

print('\n===============[ Test Result Report ]===============')
print(' * Dataset\t: '..opt.dataset)
print(' * Network\t: '..opt.netType..' '..opt.depth..'x'..opt.widen_factor)
print(' * Dropout\t: '..opt.dropout)
print(' * nGPU\t\t: '..opt.nGPU)
print(' * Top1\t\t: '..string.format('%6.3f', top1)..'%')
if opt.top5_display then
    print(' * Top5\t\t: '..string.format('%6.3f', top5)..'%')
end
print('=====================================================')


--[[ "Extra data loading script"
cachePath = paths.concat(opt.gen, opt.dataset .. '.t7')
local imageInfo = torch.load(cachePath)
local Dataset = require('datasets/'..opt.dataset)

loader = Dataset(imageInfo, opt, 'val')
test_size = loader:size()

_G.preprocess = loader:preprocess()

get_input = _G.preprocess(loader:get(1).input)

-- set this manually! --
sz = 10
------------------------

target = torch.IntTensor(sz):zero()
smp_target = loader:get(1).target
target[smp_target] = 1
local output = model:forward(get_input:cuda()):float()

print(output)

local _, predictions = output:sort(2, true)
]]--
