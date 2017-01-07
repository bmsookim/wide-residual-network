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

-- List of ensemble options
local opt = opts.parse(arg)
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

-- ensemble depths
<<<<<<< HEAD
ens_depth         = torch.Tensor({28, 28, 28, 28, 40, 40, 40})
ens_widen_factor  = torch.Tensor({20, 20, 20, 20, 10, 14, 14})
ens_nExperiment   = torch.Tensor({ 2,  3,  4,  5,  5,  3,  4})
=======
ens_depth         = torch.Tensor({28, 28, 40, 40, 40})
ens_widen_factor  = torch.Tensor({20, 20, 10, 14, 14})
ens_nExperiment   = torch.Tensor({ 1,  2,  5,  3,  4})
>>>>>>> 1330b24f4aeb8fc9b968adc54ab321e8d5995f04

-- get ensemble numbers
opt.nEnsemble = ens_depth:size(1)
opt.ensembleMode = 'min'

function set_opt(opt, id)
    opt.depth = ens_depth[id]
    opt.widen_factor = ens_widen_factor[id]
    opt.nExperiment = ens_nExperiment[id]
    -- assume ensembles are only done for wide-resnets
    opt.resume = 'modelState/'..opt.dataset..'/'..
                  opt.netType..'-'..opt.depth..'x'..opt.widen_factor..'/'..opt.nExperiment..'/'
end

model_tensor = {}

local timer = torch.Timer()
for i=1,opt.nEnsemble do
    set_opt(opt, i)
    local checkpoint, optimState = checkpoints.best(opt)
    model_tensor[i], criterion = models.setup(opt, checkpoint)
end

local _, valLoader = DataLoader.create(opt)

-- Testing ensemble model
local tester = Tester(model_tensor, criterion, opt, optimState)

local top1, top5 = tester:test(opt.nEpochs, valLoader)

print('\n===============[ Test Result Report ]===============')
print(' * Dataset\t: '..opt.dataset)
print(' * Ensemble Network : ')
for i=1,opt.nEnsemble do
    print('   | Network'..i..'\t: '..opt.netType..' '..ens_depth[i]..'x'..ens_widen_factor[i]..', '..ens_nExperiment[i])
end
print(' * Dropout\t: '..opt.dropout)
print(' * nGPU\t\t: '..opt.nGPU)
print(' * Top1\t\t: '..string.format('%6.3f', top1)..'%')
if opt.top5_display then
    print(' * Top5\t\t: '..string.format('%6.3f', top5)..'%')
end
print(' * Elapsed time\t: '..xlua.formatTime(timer:time().real))
print('=====================================================')
