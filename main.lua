-- ************************************************************
-- Author : Bumsoo Kim, 2016
-- Github : https://github.com/meliketoy/wide-residual-network
--
-- Korea University, Data-Mining Lab
-- wide-residual-networks Torch implementation
--
-- Description : main.lua
-- The main code for training & validation of datasets.
-- ***********************************************************

require 'torch'
require 'paths'
require 'optim'
require 'nn'

--------- Import each modules --------- 
local Dataloader = require 'dataloader'
local models = require 'networks/init'
local Trainer = require 'train'
local opts = require 'opts'
local checkpoints = require 'checkpoints'
---------------------------------------

--------- Set default options ---------
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)
---------------------------------------

-- Parsing command options
local opt = opts.parse(arg)
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

-- Load previous checkpoints, else initialize
local checkpoint, optimState = checkpoints.best(opt)

-- Create & Setting up the model
local model, criterion = models.setup(opt, checkpoint)
print(model) -- print the model layout, torch

-- Create modules for loading batch data in the training & validation process
local trainLoader, valLoader = DataLoader.create(opt)

-- The 'trainer' module will handle the training & validation loop
local trainer = Trainer(model, criterion, opt, optimState)

if opt.testOnly then
    local top1Err, top5Err = trainer:test(0, valLoader)
    print(' * Results Top1 : ', string.format('%6.3f', top1Err)..'%\n')
    if opt.top5_display then
        -- print('           Top5 : ', string.format('%6.3f', top5Err)..'%\n'..)
    end
    return
end

-- Start from the next checkpoint from where the model was saved
local startEpoch = checkpoint and checkpoint.epoch + 1 or opt.epochNumber
local bestTop1, bestTop5 = 0, 0

-- Training iteration
for epoch = startEpoch, opt.nEpochs do
    -- Train for a single epoch
    local trainTop1, trainTop5, trainLoss = trainer:train(epoch, trainLoader)

    -- Run model on validation set
    local testTop1, testTop5 = trainer:test(epoch, valLoader)

    local bestModel = false
    if testTop1 > bestTop1 then
        bestModel = true
        bestTop1 = testTop1
        bestTop5 = testTop5
        print('==================================================================')
        print(' * Best model (Top1): ', string.format('%5.2f', testTop1)..'%\n')
        if opt.top5_display then
            print('              (Top5): ', string.format('%5.2f', testTop5)..'%\n')
        end
        print('=> Saving the best model in '..opt.save)
        print('==================================================================\n')
    end

    -- Save the model if it is the current best model
    checkpoints.save(epoch, model, trainer.optimState, bestModel, opt)
end

print(' * Results Top1 : ', string.format('%6.3f', top1Err)..'%\n')
if opt.top5_display then
    --print('           Top5 : ', string.format('%6.3f', top5Err)..'%\n'..)
end 
