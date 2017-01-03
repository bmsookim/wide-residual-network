-- ************************************************************
-- Author : Bumsoo Kim, 2016
-- Github : https://github.com/meliketoy/wide-residual-network
--
-- Korea University, Data-Mining Lab
-- wide-residual-networks Torch implementation
--
-- Description : train.lua
-- The training code for each datasets.
-- ***********************************************************

local optim = require 'optim'

local M = {}
local Trainer = torch.class('resnet.Trainer', M)
local elapsed_time = 0

-- Initialize training class
function Trainer:__init(model, criterion, opt, optimState)
    self.model = model
    self.criterion = criterion
    self.optimState = optimState or {
        learningRate = opt.LR,
        learningRateDecay = 0.0,
        momentum = opt.momentum,
        nesterov = true,
        dampening = 0.0,
        weightDecay = opt.weightDecay,
    }
    self.opt = opt
    self.params, self.gradParams = model:getParameters()
end

-- Training process
function Trainer:train(epoch, dataloader)
    -- Trains the model for a single epoch
    self.optimState.learningRate = self:learningRate(epoch)

   local timer = torch.Timer()
   local dataTimer = torch.Timer()

   local function feval()
      return self.criterion.output, self.gradParams
   end

   local trainSize = dataloader:size()
   local top1Sum, top5Sum, lossSum = 0.0, 0.0, 0.0
   local N = 0

   print('\n => Training Epoch # '..epoch)
   -- set the batch norm to training mode
   self.model:training()
   local trainTime = 0.0
   for n, sample in dataloader:run() do
      local dataTime = dataTimer:time().real

      -- Copy input and target to the GPU
      self:copyInputs(sample)

      local output = self.model:forward(self.input):float()
      local batchSize = output:size(1)
      local loss = self.criterion:forward(self.model.output, self.target)

      self.model:zeroGradParameters()
      self.criterion:backward(self.model.output, self.target)
      self.model:backward(self.input, self.criterion.gradInput)

      optim.sgd(feval, self.params, self.optimState)

      local top1, top5 = self:computeScore(output, sample.target, 1)
      top1Sum = top1Sum + top1*batchSize
      top5Sum = top5Sum + top5*batchSize
      lossSum = lossSum + loss*batchSize
      N = N + batchSize
      elapsed_time = elapsed_time + timer:time().real + dataTime

      -- print out a progress bar
      xlua.progress(n, trainSize)

      -- print out training report
      if n == trainSize then
          if self.opt.top5_display then
              print((' | [#%3d][training report]\tTop1 %6.3f%s  Top5 %6.3f%s  Loss %1.4f  Time %.3f')
              :format(epoch, top1, '%', top5, '%', loss, trainTime))
          else
              print((' | [#%3d][training report]\tTop1 %6.3f%s  Loss %1.4f  Time %.3f')
              :format(epoch, top1, '%', loss, trainTime))
          end
      else
          trainTime = trainTime + timer:time().real + dataTime
      end

      -- check that the storage didn't get changed do to an unfortunate getParameters call
      assert(self.params:storage() == self.model:parameters()[1]:storage())

      timer:reset()
      dataTimer:reset()
   end

   return top1Sum / N, top5Sum / N, lossSum / N
end

-- Validation process
function Trainer:test(epoch, dataloader)
    -- Computes the top N scores of the validation set
    local timer = torch.Timer()
    local dataTimer = torch.Timer()
    local size = dataloader:size()

    local nCrops = self.opt.tenCrop and 10 or 1
    local top1Sum, top5Sum = 0.0, 0.0
    local N = 0

    -- Set the batch normalization to validation mode : moving average 0.9
    self.model:evaluate()

    for n, sample in dataloader:run() do
        local dataTime = dataTimer:time().real

        -- Copy input and target into the GPUs
        self:copyInputs(sample)

        local output = self.model:forward(self.input):float()
        local batchSize = output:size(1) / nCrops
        local loss = self.criterion:forward(self.model.output, self.target)

        local top1, top5 = self:computeScore(output, sample.target, nCrops)
        top1Sum = top1Sum + top1*batchSize
        top5Sum = top5Sum + top5*batchSize
        N = N + batchSize
        elapsed_time = elapsed_time + timer:time().real + dataTime

        timer:reset()
        dataTimer:reset()
    end
    self.model:training()

    if self.opt.testOnly then
        return top1Sum/N, top5Sum/N
    end

    if self.opt.top5_display then
        print((' | [#%3d][validate report]\tTop1 %6.3f%s  Top5 %6.3f%s')
        :format(epoch, top1Sum / N, '%', top5Sum / N, '%'))
    else
        print((' | [#%3d][validate report]\tTop1 %6.3f%s'):format(epoch, top1Sum / N, '%'))
    end
    print(' | Elapsed time: '..math.floor(elapsed_time/3600)..' hours '..
                               math.floor((elapsed_time%3600)/60)..' minutes '..
                               math.floor((elapsed_time%3600)%60)..' seconds')
    return top1Sum/N, top5Sum/N
end

-- Scoring Process
function Trainer:computeScore(output, target, nCrops)
    if nCrops > 1 then
        -- Sum over crops
        output = output:view(output:size(1) / nCrops, nCrops, output:size(2)):sum(2):squeeze(2)
    end
    
    -- Coputes the top1 and top5 error rate
    local batchSize = output:size(1)
    local _ , predictions = output:float():sort(2, true) -- sort in descending orders

    -- Find which predictions match the target
    local correct = predictions:eq(
        target:long():view(batchSize, 1):expandAs(output))
        
    -- Top-1 score
    local top1 = (correct:narrow(2, 1, 1):sum() / batchSize)

    -- Top-5 score, if there are at least 5 classes
    local len = math.min(5, correct:size(2))
    local top5 = (correct:narrow(2, 1, len):sum() / batchSize)

    return top1*100, top5*100
end

-- Copying Inputs into the GPUs
function Trainer:copyInputs(sample)
    -- Copies the inputs into a CUDA tensor, if using 1 GPU, or to pinned memory
    -- If using DataParallelTable, the target is always copied to a CUDA tensor
    self.input = self.input or (self.opt.nGPU == 1 and torch.CudaTensor() or cutorch.createCudaHostTensor())
    self.target = self.target or torch.CudaTensor()
    self.input:resize(sample.input:size()):copy(sample.input)
    self.target:resize(sample.target:size()):copy(sample.target)
end

-- Decaying Learning Rate
function Trainer:learningRate(epoch)
    -- Learning rate according to training schedule
    local decay = 0
    if self.opt.dataset == 'cifar10' then 
        decay = epoch >= 160 and 3 or epoch >= 120 and 2 or epoch >= 60 and 1 or 0
    elseif self.opt.dataset == 'cifar100' then 
        decay = epoch >= 160 and 3 or epoch >= 120 and 2 or epoch >= 60 and 1 or 0
    elseif self.opt.dataset == 'svhn' then
        decay = epoch >= 160 and 3 or epoch >= 120 and 2 or epoch >= 60 and 1 or 0
    end
    return self.opt.LR * math.pow(0.2, decay)
end

return M.Trainer
