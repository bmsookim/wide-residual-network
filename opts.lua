-- ************************************************************
-- Author : Bumsoo Kim, 2016
-- Github : https://github.com/meliketoy/wide-residual-network
--
-- Korea University, Data-Mining Lab
-- wide-residual-networks Torch implementation
--
-- Description : opts.lua
-- The command line options for configuration.
-- ***********************************************************

local M = { }

function M.parse(arg)
   local cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Torch-7 wide-resNet Training script')
   cmd:text('See https://github.com/meliketoy/wide-residual-network')
   cmd:text()
   cmd:text('Options:')
   ---------------------- General options --------------------
   cmd:option('-data',       '',         'Path to dataset')
   cmd:option('-dataset',    '',         'Options: cifar10 | cifar100')
   cmd:option('-manualSeed', 0,          'Manually set RNG seed')
   cmd:option('-nGPU',       2,          'Number of GPUs to use by default')
   cmd:option('-backend',    'cudnn',    'Options: cudnn | cunn')
   cmd:option('-cudnn',      'fastest',  'Options: fastest | default | deterministic')
   cmd:option('-gen',        'gen',      'Path to save generated files')
   
   ---------------------- Dataloader options -----------------
   cmd:option('-nThreads',        16, 'number of data loading threads')
   
   ---------------------- Training options -------------------
   cmd:option('-nEpochs',         0,       'Number of total epochs to run')
   cmd:option('-epochNumber',     1,       'Manual epoch number (useful on restarts)')
   cmd:option('-batchSize',       128,     'mini-batch size (1 = pure stochastic)')
   cmd:option('-top5_display',    'false', 'display top5 accuracy')
   cmd:option('-testOnly',        'false', 'Run on validation set only')
   cmd:option('-tenCrop',         'false', 'Ten-crop testing')
   
   ---------------------- Checkpoints options ----------------
   cmd:option('-save',            'modelState', 'Directory in which to save checkpoints')
   cmd:option('-resume',          '',           'Resume from the latest checkpoint in this directory')
   cmd:option('-saveLatest',      'false',      'Save the latest file')
   
   ---------------------- Optimization options ---------------
   cmd:option('-LR',              0.1,     'initial learning rate')
   cmd:option('-momentum',        0.9,     'momentum')
   cmd:option('-weightDecay',     0.0005,  'weight decay')
   
   ---------------------- Model options ----------------------
   cmd:option('-netType',         'wide-resnet', 'Options: vggnet | resnet | wide-resnet')
   cmd:option('-depth',           28,         'ResNet depth: 6n+4', 'number')
   cmd:option('-widen_factor',    10,         'Wide-Resnet width', 'number')
   cmd:option('-dropout',         0.3,        'Dropout rate')
   cmd:option('-shortcutType',    '',         'Options: A | B | C')
   cmd:option('-retrain',         'none',     'fine-tuning, Path to model to retrain with')
   cmd:option('-optimState',      'none',     'Path to an optimState to reload from')
   
   ---------------------- Gradients options ------------------
   cmd:option('-shareGradInput',  'false', 'Share gradInput tensors to reduce memory usage')
   cmd:option('-optnet',          'true',  'Use optnet to reduce memory usage')
   cmd:option('-resetClassifier', 'false', 'Reset the fully connected layer for fine-tuning')
   cmd:option('-nClasses',         0,      'Number of classes in the dataset')
   cmd:text()

   local opt = cmd:parse(arg or {})

   opt.testOnly = opt.testOnly ~= 'false'
   opt.saveLatest = opt.saveLatest ~= 'false'
   opt.tenCrop = opt.tenCrop ~= 'false'
   opt.shareGradInput = opt.shareGradInput ~= 'false'
   opt.optnet = opt.optnet ~= 'false'
   opt.resetClassifier = opt.resetClassifier ~= 'false'
   opt.top5_display = opt.top5_display ~= 'false'
   opt.nGPU = cutorch.getDeviceCount()

   if opt.netType == 'wide-resnet' then 
       opt.save = opt.save..'/'..opt.dataset..'/'..opt.netType..'-'..opt.depth..'x'..opt.widen_factor..'/'
       if opt.resume ~= '' then 
           opt.resume = opt.resume..'/'..opt.dataset..'/'..opt.netType..'-'..opt.depth..'x'..opt.widen_factor..'/'
       end
   elseif opt.netType == 'resnet' then
       opt.save = opt.save..'/'..opt.dataset..'/'..opt.netType..'-'..opt.depth..'/'
       if opt.resume ~= '' then
           opt.resume = opt.resume..'/'..opt.dataset..'/'..opt.netType..'-'..opt.depth..'/'
       end
   else
       opt.save = opt.save..'/'..opt.dataset..'/'..opt.netType..'/'
       if opt.resume ~= '' then
           opt.resume = opt.resume..'/'..opt.dataset..'/'..opt.netType..'/'
       end
   end

   if not paths.dirp(opt.save) and not paths.mkdir(opt.save) then
      cmd:error('error: unable to create checkpoint directory: ' .. opt.save .. '\n')
   end

   if opt.dataset == 'cifar10' then
      -- Default shortcutType=A and nEpochs=200
      opt.shortcutType = opt.shortcutType == '' and 'A' or opt.shortcutType
      opt.nEpochs = opt.nEpochs == 0 and 200 or opt.nEpochs
   elseif opt.dataset == 'cifar100' then
      -- Default shortcutType=A and nEpochs=200
      opt.shortcutType = opt.shortcutType == '' and 'A' or opt.shortcutType
      opt.nEpochs = opt.nEpochs == 0 and 200 or opt.nEpochs
   else
      cmd:error('unknown dataset: ' .. opt.dataset)
   end

   if opt.resetClassifier then
      if opt.nClasses == 0 then
         cmd:error('-nClasses required when resetClassifier is set')
      end
   end

   if opt.shareGradInput and opt.optnet then
      cmd:error('error: cannot use both -shareGradInput and -optnet')
   end

   return opt
end

return M
