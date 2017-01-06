-- ************************************************************
-- Author : Bumsoo Kim, 2016
-- Github : https://github.com/meliketoy/wide-residual-network
--
-- Korea University, Data-Mining Lab
-- wide-residual-networks Torch implementation
--
-- Description : checkpoints.lua
-- The checkpoint code for saving the models.
-- ***********************************************************

local checkpoint = {}
local before_was_best = false

local function deepCopy(tbl)
   -- creates a copy of a network with new modules and the same tensors
   local copy = {}
   for k, v in pairs(tbl) do
      if type(v) == 'table' then
         copy[k] = deepCopy(v)
      else
         copy[k] = v
      end
   end
   if torch.typename(tbl) then
      torch.setmetatable(copy, torch.typename(tbl))
   end
   return copy
end

function checkpoint.latest(opt)
   if opt.resume == 'none' then
      return nil
   end
   local latestPath = paths.concat(opt.resume, 'latest.t7')
   if not paths.filep(latestPath) then
      return nil
   end

   print('=> Loading checkpoint ' .. latestPath)
   local latest = torch.load(latestPath)
   local optimState = torch.load(paths.concat(opt.resume, latest.optimFile))

   return latest, optimState
end

function checkpoint.best(opt)
   if opt.resume == 'none' then
      return nil
   end
   local bestPath = paths.concat(opt.resume, 'best.t7')
   if not paths.filep(bestPath) then
      return nil
   end

   -- print('=> Loading checkpoint ' .. bestPath)
   local best = torch.load(bestPath)
   local optimState = torch.load(paths.concat(opt.resume, best.optimFile))

   return best, optimState
end

function checkpoint.save(epoch, model, optimState, isBestModel, opt)
   -- don't save the DataParallelTable for easier loading on other machines
   if torch.type(model) == 'nn.DataParallelTable' then
      model = model:get(1)
   end

   -- create a clean copy on the CPU without modifying the original network
   model = deepCopy(model):float():clearState()

   local modelFile = 'model_' .. epoch .. '.t7'
   local optimFile = 'optimState_' .. epoch .. '.t7'

   if(opt.saveLatest) then
      torch.save(paths.concat(opt.save, modelFile), model)
      torch.save(paths.concat(opt.save, optimFile), optimState)

      --save the latest file
      torch.save(paths.concat(opt.save, 'latest.t7'), {
         epoch = epoch,
         modelFile = modelFile,
         optimFile = optimFile,
       })

      if before_was_best == false then
         bef_model = 'model_' .. (epoch-1)..'.t7'
         bef_optim = 'optimState_' .. (epoch-1)..'.t7'
         os.remove(paths.concat(opt.save, bef_model))
         os.remove(paths.concat(opt.save, bef_optim))
      end
   end

   if isBestModel then
      torch.save(paths.concat(opt.save, modelFile), model)
      torch.save(paths.concat(opt.save, optimFile), optimState)
      torch.save(paths.concat(opt.save, 'best.t7'), {
         epoch = epoch,
         modelFile = modelFile,
         optimFile = optimFile,
      })
      if(epoch ~= 1) then 
         for i=1, (epoch-1) do
            bef_model = 'model_' .. (i) .. '.t7'
            bef_optim = 'optimState_' .. (i) .. '.t7'
            os.remove(paths.concat(opt.save, bef_model))
            os.remove(paths.concat(opt.save, bef_optim))
            before_was_best = true
         end
      end
   else
      before_was_best = false
   end
end

return checkpoint
