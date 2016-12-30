-- ************************************************************
-- Author : Bumsoo Kim, 2016
-- Github : https://github.com/meliketoy/wide-residual-network
--
-- Korea University, Data-Mining Lab
-- wide-residual-networks Torch implementation
--
-- Description : init.lua
-- Initializes CIFAR-10, CIFAR-100 datasets
-- ***********************************************************

local M = {}

local function isvalid(opt, cachePath)
    local imageInfo = torch.load(cachePath)
    if imageInfo.basedir and imageInfo.basedir ~= opt.data then
        return false
    end
    return true
end

function M.create(opt, split)
    local cachePath = paths.concat(opt.gen, opt.dataset .. '.t7')
    if not paths.filep(cachePath) or not isvalid(opt, cachePath) then
        paths.mkdir('gen')
 
        local script = paths.dofile(opt.dataset .. '-gen.lua')
        script.exec(opt, cachePath)
    end
    local imageInfo = torch.load(cachePath)

    local Dataset = require('datasets/' .. opt.dataset)
    return Dataset(imageInfo, opt, split)
end

return M
