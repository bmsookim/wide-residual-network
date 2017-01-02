 -- ************************************************************
 -- Author : Bumsoo Kim, 2016
 -- Github : https://github.com/meliketoy/wide-residual-network
 --
 -- Korea University, Data-Mining Lab
 -- wide-residual-networks Torch implementation
 --
 -- Description : svhn-gen.lua
 -- Automatically downloads SVHN dataset from
 -- http://torch7.s3-website-us-east-1.amazonaws.com/data/svhn.t7.tgz
 -- ***********************************************************
 
 local URL = 'http://torch7.s3-website-us-east-1.amazonaws.com/data/svhn.t7.tgz'

 local M = {}

 function M.exec(opt, cacheFile)
     print("=> Downloading SVHN dataset from "..URL)
     local ok = os.execute('curl ' .. URL .. ' | tar xz -C gen/')
     assert(ok == true or ok == 0, 'error downloading SVHN')
     local train = torch.load('gen/housenumbers/train_32x32.t7', 'ascii')
     local extra = torch.load('gen/housenumbers/extra_32x32.t7', 'ascii')
     local test  = torch.load('gen/housenumbers/test_32x32.t7',  'ascii')

     print(" | combining dataset into a single file")
     local trainData = {
         data = torch.cat(train.X:transpose(3,4), extra.X:transpose(3,4), 1),
         labels = torch.cat(train.y[1], extra.y[1], 1),
     }
     local testData = {
         data = test.X:transpose(3,4),
         labels = test.y[1],
     }

     print(" | saving SVHN dataset to " .. cacheFile)
     torch.save(cacheFile, {
         train = trainData,
         val = testData,
     })
 end

 return M
