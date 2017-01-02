require 'torch'
require 'paths'
require 'optim'
require 'nn'
require 'cutorch'

local datasets = require 'datasets/init'
local opts = require 'opts'
local opt = opts.parse(arg)

datasets.generate(opt)
