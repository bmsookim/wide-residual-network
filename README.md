# wide-residual-network
Wide-residual network implementations for cifar10, cifar100, and other kaggle challenges
Torch Implementation

## Requirements
See the [installation instruction](INSTALL.md) for a step-by-step installation guide.
See the [server instruction](SERVER.md) for server settup.
- Install [Torch](http://torch.ch/docs/getting-started.html)
- Install [cuda-8.0](https://developer.nvidia.com/cuda-downloads)
- Install [cudnn v5.1](https://developer.nvidia.com/cudnn)
- Install luarocks packages
```bash
$ luarocks install cutorch
$ luarocks install optnet
```
## Directions and datasets
- modelState    : The best model will be saved in this directory
- datasets      : Data preparation & preprocessing directory
- models        : Wide-residual network model structure file directory
- gen           : Generated t7 file for each dataset will be saved in this directory
- scripts       : Directory where the run file scripts are contained

## How to run
You can train each dataset of either cifar10, cifar100 by running the script below.
```bash
$ sudo sh scripts/[:dataset]_train.sh

# For example, if you want to train the model on cifar10, you simply type
$ sudo sh scripts/cifar10_train.sh
```

You can test your own trained model of either cifar10, cifar100 by running the script below.
```bash
$ sudo sh scripts/[:dataset]_test.sh
```

## Best Results
|   Dataset   | network           | dropout | Optimizer| Memory | epoch | per epoch    | Top1 acc(%)|
|:-----------:|:-----------------:|:-------:|----------|:------:|:-----:|:------------:|:----------:|
| CIFAR-10    | wide-resnet 40x14 |   0.3   | Momentum | 15.06G | 200   | 4 min 10 sec |  **96.44** |
| CIFAR-100   | wide-resnet 28x20 |   0.3   | Momentum | 15.06G | 200   | 4 min 05 sec |  **82.38** |

## Implementation Details
|   epoch   | learning rate |  weight decay | Optimizer |
|:---------:|:-------------:|:-------------:|:---------:|
|   0 ~ 60  |      0.1      |     0.0005    | Momentum  |
|  61 ~ 120 |      0.02     |     0.0005    | Momentum  |
| 121 ~ 160 |     0.004     |     0.0005    | Momentum  |
| 161 ~ 200 |     0.0008    |     0.0005    | Momentum  |


## CIFAR-10 Results
 
![alt tag](IMAGES/cifar10_image.png)

Below is the result of the test set accuracy for **CIFAR-10 dataset** training.

**Accuracy is the average of 5 runs**

| network           | dropout | preprocess | GPU:0 | GPU:1 | per epoch    | accuracy(%) |
|:-----------------:|:-------:|:----------:|:-----:|:-----:|:------------:|:-----------:|
| wide-resnet 28x10 |    0    |     ZCA    | 5.90G |   -   | 2 min 03 sec |    95.84    |
| wide-resnet 28x10 |    0    |   meanstd  | 5.90G |   -   | 2 min 03 sec |    96.15    |
| wide-resnet 28x10 |   0.3   |   meanstd  | 5.90G |   -   | 2 min 03 sec |    96.30    |
| wide-resnet 28x20 |   0.3   |   meanstd  | 8.13G | 6.93G | 4 min 10 sec |    96.24    |
| wide-resnet 40x10 |   0.3   |   meanstd  | 8.08G |   -   | 3 min 13 sec |    96.28    |
| wide-resnet 40x14 |   0.3   |   meanstd  | 7.37G | 6.46G | 3 min 23 sec |  **96.44**  |

## CIFAR-100 Results

![alt tag](IMAGES/cifar100_image.png)

Below is the result of the test set accuracy for **CIFAR-100 dataset** training.

**Accuracy is the average of 5 runs**

| network           | dropout |  preprocess | GPU:0 | GPU:1 | per epoch    | Top1 acc(%)| Top5 acc(%) |
|:-----------------:|:-------:|:-----------:|:-----:|:-----:|:------------:|:----------:|:-----------:|
| wide-resnet 28x10 |    0    |     ZCA     | 5.90G |   -   | 2 min 03 sec |    80.03   |    95.01    |
| wide-resnet 28x10 |    0    |   meanstd   | 5.90G |   -   | 2 min 03 sec |    81.01   |    95.44    |
| wide-resnet 28x10 |   0.3   |   meanstd   | 5.90G |   -   | 2 min 03 sec |    81.21   |    95.22    |
| wide-resnet 28x20 |   0.3   |   meanstd   | 8.13G | 6.93G | 4 min 05 sec |  **82.38** |  **96.06**  |
| wide-resnet 40x10 |   0.3   |   meanstd   | 8.93G |   -   | 3 min 06 sec |    81.47   |    95.65    |
| wide-resnet 40x14 |   0.3   |   meanstd   | 7.39G | 6.46G | 3 min 23 sec |    81.83   |    95.50    |

