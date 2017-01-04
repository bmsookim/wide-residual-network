# Wide Residual Networks
Wide-residual network implementations for cifar10, cifar100, and other kaggle challenges
Torch Implementation of Sergey Zagoruyko's [Wide Residual Networks](https://arxiv.org/pdf/1605.07146v2.pdf)

## Requirements
See the [installation instruction](INSTALL.md) for a step-by-step installation guide.
See the [server instruction](SERVER.md) for server settup.
- Install [Torch](http://torch.ch/docs/getting-started.html)
- Install [cuda-8.0](https://developer.nvidia.com/cuda-downloads)
- Install [cudnn v5.1](https://developer.nvidia.com/cudnn)
- Install luarocks packages
```bash
$ luarocks install cutorch
$ luarocks install xlua
$ luarocks install optnet
```
## Directions and datasets
- modelState    : The best model will be saved in this directory
- datasets      : Data preparation & preprocessing directory
- models        : Wide-residual network model structure file directory
- gen           : Generated t7 file for each dataset will be saved in this directory
- scripts       : Directory where the run file scripts are contained

## How to run
You can train each dataset of either cifar10, cifar100, svhn by running the script below.
```bash
$ sudo sh scripts/[:dataset]_train.sh

# For example, if you want to train the model on cifar10, you simply type
$ sudo sh scripts/cifar10_train.sh
```

You can test your own trained model of either cifar10, cifar100, svhn by running the script below.
```bash
$ sudo sh scripts/[:dataset]_test.sh
```

## Best Results
|   Dataset   | network              | dropout | Optimizer| Memory | Top1 acc(%)|
|:-----------:|:--------------------:|:-------:|----------|:------:|:----------:|
| CIFAR-10    | wide-resnet ensemble |   0.3   | Momentum | 20.21G |  **96.75** |
| CIFAR-100   | wide-resnet 28x20    |   0.3   | Momentum | 15.06G |  **82.38** |
| SVHN        | wide-resnet 16x8     |   0.4   | Momentum |        |            |

## Implementation Details

* CIFAR-10, CIFAR-100

|   epoch   | learning rate |  weight decay | Optimizer | Momentum | Nesterov |
|:---------:|:-------------:|:-------------:|:---------:|:--------:|:--------:|
|   0 ~ 60  |      0.1      |     0.0005    | Momentum  |    0.9   |   true   |
|  61 ~ 120 |      0.02     |     0.0005    | Momentum  |    0.9   |   true   |
| 121 ~ 160 |     0.004     |     0.0005    | Momentum  |    0.9   |   true   |
| 161 ~ 200 |     0.0008    |     0.0005    | Momentum  |    0.9   |   true   |

* SVHN

|   epoch   | learning rate |  weight decay | Optimizer | Momentum | Nesterov |
|:---------:|:-------------:|:-------------:|:---------:|:--------:|:--------:|
|   0 ~ 80  |      0.01     |     0.0005    | Momentum  |    0.9   |   true   |
|  81 ~ 120 |     0.001     |     0.0005    | Momentum  |    0.9   |   true   |
| 121 ~ 160 |     0.0001    |     0.0005    | Momentum  |    0.9   |   true   |


## CIFAR-10 Results
 
![alt tag](IMAGES/cifar10_image.png)

Below is the result of the test set accuracy for **CIFAR-10 dataset** training.

**Accuracy is the average of 5 runs**

| network           | dropout | preprocess | GPU:0 | GPU:1 | per epoch    | accuracy(%) |
|:-----------------:|:-------:|:----------:|:-----:|:-----:|:------------:|:-----------:|
| pre-ResNet-1001   |    0    |   meanstd  |   -   |   -   | 3 min 25 sec |    95.08    |
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
| pre-ResNet-1001   |    0    |   meanstd   |   -   |   -   | 3 min 25 sec |    77.29   |    93.44    |
| wide-resnet 28x10 |    0    |     ZCA     | 5.90G |   -   | 2 min 03 sec |    80.03   |    95.01    |
| wide-resnet 28x10 |    0    |   meanstd   | 5.90G |   -   | 2 min 03 sec |    81.01   |    95.44    |
| wide-resnet 28x10 |   0.3   |   meanstd   | 5.90G |   -   | 2 min 03 sec |    81.21   |    95.22    |
| wide-resnet 28x20 |   0.3   |   meanstd   | 8.13G | 6.93G | 4 min 05 sec |  **82.38** |  **96.06**  |
| wide-resnet 40x10 |   0.3   |   meanstd   | 8.93G |   -   | 3 min 06 sec |    81.47   |    95.65    |
| wide-resnet 40x14 |   0.3   |   meanstd   | 7.39G | 6.46G | 3 min 23 sec |    81.83   |    95.50    |

## SVHN Results

![alt tag](IMAGES/svhn_image.png)

Below is the result of the test set accrucay for **SVHN dataset** training.

**Accuracy is the result of 1 run**

| network           | dropout |  preprocess | GPU:0 | GPU:1 | per epoch    | Top1 acc(%)|
|:-----------------:|:-------:|:-----------:|:-----:|:-----:|:------------:|:----------:|
| wide-resnet 10x1  |   0.4   |   meanstd   | 0.47G | 0.46G | 1 min 37 sec |   93.815   |
| wide-resnet 10x8  |   0.4   |   meanstd   |   -   |   -   |   min    sec |            |
| wide-resnet 16x8  |   0.4   |   meanstd   |   -   |   -   |   min    sec |            | 
| wide-resnet 22x8  |   0.4   |   meanstd   |   -   |   -   |   min    sec |            |

