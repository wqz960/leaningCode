# DPN and DenseNet Series

## Overview
DenseNet is a new network structure published in CVPR as the best paper in 2017. The network is based on a new cross-layer connection block, that is dense-block. Compared to the bottleneck in ResNet, a more aggressive dense connection mechanism is designed in dense-block, that is, connecting all the layers to each other, each layer will be feed by all the outputs of previous layers as its additional input. DenseNet stacks all dense-blocks into a densely connected network. The way of dense connection makes DenseNet easier to backpropagate, making the network easier to train.
The full name of DPN is Dual Path Networks, that is, dual-channel network. The network is a combination of DenseNet and ResNeXt, DenseNet can extract new features from the previous layers, and ResNeXt can reuse the extracted features in the previous layers. The author further found that, ResNeXt has high reuse rate for features, but low redundancy, DenseNet can create new features, but the redundancy is high, Combining the advantages of the two structures, the author designed the DPN. The DPN network achieved better results than ResNeXt and DenseNet under the same FLOPS and parameters.

The FLOPS, parameter amount, and forward time on T4 GPU are shown in the figure below.

![](../../images/models/T4_benchmark/t4.fp32.bs4.DPN.flops.png)

![](../../images/models/T4_benchmark/t4.fp32.bs4.DPN.params.png)

![](../../images/models/T4_benchmark/t4.fp32.bs4.DPN.png)

![](../../images/models/T4_benchmark/t4.fp16.bs4.DPN.png)

Now there are 10 pretrained models for DenseNet and DPN in Paddle, Its performance is shown in the figure above, we can see that, Under the same FLOPS and parameter amount, DPN can reach higher accuracy than DenseNet, but because DPN has more branches, its inference speed is slower than DenseNet. Because DenseNet264 has the deepest network structure, DenseNet264 is the network with the largest amount of parameters in the DenseNet series, and DenseNet161 has the largest width of network, which makes it the most computationally intensive and highest accuracy network of the DenseNet series. From the perspective of inference speed, DenseNet161 has a faster speed than DenseNet264, so it has a greater advantage than DenseNet264.

For DPN series, the larger FLOPS and parameters of the model, the higher the accuracy of model. Among them, since DPN107 has largest width of network, it is the network with the largest amount of parameters and calculations in this series of networks.

## Accuracy, FLOPS and Parameters

| Models      | Top1   | Top5   | Reference<br>top1 | Reference<br>top5 | FLOPS<br>(G) | Parameters<br>(M) |
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| DenseNet121 | 0.757  | 0.926  | 0.750             |                   | 5.690        | 7.980             |
| DenseNet161 | 0.786  | 0.941  | 0.778             |                   | 15.490       | 28.680            |
| DenseNet169 | 0.768  | 0.933  | 0.764             |                   | 6.740        | 14.150            |
| DenseNet201 | 0.776  | 0.937  | 0.775             |                   | 8.610        | 20.010            |
| DenseNet264 | 0.780  | 0.939  | 0.779             |                   | 11.540       | 33.370            |
| DPN68       | 0.768  | 0.934  | 0.764             | 0.931             | 4.030        | 10.780            |
| DPN92       | 0.799  | 0.948  | 0.793             | 0.946             | 12.540       | 36.290            |
| DPN98       | 0.806  | 0.951  | 0.799             | 0.949             | 22.220       | 58.460            |
| DPN107      | 0.809  | 0.953  | 0.802             | 0.951             | 35.060       | 82.970            |
| DPN131      | 0.807  | 0.951  | 0.801             | 0.949             | 30.510       | 75.360            |




## Inference Speed on V100 GPU

| Models                               | Crop Size | Resize Short Size | FP32<br>Batch Size=1<br>(ms) |
|-------------|-----------|-------------------|--------------------------|
| DenseNet121 | 224       | 256               | 4.371                    |
| DenseNet161 | 224       | 256               | 8.863                    |
| DenseNet169 | 224       | 256               | 6.391                    |
| DenseNet201 | 224       | 256               | 8.173                    |
| DenseNet264 | 224       | 256               | 11.942                   |
| DPN68       | 224       | 256               | 11.805                   |
| DPN92       | 224       | 256               | 17.840                   |
| DPN98       | 224       | 256               | 21.057                   |
| DPN107      | 224       | 256               | 28.685                   |
| DPN131      | 224       | 256               | 28.083                   |



## Inference Speed on T4 GPU

| Models      | Crop Size | Resize Short Size | FP16<br>Batch Size=1<br>(ms) | FP16<br>Batch Size=4<br>(ms) | FP16<br>Batch Size=8<br>(ms) | FP32<br>Batch Size=1<br>(ms) | FP32<br>Batch Size=4<br>(ms) | FP32<br>Batch Size=8<br>(ms) |
|-------------|-----------|-------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|------------------------------|
| DenseNet121 | 224       | 256               | 4.16436                      | 7.2126                       | 10.50221                     | 4.40447                      | 9.32623                      | 15.25175                     |
| DenseNet161 | 224       | 256               | 9.27249                      | 14.25326                     | 20.19849                     | 10.39152                     | 22.15555                     | 35.78443                     |
| DenseNet169 | 224       | 256               | 6.11395                      | 10.28747                     | 13.68717                     | 6.43598                      | 12.98832                     | 20.41964                     |
| DenseNet201 | 224       | 256               | 7.9617                       | 13.4171                      | 17.41949                     | 8.20652                      | 17.45838                     | 27.06309                     |
| DenseNet264 | 224       | 256               | 11.70074                     | 19.69375                     | 24.79545                     | 12.14722                     | 26.27707                     | 40.01905                     |
| DPN68       | 224       | 256               | 11.7827                      | 13.12652                     | 16.19213                     | 11.64915                     | 12.82807                     | 18.57113                     |
| DPN92       | 224       | 256               | 18.56026                     | 20.35983                     | 29.89544                     | 18.15746                     | 23.87545                     | 38.68821                     |
| DPN98       | 224       | 256               | 21.70508                     | 24.7755                      | 40.93595                     | 21.18196                     | 33.23925                     | 62.77751                     |
| DPN107      | 224       | 256               | 27.84462                     | 34.83217                     | 60.67903                     | 27.62046                     | 52.65353                     | 100.11721                    |
| DPN131      | 224       | 256               | 28.58941                     | 33.01078                     | 55.65146                     | 28.33119                     | 46.19439                     | 89.24904                     |
