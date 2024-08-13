# Luma Neural Package

Deep learning models and neural network utilities of Luma

---

## Neural Layers

*luma.neural.layer 🔗*

### Convolution

| Class | Input Shape | Output Shape |
| --- | --- | --- |
| Convolution1D | $(N,C_{in},W)$ | $(N,C_{out},W)$ |
| Convolution2D | $(N,C_{in},H,W)$ | $(N,C_{out},H,W)$ |
| Convolution3D | $(N,C_{in},D,H,W)$ | $(N,C_{out},D,H,W)$ |

### Pooling

| Class | Input Shape | Output Shape |
| --- | --- | --- |
| Pooling1D | $(N,C,W_{in})$ | $(N,C,W_{in})$ |
| Pooling2D | $(N,C,H_{in},W_{in})$ | $(N,C,H_{out},W_{out})$ |
| Pooling3D | $(N,C,D_{in},H_{in},W_{in})$ | $(N,C,D_{out},H_{out},W_{out})$ |
| GlobalAvgPooling1D | $(N,C,W)$ | $(N,C,1)$ |
| GlobalAvgPooling2D | $(N,C,H,W)$ | $(N,C,1,1)$ |
| GlovalAvgPooling3D | $(N,C,D,H,W)$ | $(N,C,1,1,1)$ |
| AdaptiveAvgPooling1D | $(N,C,W_{in})$ | $(N,C,W_{out})$ |
| AdaptiveAvgPooling2D | $(N,C,H_{in},W_{in})$ | $(N,C,H_{out},W_{out})$ |
| AdaptiveAvgPooling3D | $(N,C,D_{in},H_{in},W_{in})$ | $(N,C,D_{out},H_{out},W_{out})$ |
| LpPooling1D | $(N,C,W_{in})$ | $(N,C,W_{out})$ |
| LpPooling2D | $(N,C,H_{in}, W_{in})$ | $(N,C,H_{out},W_{out})$ |
| LpPooling3D | $(N,C,D_{in},H_{in},W_{in})$ | $(N,C,D_{out},H_{out},W_{out})$ |

### Dropout

| Class | Input Shape | Output Shape |
| --- | --- | --- |
| Dropout | $(*)$ | $(*)$ |
| Dropout1D | $(N,C,W)$ | $(N,C,W)$ |
| Dropout2D | $(N,C,H,W)$ | $(N,C,H,W)$ |
| Dropout3D | $(N,C,D,H,W)$ | $(N,C,D,H,W)$ |

### Linear

| Class | Input Shape | Output Shape |
| --- | --- | --- |
| Flatten | $(N, *)$ | $(N, -1)$ |
| Dense | $(N,L_{in})$ | $(N,L_{out})$ |
| Identity | $(*)$ | $(*)$ |

### Normalization

| Class | Input Shape | Output Shape |
| --- | --- | --- |
| BatchNorm1D | $(N,C,W)$ | $(N,C,W)$ |
| BatchNorm2D | $(N,C,H,W)$ | $(N,C,H,W)$ |
| BatchNorm3D | $(N,C,D,H,W)$ | $(N,C,D,H,W)$ |
| LocalResponseNorm | $(N,C,*)$ | $(N,C,*)$ |
| LayerNorm | $(N,*)$ | $(N,*)$ |

---

## Neural Blocks

*luma.neural.block 🔗*

| Class | # of Layers | Input Shape | Output Shape |
| --- | --- | --- | --- |
| ConvBlock1D | 2~3 | $(N,C,W_{in})$ | $(N,C,W_{out})$ |
| ConvBlock2D | 2~3 | $(N,C,H_{in}, W_{in})$ | $(N,C,H_{out}, W_{out})$ |
| ConvBlock3D | 2~3 | $(N,C,D_{in},H_{in},W_{in})$ | $(N,C,D_{out},H_{out},W_{out})$ |
| DenseBlock | 2~3 | $(N,L_{in})$ | $(N,L_{out})$ |
| IncepBlock.V1 | 19 | $(N,C_{in},H_{in},W_{in})$ | $(N,C_{out},H_{out},W_{out})$ |
| IncepBlock.V2_TypeA | 22 | $(N,C_{in},H_{in},W_{in})$ | $(N,C_{out},H_{out},W_{out})$ |
| IncepBlock.V2_TypeB | 31 | $(N,C_{in},H_{in},W_{in})$ | $(N,C_{out},H_{out},W_{out})$ |
| IncepBlock.V2_TypeC | 28 | $(N,C_{in},H_{in},W_{in})$ | $(N,C_{out},H_{out},W_{out})$ |
| IncepBlock.V2_Redux | 16 | $(N,C_{in},H_{in},W_{in})$ | $(N,C_{out},H_{out},W_{out})$ |
| IncepBlock.V4_Stem | 38 | $(N,3,299,299)$ | $(N,384,35,35)$ |
| IncepBlock.V4_TypeA | 24 | $(N,384,35,35)$ | $(N,384,35,35)$ |
| IncepBlock.V4_TypeB | 33 | $(N,1024,17,17)$ | $(N,1024,17,17)$ |
| IncepBlock.V4_TypeC | 33 | $(N,1536,8,8)$ | $(N,1536,8,8)$ |
| IncepBlock.V4_ReduxA | 15 | $(N,384,35,35)$ | $(N,1024,17,17)$ |
| IncepBlock.V4_ReduxB | 21 | $(N,1024,17,17)$ | $(N,1536,8,8)$ |
| IncepResBlock.V1_Stem | 17 | $(N,3,299,299)$ | $(N,256,35,35)$ |
| IncepResBlock.V1_TypeA | 22 | $(N,256,35,35)$ | $(N,256,35,35)$ |
| IncepResBlock.V1_TypeB | 16 | $(N,896,17,17)$ | $(N,896,17,17)$ |
| IncepResBlock.V1_TypeC | 16 | $(N,1792,8,8)$ | $(N,1792,8,8)$ |
| IncepResBlock.V1_Redux | 24 | $(N,896,17,17)$ | $(N,1792,8,8)$ |
| IncepResBlock.V2_TypeA | 22 | $(N,384,35,35)$ | $(N,384,35,35)$ |
| IncepResBlock.V2_TypeB | 16 | $(N,1280,17,17)$ | $(N,1280,17,17)$ |
| IncepResBlock.V2_TypeC | 16 | $(N,2272,8,8)$ | $(N,2272,8,8)$ |
| IncepResBlock.V2_Redux | 24 | $(N,1280,17,17)$ | $(N,2272,8,8)$ |
| ResNetBlock.Basic | 7~ | $(N,C_{in},H_{in},W_{in})$ | $(N,C_{out},H_{out},W_{out})$ |
| ResNetBlock.Bottleneck | 10~ | $(N,C_{in},H_{in},W_{in})$ | $(N,C_{out},H_{out},W_{out})$ |

---

## Neural Models

*luma.neural.model 🔗*

### LeNet Series

> LeCun, Yann, et al. "Backpropagation Applied to Handwritten Zip Code Recognition." Neural Computation, vol. 1, no. 4, 1989, pp. 541-551.
> 

| Class | # of Layers | Input Shape | Weights | Biases | Total Param. | Implemented |
| --- | --- | --- | --- | --- | --- | --- |
| LeNet_1 | 6 | $(N,1,28,28)$ | 2,180 | 22 | 2,202 | ✅ |
| LeNet_4 | 8 | $(N,1,32,32)$ | 50,902 | 150 | 51,052 | ✅ |
| LeNet_5 | 10 | $(N,1,32,32)$ | 61,474 | 236 | 61,170 | ✅ |

### AlexNet Series

> Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "ImageNet Classification with Deep Convolutional Neural Networks." Advances in Neural
Information Processing Systems, 2012.
> 

| Class | # of Layers | Input Shape | Weights | Biases | Total Param. | Implemented |
| --- | --- | --- | --- | --- | --- | --- |
| AlexNet | 21 | $(N,3,227,227)$ | 62,367,776 | 10,568 | 62,378,344 | ✅ |
| ZFNet | 21 | $(N,3,227,227)$ | 58,292,000 | 9,578 | 58,301,578 | ✅ |

### VGGNet Series

> Simonyan, Karen, and Andrew Zisserman. "Very Deep Convolutional Networks for Large-Scale Image Recognition." arXiv preprint arXiv:1409.1556, 2014.
> 

| Class | # of Layers | Input Shape | Weights | Biases | Total Param. | Implemented |
| --- | --- | --- | --- | --- | --- | --- |
| VGGNet_11 | 27 | $(N,3,224,224)$ | 132,851,392 | 11,944 | 132,863,336 | ✅ |
| VGGNet_13 | 31 | $(N,3,224,224)$ | 133,035,712 | 12,136 | 133,047,848 | ✅ |
| VGGNet_16 | 37 | $(N,3,224,224)$ | 138,344,128 | 13,416 | 138,357,544 | ✅ |
| VGGNet_19 | 43 | $(N,3,224,224)$ | 143,652,544 | 14,696 | 143,667,240 | ✅ |

### InceptionNet Series

*InceptionNet-v1, v2, v3*

> Szegedy, Christian, et al. “Going Deeper with Convolutions.” Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015, pp. 1-9.
> 

| Class | # of Layers | Input Shape | Weights | Biases | Total Param. | Implemented |
| --- | --- | --- | --- | --- | --- | --- |
| InceptionNet_V1 | 182 | $(N,3,224,224)$ | 6,990,272 | 8,280 | 6,998,552 | ✅ |
| InceptionNet_V2 | 242 | $(N,3,299,299)$ | 24,974,688 | 20,136 | 24,994,824 | ✅ |
| InceptionNet_V3 | 331 | $(N,3,299,299)$ | 25,012,960 | 20,136 | 25,033,096 | ✅ |

*InceptionNet-v4, InceptionResNet-v1, v2*

> Szegedy, Christian, et al. “Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning.” Proceedings of the Thirty-First AAAI Conference on Artificial Intelligence, 2017, pp. 4278-4284.
> 

| Class | # of Layers | Input Shape | Weights | Biases | Total Param. | Implemented |
| --- | --- | --- | --- | --- | --- | --- |
| InceptionNet_V4 | 504 | $(N,3,299,299)$ | 42,641,952 | 32,584 | 42,674,536 | ✅ |
| InceptionResNet_V1 | 410 | $(N,3,299,299)$ | 21,611,648 | 33,720 | 21,645,368 | ✅ |
| InceptionResNet_V2 | 431 | $(N,3,299,299)$ | 34,112,608 | 43,562 | 34,156,170 | ✅ |

### ResNet Series

> He, Kaiming, et al. “Deep Residual Learning for Image Recognition.“ Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016, pp. 770-778.
> 

| Class | # of Layers | Input Shape | Weights | Biases | Total Param. | Implemented |
| --- | --- | --- | --- | --- | --- | --- |
| ResNet_18 | 77 | $(N,3,224,224)$ | 11,688,512 | 5,800 | 11,694,312 | ✅ |
| ResNet_34 | 149 | $(N,3,224,224)$ | 21,796,672 | 9,512 | 21,806,184 | ✅ |
| ResNet_50 | 181 | $(N,3,224,224)$ | 25,556,032 | 27,560 | 25,583,592 | ✅ |
| ResNet_101 |  |  |  |  |  | ❌ |
| ResNet_152 |  |  |  |  |  | ❌ |
| ResNet_200 |  |  |  |  |  | 🔮 |
| ResNet_269 |  |  |  |  |  | 🔮 |
| ResNet_1001 |  |  |  |  |  | 🔮 |

### MobileNet Series

*Waiting for future updates…*