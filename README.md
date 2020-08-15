# FunnelAct Pytorch
Pytorch implementation of Funnel Activation (FReLU): https://arxiv.org/pdf/2007.11824.pdf

Validation results are listed below:

|        Model             | Activation |   Err@1   |   Err@5   |
| :----------------------  | :--------: | :------:  | :------:  |
|    ResNet50              |  FReLU     | **22.40** | **6.164** |

Note that from the file resnet_frelu.py you can call ResNet18, ResNet34, ResNet50, ResNet101 and ResNet152
but the weights in this repo only available for ResNet50 and I never tried to train other models,
so no guaranties there!