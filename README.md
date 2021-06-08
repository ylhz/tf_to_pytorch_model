# tf_to_torch_model

Language : EN | [CN](./README.cn.md) 

In this repo, we convert some common tf models used in adversarial attacks to torch models and provide the resultant models. We also provide the pytorch code for you to implement attacks, e.g., FGSM.
## File Description

data: test images

nets: Original tensorflow models.

nets_weight:  Put the original network weight file into this directory. (tensorflow)

torch_nets: Converted torch model.

torch_nets_weight: Put the converted network weight file into this directory. (PyTorch)

tf_attack.py: Sample attack method with tensorflow.

tf_test_acc.py: Test the accuracy of the original model.

torch_attack.py: Sample attack method with PyTorch.

torch_test_acc.py: Test the accuracy of the converted model.

##  Model Accuracy

The following table shows the source of the converted model and the accuracy of the model on the 1000 test pictures (selected from Imagenet) given.

| Converted model name                                         | Model source | torch Accuracy(%) | tf Accuracy(%) | input size |
| ------------------------------------------------------------ | ------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [tf_inception_v3](https://github.com/ylhz/tf_to_pytorch_model/releases/download/v1.0/tf_inception_v3.npy) |              | 99.90 | 99.90 | 299*299 |
| [tf_inception_v4](https://github.com/ylhz/tf_to_pytorch_model/releases/download/v1.0/tf_inception_v4.npy) |              | 99.90 | 100.00 | 299*299 |
|[tf_inc_res_v2](https://github.com/ylhz/tf_to_pytorch_model/releases/download/v1.0/tf_inc_res_v2.npy)|| 97.20 | 97.20 | 299*299 |
|tf_resnet_v2_50||97.20|97.20|  299*299|
|[tf_resnet_v2_152](https://github.com/ylhz/tf_to_pytorch_model/releases/download/v1.0/tf_resnet_v2_152.npy)|| 97.50 | 97.50 | 299*299 |
|tf_resnet_v2_101||99.80|99.80|  299*299|
|[tf_resnet_v2_152](https://github.com/ylhz/tf_to_pytorch_model/releases/download/v1.0/tf_resnet_v2_152.npy)|| 97.50 | 97.50 | 299*299 |
| tf_inc_res_v2 ||99.90| 99.90 | 299*299 |
| [tf_adv_inception_v3](https://github.com/ylhz/tf_to_pytorch_model/releases/download/v1.0/tf_adv_inception_v3.npy) |              | 100.00 | 100.00 | 299*299 |
| [tf_ens3_adv_inc_v3](https://github.com/ylhz/tf_to_pytorch_model/releases/download/v1.0/tf_ens3_adv_inc_v3.npy) |              | 99.80 | 99.80 | 299*299 |
| [tf_ens4_adv_inc_v3](https://github.com/ylhz/tf_to_pytorch_model/releases/download/v1.0/tf_ens4_adv_inc_v3.npy) |              | 99.90 | 99.90 | 299*299 |

# Implementation of sample attack

method: I-FGSM

white box: inception_v3

black_box: inception_resnet_v2

# Note !

Model input is best specified as required, otherwise errors may occur.

