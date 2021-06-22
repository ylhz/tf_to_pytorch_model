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
| [tf_inception_v3](https://github.com/ylhz/tf_to_pytorch_model/releases/download/v1.0/tf_inception_v3.npy) |    [inception_v3](https://drive.google.com/drive/folders/10cFNVEhLpCatwECA6SPB-2g0q5zZyfaw)          | 99.90 | 99.90 | 299*299 |
| [tf_inception_v4](https://github.com/ylhz/tf_to_pytorch_model/releases/download/v1.0/tf_inception_v4.npy) | [inception_v4](https://drive.google.com/drive/folders/10cFNVEhLpCatwECA6SPB-2g0q5zZyfaw) | 99.90 | 100.00 | 299*299 |
|[tf_resnet_v2_50](https://github.com/ylhz/tf_to_pytorch_model/releases/download/v1.0/tf_resnet_v2_50.npy)|[resnet_v2_50_2017_04_14](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models)|97.20|97.20|  299*299|
|[tf_resnet_v2_101](https://github.com/ylhz/tf_to_pytorch_model/releases/download/v1.0/tf_resnet_v2_101.npy)|[resnet_v2_101](https://drive.google.com/drive/folders/10cFNVEhLpCatwECA6SPB-2g0q5zZyfaw)|99.80|99.80|  299*299|
|[tf_resnet_v2_152](https://github.com/ylhz/tf_to_pytorch_model/releases/download/v1.0/tf_resnet_v2_152.npy)|[resnet_v2_152_2017_04_14](https://drive.google.com/drive/folders/10cFNVEhLpCatwECA6SPB-2g0q5zZyfaw)| 97.50 | 97.50 | 299*299 |
| [tf_inc_res_v2](https://github.com/ylhz/tf_to_pytorch_model/releases/download/v1.0/tf_inc_res_v2.npy) |[inception_resnet_v2_2016_08_30](https://drive.google.com/drive/folders/10cFNVEhLpCatwECA6SPB-2g0q5zZyfaw)|99.90| 99.90 | 299*299 |
| [tf_adv_inception_v3](https://github.com/ylhz/tf_to_pytorch_model/releases/download/v1.0/tf_adv_inception_v3.npy) |  [adv_inception_v3_rename](https://drive.google.com/drive/folders/10cFNVEhLpCatwECA6SPB-2g0q5zZyfaw)    | 100.00 | 100.00 | 299*299 |
| [tf_ens3_adv_inc_v3](https://github.com/ylhz/tf_to_pytorch_model/releases/download/v1.0/tf_ens3_adv_inc_v3.npy) |      [ens3_adv_inception_v3_rename](https://drive.google.com/drive/folders/10cFNVEhLpCatwECA6SPB-2g0q5zZyfaw)        | 99.80 | 99.80 | 299*299 |
| [tf_ens4_adv_inc_v3](https://github.com/ylhz/tf_to_pytorch_model/releases/download/v1.0/tf_ens4_adv_inc_v3.npy) |  [ens4_adv_inception_v3_rename](https://drive.google.com/drive/folders/10cFNVEhLpCatwECA6SPB-2g0q5zZyfaw)  | 99.90 | 99.90 | 299*299 |
| [tf_ens_adv_inc_res_v2](https://github.com/ylhz/tf_to_pytorch_model/releases/download/v1.0/tf_ens_adv_inc_res_v2.npy) | [ens_adv_inception_resnet_v2_rename](https://drive.google.com/drive/folders/10cFNVEhLpCatwECA6SPB-2g0q5zZyfaw) | 99.90 | 99.90 | 299*299 |


# Implementation of sample attack

white box: inception_v3

our result / paper result

("*" indicates white-box attack)

| attack method | inc_v3       | inc_v4    | resnet_v2_152 | inc_res_v2 | ens3_adv_inc_v3 | ens4_adv_inc_v3 | ens_adv_inc_res_v2 |
| ------------- | ------------ | --------- | ------------- | ---------- | --------------- | --------------- | ------------------ |
| FGSM          | 81.0/80.9*   | 37.4/38.0 | 33.0/33.1     | 33.9/33.1  | 16.9/16.8       | 15.7/15.8       | 8.2/8.3            |
| I-FGSM        | 100.0/100.0* | 30.1/29.6 | 19.4/19.4     | 21.4/20.3  | 12.0/11.7       | 12.4/12.1       | 5.5/5.5            |
| MI-FGSM       | 100.0/100.0* | 55.1/54.1 | 42.8/43.5     | 51.7/50.9  | 22.2/21.9       | 21.6/21.1       | 11.2/10.5          |
| DI-FGSM       | 99.7/99.8*   | 55.3/54.2 | 33.4/32.1     | 43.5/43.6  | 15.9/15.0       | 16.4/16.2       | 8.6/7.1            |
| TI-FGSM       | /            | /         | /             | /          | 31.2/30.8       | 31.1/30.6       | 22.9/22.7          |
| PI-FGSM       | 100.0/100.0* | 57.5/58.6 | 47.6/45.0     | 52.2/51.3  | 38.4/39.3       | 39.0/39.5       | 28.0/28.8          |



# Note !

1. If the model has aux_logits output, the output will be ```[logits, aux_logits]```. Otherwise, the output is ```[logits]```. So ```logits = model(input)[0]```.

    Models with aux_logits: 

    * tf_inception_v3, 
    * tf_inception_v4, 
    * tf_inc_res_v2, 
    * tf_adv_inception_v3, 
    * tf_ens3_adv_inc_v3, 
    * tf_ens4_adv_inc_v3.

2. Model input is best specified as required, otherwise errors may occur.

# reference

https://github.com/JHL-HUST/VT
