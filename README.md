# tf_to_torch_model 

In this repo, we convert some common Tensorflow models used in adversarial attacks to PyTorch models and provide the resultant models. 
Since these models are converted from their Tensorflow version, the inputs need the same normalization, i.e., [-1,1]. We have already done this, so you can use it directly.
```python
model = nn.Sequential(
    # Images for inception classifier are normalized to be in [-1, 1] interval.
    Normalize('tensorflow'), 
    net.KitModel(model_path).eval().cuda())
```
We also provide the PyTorch code for you to implement attacks on our converted models, e.g., I-FGSM (run the following command):
```python
python torch_attack.py
````


## File Description

dataset: Test images.

nets: Original tensorflow models.

nets_weight: Put the original Tensorflow network weight file into this directory.

torch_nets: Converted torch model. 

torch_nets_weight: Put the converted Pytorch network weight file into this directory. (You can find them in **[Releases](https://github.com/ylhz/tf_to_pytorch_model/releases)**)

tf_attack.py: Sample attack method with tensorflow.

torch_attack.py: Sample attack method with PyTorch.

##  Model Accuracy

The following table shows the source of the converted model and the accuracy of the model on the 1000 test pictures (selected from Imagenet) given.

| Converted model                                         | Model source | torch Accuracy(%) | tf Accuracy(%) | input size |
| ------------------------------------------------------------ | ------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [tf_inception_v3](https://github.com/ylhz/tf_to_pytorch_model/releases/download/v1.0/tf_inception_v3.npy) | [inception_v3_2016_08_28](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models) | 96.20 | 96.20 | 299*299 |
| [tf_inception_v4](https://github.com/ylhz/tf_to_pytorch_model/releases/download/v1.0/tf_inception_v4.npy) | [inception_v4_2016_09_09](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models) | 97.40 | 97.40 | 299*299 |
|[tf_resnet_v2_50](https://github.com/ylhz/tf_to_pytorch_model/releases/download/v1.0/tf_resnet_v2_50.npy)|[resnet_v2_50_2017_04_14](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models)|94.90|94.90|  299*299|
|[tf_resnet_v2_101](https://github.com/ylhz/tf_to_pytorch_model/releases/download/v1.0/tf_resnet_v2_101.npy)|[resnet_v2_101_2017_04_14](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models)|96.30|96.30|  299*299|
|[tf_resnet_v2_152](https://github.com/ylhz/tf_to_pytorch_model/releases/download/v1.0/tf_resnet_v2_152.npy)|[resnet_v2_152_2017_04_14](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models)| 95.80 | 95.80 | 299*299 |
| [tf_inc_res_v2](https://github.com/ylhz/tf_to_pytorch_model/releases/download/v1.0/tf_inc_res_v2.npy) |[inception_resnet_v2_2016_08_30](https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models)|99.80| 99.80 | 299*299 |
| [tf_adv_inception_v3](https://github.com/ylhz/tf_to_pytorch_model/releases/download/v1.0/tf_adv_inception_v3.npy) | [adv_inception_v3_2017_08_18](https://github.com/tensorflow/models/tree/archive/research/adv_imagenet_models#available-models) | 94.90 | 94.90 | 299*299 |
| [tf_ens3_adv_inc_v3](https://github.com/ylhz/tf_to_pytorch_model/releases/download/v1.0/tf_ens3_adv_inc_v3.npy) | [ens3_adv_inception_v3_2017_08_18](https://github.com/tensorflow/models/tree/archive/research/adv_imagenet_models#available-models) | 93.70 | 93.70 | 299*299 |
| [tf_ens4_adv_inc_v3](https://github.com/ylhz/tf_to_pytorch_model/releases/download/v1.0/tf_ens4_adv_inc_v3.npy) |  [ens4_adv_inception_v3_2017_08_18](https://github.com/tensorflow/models/tree/archive/research/adv_imagenet_models#available-models)  | 91.60 | 91.60 | 299*299 |
| [tf_ens_adv_inc_res_v2](https://github.com/ylhz/tf_to_pytorch_model/releases/download/v1.0/tf_ens_adv_inc_res_v2.npy) | [ens_adv_inception_resnet_v2_2017_08_18](https://github.com/tensorflow/models/tree/archive/research/adv_imagenet_models#available-models) | 97.60 | 97.60 | 299*299 |


# Implementation of sample attack

This table shows our result / paper result ("*" indicates white-box attack). The paper result is from [Patch-wise Attack for Fooling Deep Neural Network](http://arxiv.org/abs/2007.06765), and we can see that we have obtained similar results with the converted model. The specific parameter settings can be found in the paper. 



| attack method | inc_v3*       | inc_v4    | resnet_v2_152 | inc_res_v2 | ens3_adv_inc_v3 | ens4_adv_inc_v3 | ens_adv_inc_res_v2 |
| ------------- | ------------ | --------- | ------------- | ---------- | --------------- | --------------- | ------------------ |
| FGSM          | 81.0/80.9   | 37.4/38.0 | 33.0/33.1     | 33.9/33.1  | 16.9/16.8       | 15.7/15.8       | 8.2/8.3            |
| I-FGSM        | 100.0/100.0 | 30.1/29.6 | 19.4/19.4     | 21.4/20.3  | 12.0/11.7       | 12.4/12.1       | 5.5/5.5            |
| MI-FGSM       | 100.0/100.0 | 55.1/54.1 | 42.8/43.5     | 51.7/50.9  | 22.2/21.9       | 21.6/21.1       | 11.2/10.5          |
| DI-FGSM       | 99.7/99.8   | 55.3/54.2 | 33.4/32.1     | 43.5/43.6  | 15.9/15.0       | 16.4/16.2       | 8.6/7.1            |
| TI-FGSM       | /            | /         | /             | /          | 31.2/30.8       | 31.1/30.6       | 22.9/22.7          |
| PI-FGSM       | 100.0/100.0 | 57.5/58.6 | 47.6/45.0     | 52.2/51.3  | 38.4/39.3       | 39.0/39.5       | 28.0/28.8          |



# Note !

1. If the model has aux_logits output, the output will be ```[logits, aux_logits]```. Otherwise, the output is ```[logits]```. So ```logits = model(input)[0]```.

    Models with aux_logits: 

    * tf_inception_v3, 
    * tf_inception_v4, 
    * tf_inc_res_v2, 
    * tf_adv_inception_v3, 
    * tf_ens3_adv_inc_v3, 
    * tf_ens4_adv_inc_v3.

