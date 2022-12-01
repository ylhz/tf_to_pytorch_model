# encoding:utf-8
"""Implementation of sample attack."""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
import torch.nn.functional as F
from torch.autograd import Variable as V
# from torch.autograd.gradcheck import zero_gradients
from torch.utils import data
import os
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

from torch_nets import (
    tf2torch_inception_v3,
    tf2torch_inception_v4, 
    tf2torch_resnet_v2_50,
    tf2torch_resnet_v2_101,
    tf2torch_resnet_v2_152,
    tf2torch_inc_res_v2,
    tf2torch_adv_inception_v3,
    tf2torch_ens3_adv_inc_v3,
    tf2torch_ens4_adv_inc_v3,
    tf2torch_ens_adv_inc_res_v2,
    )


list_nets = [
    'tf2torch_inception_v3',
    'tf2torch_inception_v4',
    'tf2torch_resnet_v2_50',
    'tf2torch_resnet_v2_101',
    'tf2torch_resnet_v2_152',
    'tf2torch_inc_res_v2',
    'tf2torch_adv_inception_v3',
    'tf2torch_ens3_adv_inc_v3',
    'tf2torch_ens4_adv_inc_v3',
    'tf2torch_ens_adv_inc_res_v2'
    ]

parser = argparse.ArgumentParser()

parser.add_argument('--gpu', type=str, default='0', help='The ID of GPU to use.')
parser.add_argument('--input_csv', type=str, default='dataset/dev_dataset.csv', help='Input csv with images.')
parser.add_argument('--input_dir', type=str, default='dataset/images/', help='Input images.')
parser.add_argument('--output_dir', type=str, default='adv_img_torch/', help='Output directory with adv images.')
parser.add_argument('--model_dir', type=str, default='torch_nets_weight/', help='Model weight directory.')

parser.add_argument('--white_model', type=str, default='tf2torch_inception_v3', help='Substitution model.')

parser.add_argument("--max_epsilon", type=float, default=16.0, help="Maximum size of adversarial perturbation.")
parser.add_argument("--num_iter", type=int, default=10, help="Number of iterations.")
parser.add_argument("--batch_size", type=int, default=10, help="How many images process at one time.")
parser.add_argument("--momentum", type=float, default=1.0, help="Momentum")

opt = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu


def seed_torch(seed):
    """Set a random seed to ensure that the results are reproducible"""  
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def mkdir(path):
    """Check if the folder exists, if it does not exist, create it"""
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)


class Normalize(nn.Module):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        """
        (input - mean) / std
        ImageNet normalize:
            'tensorflow': mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
            'torch': mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        """
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, input):
        size = input.size()
        x = input.clone()
            
        for i in range(size[1]):
            x[:, i] = (x[:, i] - self.mean[i]) / self.std[i]
        return x


class ImageNet(data.Dataset):
    """load data from img and csv"""
    def __init__(self, dir, csv_path, transforms=None):
        self.dir = dir
        self.csv = pd.read_csv(csv_path)
        self.transforms = transforms

    def __getitem__(self, index):
        img_obj = self.csv.loc[index]
        ImageID = img_obj['ImageId'] + '.png'
        Truelabel = img_obj['TrueLabel']
        img_path = os.path.join(self.dir, ImageID)
        pil_img = Image.open(img_path).convert('RGB')
        if self.transforms:
            data = self.transforms(pil_img)
        else:
            data = pil_img
        return data, ImageID, Truelabel

    def __len__(self):
        return len(self.csv)


def get_model(net_name, model_dir):
    """Load converted model"""
    model_path = os.path.join(model_dir, net_name + '.npy')

    if net_name == 'tf2torch_inception_v3':
        net = tf2torch_inception_v3
    elif net_name == 'tf2torch_inception_v4':
        net = tf2torch_inception_v4
    elif net_name == 'tf2torch_resnet_v2_50':
        net = tf2torch_resnet_v2_50
    elif net_name == 'tf2torch_resnet_v2_101':
        net = tf2torch_resnet_v2_101
    elif net_name == 'tf2torch_resnet_v2_152':
        net = tf2torch_resnet_v2_152
    elif net_name == 'tf2torch_inc_res_v2':
        net = tf2torch_inc_res_v2
    elif net_name == 'tf2torch_adv_inception_v3':
        net = tf2torch_adv_inception_v3
    elif net_name == 'tf2torch_ens3_adv_inc_v3':
        net = tf2torch_ens3_adv_inc_v3
    elif net_name == 'tf2torch_ens4_adv_inc_v3':
        net = tf2torch_ens4_adv_inc_v3
    elif net_name == 'tf2torch_ens_adv_inc_res_v2':
        net = tf2torch_ens_adv_inc_res_v2
    else:
        print('Wrong model name:', net_name, '!')
        exit()

    if 'inc' in net_name:
        model = nn.Sequential(
            # Images for inception classifier are normalized to be in [-1, 1] interval.
            Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]), 
            net.KitModel(model_path, aux_logits=True).eval().cuda(),)
    else:
        model = nn.Sequential(
            # Images for inception classifier are normalized to be in [-1, 1] interval.
            Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]), 
            net.KitModel(model_path).eval().cuda(),)
    return model


def get_models(list_nets, model_dir):
    """load models with dict"""
    nets = {}
    for net in list_nets:
        nets[net] = get_model(net, model_dir)
    return nets


def save_img(images, filenames, output_dir):
    """save high quality jpeg"""
    mkdir(output_dir)
    for i, filename in enumerate(filenames):
        # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
        ndarr = images[i].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        img = Image.fromarray(ndarr)
        img.save(os.path.join(output_dir, filename))


def attack(model, img, label):
    """generate adversarial images"""
    eps = opt.max_epsilon / 255.0
    num_iter = opt.num_iter
    alpha = eps / num_iter
    momentum = opt.momentum

    noise = torch.zeros_like(img, requires_grad=True)

    old_grad = 0.0
    for i in range(num_iter):
        # zero_gradients(noise)
        x = img + noise

        output = model(x)
        loss = F.cross_entropy(output[0], label)  # logit
        loss += F.cross_entropy(output[1], label)  # aux_logit
        loss.backward()
        grad = noise.grad.data

        # MI-FGSM
        # grad = grad / torch.abs(grad).mean([1,2,3], keepdim=True)
        # grad = momentum * old_grad + grad
        # old_grad = grad

        noise = noise + alpha * torch.sign(grad)
        # Avoid out of bound
        noise = torch.clamp(noise, -eps, eps)
        x = img + noise
        x = torch.clamp(x, 0.0, 1.0)
        noise = x - img
        noise = V(noise, requires_grad=True)

    adv = img + noise.detach()
    return adv


def main():
    transforms = T.Compose([T.ToTensor()])

    # Load inputs
    inputs = ImageNet(opt.input_dir, opt.input_csv, transforms)
    data_loader = DataLoader(inputs, batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=8)
    input_num = len(inputs)

    # Create models
    models = get_models(list_nets, opt.model_dir)

    # Initialization parameters
    correct_num = {}
    logits = {}
    for net in list_nets:
        correct_num[net] = 0

    # Start iteration
    for images, filename, label in tqdm(data_loader):
        label = label.cuda()
        images = images.cuda()

        # Start Attack
        adv_img = attack(models[opt.white_model], images, label)

        # Save adversarial examples
        save_img(adv_img, filename, opt.output_dir)

        # Prediction
        with torch.no_grad():
            for net in list_nets:
                if "inc" in net:
                    logits[net] = models[net](adv_img)[0]
                else:
                    logits[net] = models[net](adv_img)
                correct_num[net] += (torch.argmax(logits[net], axis=1) != label).detach().sum().cpu()

    # Print attack success rate
    for net in list_nets:
        print('{} attack success rate: {:.2%}'.format(net, correct_num[net]/input_num))


if __name__ == '__main__':
    seed_torch(0)
    main()
