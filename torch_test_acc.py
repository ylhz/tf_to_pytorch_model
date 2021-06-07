# encoding:utf-8
"""
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
import torch.nn.functional as F
from torch.autograd import Variable as V
from torch.autograd.gradcheck import zero_gradients
import torchvision.utils as utils
from torch.utils import data
import os
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

from torch_nets import (
    tf_inception_v3, 
    tf_inception_v4, 
    tf_resnet_v2_50, 
    # tf_resnet_v2_101, 
    tf_resnet_v2_152, 
    tf_inc_res_v2, 
    tf_adv_inception_v3, 
    tf_ens3_adv_inc_v3,
    tf_ens4_adv_inc_v3,
    tf_ens_adv_inc_res_v2,
    )


list_nets = [
    'tf_inception_v3', 
    'tf_inception_v4', 
    'tf_resnet_v2_50', 
    'tf_resnet_v2_152', 
    'tf_inc_res_v2', 
    'tf_adv_inception_v3', 
    'tf_ens3_adv_inc_v3', 
    'tf_ens4_adv_inc_v3', 
    'tf_ens_adv_inc_res_v2']

parser = argparse.ArgumentParser()

parser.add_argument('--gpu', type=str, default='0', help='The ID of GPU to use.')
parser.add_argument('--input_csv', type=str, default='/home/yuanshengming/Global_Attack/VT/dev_data/val_rs.csv', help='Input csv with images.')
# parser.add_argument('--input_dir', type=str, default='/home/yuanshengming/Global_Attack/VT/dev_data/val_rs', help='Input images.')
parser.add_argument('--input_dir', type=str, default='/home/yuanshengming/data/defenses/base_img_VT/outputs_vni_dts', help='Input images.')
parser.add_argument('--model_dir', type=str, default='/home/yuanshengming/mmdnn/net_weight/', help='Model weight directory.')

parser.add_argument("--image_width", type=int, default=299, help="Width of each input images.")
parser.add_argument("--image_height", type=int, default=299, help="Height of each input images.")
parser.add_argument("--batch_size", type=int, default=10, help="How many images process at one time.")

opt = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

# Set a random seed to ensure that the results are reproducible
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

class Normalize(nn.Module):
    """
    mode:
        'tensorflow':convert data from [0,1] to [-1,1]
        'torch':(input - mean) / std
    """
    def __init__(self, mean=0, std=1, mode='tensorflow'):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std
        self.mode = mode

    def forward(self, input):
        size = input.size()
        x = input.clone()

        if self.mode == 'tensorflow':
            x = x * 2.0 -1.0  # convert data from [0,1] to [-1,1]
        elif self.mode == 'torch':
            for i in range(size[1]):
                x[:, i] = (x[:, i] - self.mean[i]) / self.std[i]
        return x

class ImageNet(data.Dataset):
    """load data from img and csv"""
    def __init__(self, dir, csv_path, transforms = None):
        self.dir = dir 
        self.csv = pd.read_csv(csv_path)
        self.transforms = transforms
    
    def __getitem__(self, index):
        img_obj = self.csv.loc[index]
        ImageID = img_obj['filename']
        Truelabel = img_obj['label']  
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

    if net_name == 'tf_inception_v3':
        net = tf_inception_v3
    elif net_name == 'tf_inception_v4':
        net = tf_inception_v4
    elif net_name == 'tf_resnet_v2_50':
        net = tf_resnet_v2_50
    elif net_name == 'tf_resnet_v2_101':
        net = tf_resnet_v2_101
    elif net_name == 'tf_resnet_v2_152':
        net = tf_resnet_v2_152
    elif net_name == 'tf_inc_res_v2':
        net = tf_inc_res_v2     
    elif net_name == 'tf_adv_inception_v3':
        net = tf_adv_inception_v3    
    elif net_name == 'tf_ens3_adv_inc_v3':
        net = tf_ens3_adv_inc_v3    
    elif net_name == 'tf_ens4_adv_inc_v3':
        net = tf_ens4_adv_inc_v3    
    elif net_name == 'tf_ens_adv_inc_res_v2':
        net = tf_ens_adv_inc_res_v2      
    else:
        print('Wrong model name!')

    model = nn.Sequential(
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        Normalize('tensorflow'), 
        net.KitModel(model_path).eval().cuda(),)
    return model

def get_models(list_nets, model_dir):
    """load models with dict"""
    nets = {}
    for net in list_nets:
        nets[net] = get_model(net, model_dir)
    return nets

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

        # Prediction
        with torch.no_grad():
            for net in list_nets:
                logits[net] = models[net](images)
                correct_num[net] += (torch.argmax(logits[net], axis=1) == label).detach().sum().cpu()

    # Print accuracy
    for net in list_nets:
        print('{} accuracy: {:.2%}'.format(net, correct_num[net]/input_num))

if __name__ == '__main__':
    main()