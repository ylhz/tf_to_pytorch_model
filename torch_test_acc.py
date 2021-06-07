# encoding:utf-8
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

from torch_nets import tf_inception_v3, tf_inc_res_v2, tf_resnet_v2_50, tf_resnet_v2_152, tf_adv_inception_v3, tf_ens4_adv_inc_v3

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

def get_models(net, net_name, model_dir):
    """Load converted model"""
    model_path = model_dir + net_name + '.npy'
    model = nn.Sequential(
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        Normalize('tensorflow'), 
        net.KitModel(model_path).eval().cuda(),)
    return model

def main():
    transforms = T.Compose([T.ToTensor()])

    # Load inputs and models
    inputs = ImageNet(opt.input_dir, opt.input_csv, transforms)
    data_loader = DataLoader(inputs, batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=8)
    input_num = len(inputs)

    # Create models
    inc_res_v2 = get_models(tf_inc_res_v2, 'tf_inc_res_v2', opt.model_dir)
    inc_v3 = get_models(tf_inception_v3, 'tf_inception_v3', opt.model_dir)
    res_v2_50 = get_models(tf_resnet_v2_50, 'tf_resnet_v2_50', opt.model_dir)
    res_v2_152 = get_models(tf_resnet_v2_152, 'tf_resnet_v2_152', opt.model_dir)
    adv_inc_v3 = get_models(tf_adv_inception_v3, 'tf_adv_inception_v3', opt.model_dir)
    ens4_adv_inc_v3 = get_models(tf_ens4_adv_inc_v3, 'tf_ens4_adv_inc_v3', opt.model_dir)

    # Start iteration
    inc_res_v2_num, inc_v3_num = 0, 0
    res_v2_50_num = 0
    res_v2_152_num = 0
    adv_inc_v3_num = 0
    ens4_adv_inc_v3_num = 0
    for images, filename, label in tqdm(data_loader):
        label = label.cuda()
        images = images.cuda()

        # Compute accuracy
        with torch.no_grad():
            inc_res_v2_logits = inc_res_v2(images)
            inc_v3_logits = inc_v3(images)
            res_v2_50_logits = res_v2_50(images)
            res_v2_152_logits = res_v2_152(images)
            adv_inc_v3_logits = adv_inc_v3(images)
            ens4_adv_inc_v3_logits = ens4_adv_inc_v3(images)

            inc_res_v2_num += (torch.argmax(inc_res_v2_logits, axis=1) == label).detach().sum().cpu()
            inc_v3_num += (torch.argmax(inc_v3_logits, axis=1) == label).detach().sum().cpu()
            res_v2_50_num += (torch.argmax(res_v2_50_logits, axis=1) == label).detach().sum().cpu()
            res_v2_152_num += (torch.argmax(res_v2_152_logits, axis=1) == label).detach().sum().cpu()
            adv_inc_v3_num += (torch.argmax(adv_inc_v3_logits, axis=1) == label).detach().sum().cpu()
            ens4_adv_inc_v3_num += (torch.argmax(ens4_adv_inc_v3_logits, axis=1) == label).detach().sum().cpu()

            # print('True:',label,'\nPred:', torch.argmax(inc_res_v2_logits, axis=1))

    print('inc_v3 accuracy:',inc_v3_num/input_num)
    print('inc_res_v2 accuracy:',inc_res_v2_num/input_num)
    print('res_v2_50 accuracy:',res_v2_50_num/input_num)
    print('res_v2_152 accuracy:',res_v2_152_num/input_num)
    print('adv_inc_v3 accuracy:',adv_inc_v3_num/input_num)
    print('ens4_adv_inc_v3 accuracy:',ens4_adv_inc_v3_num/input_num)

if __name__ == '__main__':
    main()