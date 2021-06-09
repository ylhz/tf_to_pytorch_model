# encoding:utf-8
"""Implementation of sample attack."""
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

from torch_nets import tf_inception_v3, tf_inc_res_v2

parser = argparse.ArgumentParser()

parser.add_argument('--gpu', type=str, default='0', help='The ID of GPU to use.')
parser.add_argument('--input_csv', type=str, default='./data/val_rs.csv', help='Input csv with images.')
parser.add_argument('--input_dir', type=str, default='./data/val_rs', help='Input images.')
parser.add_argument('--output_dir', type=str, default='adv_img_torch/', help='Output directory with adv images.')
parser.add_argument('--model_dir', type=str, default='./torch_nets_weight/', help='Model weight directory.')

parser.add_argument("--max_epsilon", type=float, default=16.0, help="Maximum size of adversarial perturbation.")
parser.add_argument("--num_iter", type=int, default=10, help="Number of iterations.")
parser.add_argument("--image_width", type=int, default=299, help="Width of each input images.")
parser.add_argument("--image_height", type=int, default=299, help="Height of each input images.")
parser.add_argument("--batch_size", type=int, default=20, help="How many images process at one time.")
parser.add_argument("--momentum", type=float, default=1.0, help="Momentum")

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
    def __init__(self, mean=0, std=0, mode='tensorflow'):
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
        net.KitModel(model_path),
        )

    return model

def save_img(adv_img, img_ID, adv_dir):
    for i in range(adv_img.shape[0]):
        utils.save_image(adv_img[i], adv_dir + img_ID[i])

def attack(model, img, label):
    """generate adversarial images"""
    eps = opt.max_epsilon / 255.0
    num_iter = opt.num_iter
    alpha = eps / num_iter
    momentum = opt.momentum

    noise = torch.zeros_like(img, requires_grad=True)
    
    old_grad = 0.0
    for i in range(num_iter):
        zero_gradients(noise)
        x = img + noise
        
        output = model(x)
        loss = F.cross_entropy(output[0], label)  # 集成logit
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
        noise = V(noise, requires_grad = True)

    adv = img + noise.detach()
    return adv

def main():

    transforms = T.Compose([T.ToTensor()])

    # Load inputs and models
    inputs = ImageNet(opt.input_dir, opt.input_csv, transforms)
    data_loader = DataLoader(inputs, batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=8)
    input_num = len(inputs)
    
    # Create models
    inc_res_v2 = get_models(tf_inc_res_v2, 'tf_inc_res_v2', opt.model_dir).eval().cuda()
    inc_v3 = get_models(tf_inception_v3, 'tf_inception_v3', opt.model_dir).eval().cuda()

    # Start iteration
    inc_res_v2_error_num, inc_v3_error_num = 0, 0
    for images, filename, label in tqdm(data_loader):
        label = label.cuda()
        images = images.cuda()
        
        # Start Attack
        adv_img = attack(inc_v3, images, label)

        # Save adversarial examples
        save_img(adv_img, filename, opt.output_dir)

        # Compute accuracy
        with torch.no_grad():
            inc_res_v2_logits = inc_res_v2(adv_img)
            inc_v3_logits = inc_v3(adv_img)

            inc_res_v2_error_num += (torch.argmax(inc_res_v2_logits[0], axis=1) != label).detach().sum().cpu()
            inc_v3_error_num += (torch.argmax(inc_v3_logits[0], axis=1) != label).detach().sum().cpu()
            # print('True:',label,'\nPred:', torch.argmax(inc_res_v2_logits, axis=1))

    print('(white box)inc_v3 attack success rate:',inc_v3_error_num/input_num)
    print('(black box)inc_res_v2 attack success rate:',inc_res_v2_error_num/input_num)

if __name__ == '__main__':
    main()