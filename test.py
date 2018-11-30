#!/usr/bin/python
#coding:utf-8

import numpy as np
import sys, os
import time

sys.path.append(os.getcwd())


# crnn packages
import torch
from torch.autograd import Variable


# Monkey-patch because I trained with a newer version.
# This can be removed once PyTorch 0.4.x is out.
# See https://discuss.pytorch.org/t/question-about-rebuild-tensor-v2/14560
import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

import utils
import dataset
from PIL import Image
import models.crnn as crnn
import alphabets
str1 = alphabets.alphabet
print(str1)
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--images_path', type=str, default='test_images/test1.png', help='the path to your images')
opt = parser.parse_args()


# crnn params
# 3p6m_third_ac97p8.pth
crnn_model_path = 'trained_models/mixed_second_finetune_acc97p7.pth'
alphabet = str1.decode('utf-8')
nclass = len(alphabet)+1
#nclass = 6736

print(nclass)

# crnn 中
def crnn_recognition(cropped_image, model):

    converter = utils.strLabelConverter(alphabet)
  
    image = cropped_image.convert('L')

    ## 
    w = int(image.size[0] / (280 * 1.0 / 160))
    transformer = dataset.resizeNormalize((w, 32))
    image = transformer(image)
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)

    model.eval()
    preds = model(image)

    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    i = 0
    #print(preds_size.data[0])
    out = ''
    while i < preds_size.data[0]:
        if preds.data[i] is not 0:
		    out += alphabet[preds.data[i] - 1]
        i += 1
    print(out)
    #print(preds)
    #sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    #print('results: {0}'.format(sim_pred))


if __name__ == '__main__':

	# crnn network
    model = crnn.CRNN(32, 1, nclass, 256)
    if torch.cuda.is_available():
        model = model.cuda()
    print('loading pretrained model from {0}'.format(crnn_model_path))
    #
    model.load_state_dict(torch.load(crnn_model_path))
    
    started = time.time()
    ## read an image
    image = Image.open(opt.images_path)

    crnn_recognition(image, model)
    finished = time.time()
    print('elapsed time: {0}'.format(finished-started))
    