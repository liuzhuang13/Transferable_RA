import os, time, shutil, argparse
import numpy as np
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
import pdb
from PIL import Image
import threading

class SRImageFolder(datasets.ImageFolder):

	def __init__(self, traindir, train_transform):
		super(SRImageFolder, self).__init__(traindir, train_transform)
		self.upscale = 4

	def __getitem__(self, index):

		path, target = self.imgs[index]
		img = self.loader(path)

		if self.transform is not None:
			img_output_PIL = self.transform(img)

		lr_size = img_output_PIL.size[0] // self.upscale
		img_input_PIL = transforms.Resize((lr_size, lr_size), Image.BICUBIC)(img_output_PIL)

		img_output = transforms.ToTensor()(img_output_PIL)
		img_input = transforms.ToTensor()(img_input_PIL)

		if self.target_transform is not None:
			target = self.target_transform(target)

		return img_input, img_output, target

class DNImageFolder(datasets.ImageFolder):

	def __init__(self, traindir, train_transform, deterministic=False):
		# self.lr_size = 54
		super(DNImageFolder, self).__init__(traindir, train_transform)
		self.std = 0.1
		self.deterministic = deterministic
		print("constructing DN Image folder")
		# pass

	def __getitem__(self, index):

		path, target = self.imgs[index]
		img = self.loader(path)

		# print(len(self.imgs))

		if self.transform is not None:
			img_output_PIL = self.transform(img)
			img_output = transforms.ToTensor()(img_output_PIL)

		if self.deterministic:
			torch.manual_seed(index)
		noise = torch.randn(img_output.size()) * self.std
		img_input = torch.clamp(img_output + noise, 0, 1)


		if self.target_transform is not None:
			target = self.target_transform(target)

		return img_input, img_output, target

class JPEGImageFolder(datasets.ImageFolder):

	def __init__(self, traindir, train_transform, tmp_dir):
		super(JPEGImageFolder, self).__init__(traindir, train_transform)

		self.quality = 10
		self.tmp_dir = tmp_dir
		os.makedirs(tmp_dir, exist_ok=True)


	def __getitem__(self, index):

		path, target = self.imgs[index]
		img = self.loader(path)


		if self.transform is not None:
			img_output_PIL = self.transform(img)

		img_output_PIL.save(self.tmp_dir + '{}.jpeg'.format(index), quality=self.quality)
		img_input_PIL = Image.open(self.tmp_dir + '{}.jpeg'.format(index))
		os.remove(self.tmp_dir + "{}.jpeg".format(index))

		img_output = transforms.ToTensor()(img_output_PIL)
		img_input = transforms.ToTensor()(img_input_PIL)

		if self.target_transform is not None:
			target = self.target_transform(target)

		return img_input, img_output, target

class SelfImageFolder(datasets.ImageFolder):

	def __init__(self, traindir, train_transform):
		super(SelfImageFolder, self).__init__(traindir, train_transform)

	def __getitem__(self, index):

		path, target = self.imgs[index]
		img = self.loader(path)

		if self.transform is not None:
			img_output_PIL = self.transform(img)

		img_output = transforms.ToTensor()(img_output_PIL)
		img_input = img_output + 0.

		if self.target_transform is not None:
			target = self.target_transform(target)

		return img_input, img_output, target
		# return Variable(img_input).cuda(), Variable(img_output), Variable(target)

if __name__ =='__main__':
	traindir = '/scratch/zhuangl/datasets/imagenet/train'
	train_transform = transforms.Compose([
	            transforms.Resize(256),
	            transforms.RandomCrop(224),
	            transforms.RandomHorizontalFlip(),
	        ])
	train_dataset = JPEGImageFolder(traindir, train_transform)
	train_loader = torch.utils.data.DataLoader(
	        train_dataset, batch_size=5, shuffle=True,
	        num_workers=1, pin_memory=True, sampler=None)

	for i, (img_input, img_output, target) in enumerate(train_loader):
		print(i)
		# pdb.set_trace()
# 