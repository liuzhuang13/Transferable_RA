import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import pdb
from model_sr import SimpleNet, ResNet, Discriminator, SRDenseNet
from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
import numpy as np
import json
from math import log10
import pytorch_ssim
from networks import ResnetGenerator
from data_utils import SRImageFolder, DNImageFolder, JPEGImageFolder, SelfImageFolder
from time import gmtime, strftime

# from skimage import io, color

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', default='/home/zhuangl/datasets/imagenet',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=5, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=6, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=20, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_false',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--evaluate-notransform', action='store_true', help='whether to evaluate bicubic interpolation')
parser.add_argument('--upscale', default=4, type=int) # SR up resolution
parser.add_argument('--l', default=0, type=float) # coefficient for RA loss, lambda in paper
parser.add_argument('--save-dir', default='checkpoint/default/', type=str)
parser.add_argument('--mode', default='sr',type=str) # mode, ra, ra_transformer, ra_unsupervised
parser.add_argument('--task', default='sr', type=str) #
parser.add_argument('--std', default=0.1, type=float) # noise level for denoising
parser.add_argument('--L', default=1, type=float)
parser.add_argument('--model-sr', default='test', type=str) # for evaluation 
parser.add_argument('--model-transformer', default=None, type=str) # for evaluation
parser.add_argument('--test-batch-size', default=20, type=int)
parser.add_argument('--cross-evaluate', action='store_true')
parser.add_argument('--custom-evaluate', action='store_true')
parser.add_argument('--custom-evaluate-model', default='', type=str) 
parser.add_argument('--sr-arch', default='SRResNet', type=str)
parser.add_argument('--transformer-arch', default='pix2pix', type=str)
parser.add_argument('--lower_lr', action='store_false', help='whether to lower lr every certain epochs') # default is True
parser.add_argument('--vis', action='store_true', help='whether to visualize sr results')
parser.add_argument('--l_soft', default=0.001, type=float)
# parser.add_argument('--sr_model', action='store_true', help='whether to use the SRResNet model in dn and jpeg')
best_prec1 = 0

# get high res images output by bicubic interpolation, only used in notransform test for sr
def trans_RGB_bicubic(data):
    up = args.upscale
    ims_np = (data.clone()*255.).permute(0, 2, 3, 1).numpy().astype(np.uint8)

    hr_size = ims_np.shape[1]

    lr_size = hr_size // up

    rgb_hrs = data.new().resize_(data.size(0), 3, hr_size, hr_size).zero_()

    for i, im_np in enumerate(ims_np):
        im = Image.fromarray(im_np, 'RGB')
        rgb_lr = Resize((lr_size, lr_size), Image.BICUBIC)(im)
        rgb_hr = Resize((hr_size, hr_size), Image.BICUBIC)(rgb_lr)
        rgb_hr = ToTensor()(rgb_hr)
        rgb_hrs[i].copy_(rgb_hr)
    return rgb_hrs

# normalize the output of sr, to fit cls network input
# input 0-1, output: normalized imagenet network input
def process_to_input_cls(RGB):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    RGB_new = torch.autograd.Variable(RGB.data.new(*RGB.size()))

    RGB_new[:, 0, :, :] = (RGB[:, 0, :, :] - means[0]) / stds[0]
    RGB_new[:, 1, :, :] = (RGB[:, 1, :, :] - means[1]) / stds[1]
    RGB_new[:, 2, :, :] = (RGB[:, 2, :, :] - means[2]) / stds[2]

    return RGB_new

def main():

    global args, best_prec1
    args = parser.parse_args()

    if 'small' in args.data:
        args.epochs = 30
    else:
        args.epochs = 6

    print(args)

    args.distributed = args.world_size > 1


    if args.mode == 'ra_transform':
        save_dir_extra = '_'.join([args.sr_arch, args.transformer_arch, args.arch])
    elif args.mode == 'ra_unsupervised':
        args.l = 10 
        save_dir_extra = '_'.join([args.sr_arch, str(args.l), args.arch])
    elif args.mode == 'ra':
        save_dir_extra = '_'.join([args.sr_arch, str(args.l), args.arch])

    args.save_dir = args.save_dir + save_dir_extra

    os.makedirs(args.save_dir, exist_ok=True)
    print('making directory ', args.save_dir)

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

    # if single machine multi gpus
    if not args.distributed:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
            model.eval()
        else:
            model = torch.nn.DataParallel(model).cuda() #disable multi gpu for now
            model.eval()
    else:
        return

    # two models for sr and two models for dn/jpeg
    # before 0302, default for sr is SRResNet, default for dn/jpeg is pix2pix
    if args.task == 'sr': 
        if args.sr_arch == 'SRResNet':
            model_sr = ResNet(upscale_factor=4, channel=3, residual=False)
        elif args.sr_arch == 'SRDenseNet':
            model_sr = SRDenseNet(16,16,8,8) 
    elif args.task == 'dn' or args.task == 'jpeg':
        if args.sr_arch == 'SRResNet':
            model_sr = ResNet(upscale_factor=1, channel=3, residual=False)
        elif args.sr_arch == 'pix2pix':
            model_sr = ResnetGenerator(3, 3, n_blocks=6)
    model_sr = torch.nn.DataParallel(model_sr).cuda()
    model_sr.train()

    # not using these models for now
    if args.transformer_arch == 'SRResNet':
        model_transformer = ResNet(upscale_factor=1, channel=3, residual=False)
    elif args.transformer_arch == 'pix2pix':
        model_transformer = ResnetGenerator(3, 3, n_blocks=6)

    model_transformer = torch.nn.DataParallel(model_transformer).cuda()
    model_transformer.train()


    criterion_sr = nn.MSELoss()
    criterion_sr.cuda()
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer_sr = torch.optim.Adam(model_sr.parameters(), lr=args.lr) # previous used 0.001 as default, now 0.0001
    optimizer_transformer = torch.optim.Adam(model_transformer.parameters(), lr=args.lr)

    optimizer = torch.optim.SGD(model.parameters(), 0.01,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint, not supported now
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     # std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip()])
    val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224)])

    if args.task == 'sr':
        train_dataset = SRImageFolder(traindir, train_transform)
        val_dataset = SRImageFolder(valdir, val_transform)
    elif args.task == 'dn':
        train_dataset = DNImageFolder(traindir, train_transform)
        val_dataset = DNImageFolder(valdir, val_transform, deterministic=True)
    elif args.task == 'self':
        train_dataset = SelfImageFolder(traindir, train_transform)
        val_dataset = SelfImageFolder(valdir, val_transform)
    elif args.task == 'jpeg':
        randomfoldername = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
        randomfoldername += str(os.getpid())
        train_dataset = JPEGImageFolder(traindir, train_transform, tmp_dir=args.data + '/trash/{}_{}/'.format(randomfoldername, np.random.randint(1, 1000)))
        val_dataset = JPEGImageFolder(valdir, val_transform, tmp_dir=args.data + '/trash/{}_{}/'.format(randomfoldername, np.random.randint(1, 1000)))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=None)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.test_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    run = eval(args.mode)

    # evaluation options
    if args.evaluate:
        # model_sr = torch.load(args.model_sr).cuda()
        model_sr = load_model(args.model_sr, model_sr)

        loss_sr, loss_cls, top1, top5, psnr, ssim = validate(val_loader, model_sr, model_transformer, model, optimizer_sr, criterion_sr, criterion, run=run)
        # pdb.set_trace()
        save_file = args.model_sr + '_{}.txt'.format(args.arch)
        np.savetxt(save_file, [loss_sr, loss_cls, top1, top5, psnr, ssim])
        return

    if args.custom_evaluate: # custom R
        # model_sr = torch.load(args.model_sr).cuda()
        model_sr = load_model(args.model_sr, model_sr)

        model = torch.load(args.custom_evaluate_model).cuda()
        loss_sr, loss_cls, top1, top5, psnr, ssim = validate(val_loader, model_sr, model_transformer, model, optimizer_sr, criterion_sr, criterion, run=run)
        # pdb.set_trace()
        save_file = args.model_sr + '_custom.txt'
        np.savetxt(save_file, [loss_sr, loss_cls, top1, top5, psnr, ssim])
        return

    if args.evaluate_notransform:
        os.makedirs('notransform_results/' + args.task, exist_ok=True)
        loss_sr, loss_cls, top1, top5, psnr, ssim = validate_notransform(val_loader, model, criterion_sr, criterion)
        save_file =  'notransform_results/' + args.task + '/{}.txt'.format(args.arch)
        np.savetxt(save_file, [loss_sr, loss_cls, top1, top5, psnr, ssim])
        return


# val_loader, model_sr, model_transformer, model_D, model, criterion_sr, criterion, run
    if args.cross_evaluate:
        basic_model_list = ['resnet18','resnet50','vgg16_bn', 'resnet101', 'densenet121']

        more_model_list = ['densenet169', 'densenet201', 'vgg13_bn', 'vgg19_bn']
        other_model_list = ['vgg13', 'vgg16', 'vgg19', 'inception_v3']
        # complete_model_list = ['vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'densenet169', 'densenet201', 'inception_v3']
        model_list = basic_model_list + more_model_list + other_model_list
        # model_sr = torch.load(args.model_sr).cuda()

        model_sr = load_model(args.model_sr, model_sr)

        # pdb.set_trace()
        model_sr = nn.DataParallel(model_sr)
        log = {}
        if args.model_transformer is not None:
            model_transformer = torch.load(args.model_transformer).cuda()
            model_transformer = nn.DataParallel(model_transformer)
            run=ra_transform

        for arch in basic_model_list:
            model = models.__dict__[arch](pretrained=True)
            model = torch.nn.DataParallel(model).cuda()
            loss_sr, loss_cls, top1, top5, psnr, ssim = validate(val_loader, model_sr, model_transformer, model, optimizer_sr, criterion_sr, criterion, run=run)
            # log[arch] = top1

            if isinstance(top1, torch.Tensor):
                log[arch] = top1.item()
            else:
                log[arch] = top1

            with open(args.model_sr + '_' + run.__name__ + '.txt', 'w') as outfile:
                json.dump(log, outfile)
        return 

    if args.vis:
        model_sr = load_model(args.model_sr, model_sr)
        vis(val_loader, model_sr, model_transformer, model, criterion_sr, criterion, run)
        return

    log = []

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        if args.lower_lr:
            adjust_learning_rate(optimizer_sr, epoch)
            adjust_learning_rate(optimizer_transformer, epoch)

        log_tmp = []

        # train for one epoch
        loss_sr, loss_cls, top1, top5, psnr, ssim = train(train_loader, model_sr, model_transformer, model, optimizer_sr, optimizer_transformer, criterion_sr, criterion, epoch, run=run)
        log_tmp += [loss_sr, loss_cls, top1, top5, psnr, ssim]


        # evaluate on validation set
        loss_sr, loss_cls, top1, top5, psnr, ssim = validate(val_loader, model_sr, model_transformer, model, optimizer_sr, criterion_sr, criterion, run=run)
        log_tmp += [loss_sr, loss_cls, top1, top5, psnr, ssim]

        log.append(log_tmp)
        np.savetxt(os.path.join(args.save_dir, 'log.txt'), log)

        model_sr_out_path = os.path.join(args.save_dir, "model_sr_epoch_{}.pth".format(epoch))
        torch.save(model_sr, model_sr_out_path)
        print("Checkpoint saved to {}".format(model_sr_out_path))

        if args.mode == 'ra_transform':
            model_transformer_out_path = os.path.join(args.save_dir, "model_transformer_epoch_{}.pth".format(epoch))
            torch.save(model_transformer, model_transformer_out_path)
            print("Checkpoint saved to {}".format(model_transformer_out_path))


        args.model_sr = model_sr_out_path
        vis(val_loader, model_sr, model_transformer, model, criterion_sr, criterion, run)

# Model possibly trained from an older version of Pytorch, so need this extra custom function here
def load_model(model_path, model_sr):
    load_dict = torch.load(model_path).state_dict()
    model_dict = model_sr.state_dict()
    model_dict.update(load_dict)
    model_sr.load_state_dict(model_dict)

    return model_sr

def ra(input_sr_var, target_sr_var, target_cls_var, model_sr, model_transformer, model, 
    optimizer_sr, optimizer_transformer, criterion_sr, criterion, train=True):
    if train:
        optimizer_sr.zero_grad()

    # pdb.set_trace()
    output_sr = model_sr(input_sr_var)
    # pdb.set_trace()
    loss_sr = criterion_sr(output_sr, target_sr_var)

    loss_cls = 0

    input_cls = process_to_input_cls(output_sr)
    output_cls = model(input_cls)
    loss_cls = criterion(output_cls, target_cls_var)


    # compute ssim for every image 
    ssim = 0
    # not compute during training to save time
    if not train:
        for i in range(output_sr.size(0)):
            sr_image = output_sr[i].unsqueeze(0)
            hr_image = target_sr_var[i].unsqueeze(0)
            ssim += pytorch_ssim.ssim(sr_image, hr_image).item()
        ssim = ssim / output_sr.size(0)

    loss = loss_sr + args.l * loss_cls

    if train:
        loss.backward()
        optimizer_sr.step()

    return loss_sr, loss_cls, output_cls, ssim


def ra_unsupervised(input_sr_var, target_sr_var, target_cls_var, model_sr, model_transformer, model, 
    optimizer_sr, optimizer_transformer, criterion_sr, criterion, train=True):
    if train:
        optimizer_sr.zero_grad()

    output_sr = model_sr(input_sr_var)
    loss_sr = criterion_sr(output_sr, target_sr_var)

    loss_cls = 0

    input_cls = process_to_input_cls(output_sr)
    output_cls = model(input_cls)

    output_cls_soft_target_v = model(process_to_input_cls(target_sr_var))


    output_cls_soft_target = Variable(torch.zeros(output_cls_soft_target_v.size())).cuda()
    output_cls_soft_target.data.copy_(output_cls_soft_target_v.data) # bug found, lost a "v" here.
    loss_cls = criterion_sr(nn.Softmax(dim=1)(output_cls), nn.Softmax(dim=1)(output_cls_soft_target))

    # output_cls_soft_target = 
    # loss_cls = criterion(output_cls, target_cls_var)

    # compute ssim for every image 
    ssim = 0
    # not compute during training to save time
    if not train:
        for i in range(output_sr.size(0)):
            sr_image = output_sr[i].unsqueeze(0)
            hr_image = target_sr_var[i].unsqueeze(0)
            ssim += pytorch_ssim.ssim(sr_image, hr_image).item()
        ssim = ssim / output_sr.size(0)

    loss = loss_sr + args.l * loss_cls

    if train:
        loss.backward()
        optimizer_sr.step()

    return loss_sr, loss_cls, output_cls, ssim

# in sr transform 2, sr model only optimizes sr loss.
def ra_transform(input_sr_var, target_sr_var, target_cls_var, model_sr, model_transformer, model, 
    optimizer_sr, optimizer_transformer, criterion_sr, criterion, train=True):
    if train:
        optimizer_sr.zero_grad()
        optimizer_transformer.zero_grad()

    output_sr = model_sr(input_sr_var)
    loss_sr = criterion_sr(output_sr, target_sr_var)

    if train:
        loss_sr.backward()
        optimizer_sr.step()

    loss_cls = 0

    output_sr.detach_()
    input_cls = process_to_input_cls(model_transformer(output_sr))

    output_cls = model(input_cls)
    loss_cls = criterion(output_cls, target_cls_var)
    # compute ssim for every image 
    ssim = 0
    # not compute during training to save time
    if not train:
        for i in range(output_sr.size(0)):
            sr_image = output_sr[i].unsqueeze(0)
            hr_image = target_sr_var[i].unsqueeze(0)
            ssim += pytorch_ssim.ssim(sr_image, hr_image).item()
        ssim = ssim / output_sr.size(0)

    loss = args.l * loss_cls

    if train:
        loss.backward()
        optimizer_transformer.step()

    return loss_sr, loss_cls, output_cls, ssim



def train(train_loader, model_sr, model_transformer, model, optimizer_sr, optimizer_transformer, criterion_sr, criterion, epoch, run):

    torch.manual_seed(epoch)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    run_time = AverageMeter()
    process_time = AverageMeter()
    losses = AverageMeter()
    losses_sr = AverageMeter()
    losses_cls = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    psnr_avg = AverageMeter()
    ssim_avg = AverageMeter()
    
    model_sr.train()
    # if model_transformer is not None:
    model_transformer.train()
    
    # model.eval()
    if type(model) is list: 
        for i in range(len(model)):
            model[i].eval()
            print(model[i].training)
    else:
        model.eval()

    end = time.time()

    for i, (img_input, img_output, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input_sr_var = Variable(img_input.cuda())
        target_sr_var = Variable(img_output).cuda()
        target_cls_var = Variable(target).cuda()
        target = target.cuda()

        start_run = time.time()

        loss_sr, loss_cls, output_cls, ssim = run(input_sr_var, target_sr_var, target_cls_var, model_sr, model_transformer, model, optimizer_sr, 
            optimizer_transformer, criterion_sr, criterion, train=True)

        run_time.update(time.time() - start_run)

        process_start = time.time()
        psnr = 10 * log10(1 / (loss_sr.item()))
        process_time.update(time.time() - process_start)

        prec1, prec5 = accuracy(output_cls.data, target, topk=(1, 5))
        # losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], img_input.size(0))
        top5.update(prec5[0], img_input.size(0))
        losses_sr.update(loss_sr.item(), img_input.size(0))
        losses_cls.update(loss_cls.item(), img_input.size(0))
        psnr_avg.update(psnr, img_input.size(0))
        ssim_avg.update(ssim, img_input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Process {process_time.val:.3f} ({process_time.avg:.3f})\t'
                  'Run {run_time.val:.3f} ({run_time.avg:.3f})\t'
                  # 'Loss {loss.val:.4f} ({loss.avg:.4f})\t's
                  'Loss_sr {loss_sr.val:.4f} ({loss_sr.avg:.3f})'
                  'Loss_cls {loss_cls.val:.4f} ({loss_cls.avg: .3f})'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, process_time=process_time, run_time=run_time, loss=losses, loss_sr = losses_sr, loss_cls=losses_cls, top1=top1, top5=top5))
    return losses_sr.avg, losses_cls.avg, top1.avg, top5.avg, psnr_avg.avg, ssim_avg.avg
        # pdb.set_trace()

def validate(val_loader, model_sr, model_transformer, model, optimizer_sr, criterion_sr, criterion, run):

    torch.manual_seed(1)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_sr = AverageMeter()
    losses_cls = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    psnr_avg = AverageMeter()
    ssim_avg = AverageMeter()
    
    model_sr.eval()
    if model_transformer is not None:
        model_transformer.eval()

    if type(model) is list:
        for i in range(len(model)):
            model[i].eval()
            print(model[i].training)
    else:
        model.eval()

    end = time.time()

    for i, (img_input, img_output, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input_sr_var = Variable(img_input, volatile=True).cuda()
        target_sr_var = Variable(img_output, volatile=True).cuda()
        target_cls_var = Variable(target, volatile=True).cuda()


        loss_sr, loss_cls, output_cls, ssim = run(input_sr_var, target_sr_var, target_cls_var, model_sr, model_transformer, model, optimizer_sr=optimizer_sr, 
            optimizer_transformer=None, criterion_sr=criterion_sr, criterion=criterion, train=False)

        psnr = 10 * log10(1 / (loss_sr.item() + 1e-9))

        prec1, prec5 = accuracy(output_cls.data, target, topk=(1, 5))
        # losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], img_input.size(0))
        top5.update(prec5[0], img_input.size(0))
        losses_sr.update(loss_sr.item(), img_input.size(0))
        losses_cls.update(loss_cls.item(), img_input.size(0))
        psnr_avg.update(psnr, img_input.size(0))
        ssim_avg.update(ssim, img_input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss_sr {loss_sr.val:.4f} ({loss_sr.avg:.4f})\t'
                  'Loss_cls {loss_cls.val:.4f} ({loss_cls.avg: .3f})'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'
                  'PSNR {psnr.val:.3f} ({psnr.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss_sr=losses_sr, loss_cls=losses_cls, 
                   top1=top1, top5=top5, psnr=psnr_avg))
    return losses_sr.avg, losses_cls.avg, top1.avg, top5.avg, psnr_avg.avg, ssim_avg.avg


# evaluate "no processing"
def validate_notransform(val_loader, model, criterion_sr, criterion):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_sr = AverageMeter()
    losses_cls = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    psnr_avg = AverageMeter()
    ssim_avg = AverageMeter()

    if type(model) is list:
        for i in range(len(model)):
            model[i].eval()
            print(model[i].training)
    else:
        model.eval()

    end = time.time()

    for i, (img_input, img_output, target) in enumerate(val_loader):
        # print(i)
        if True:
            target = target.cuda(async=True)
            target_sr_var = Variable(img_output).cuda()
            target_cls_var = Variable(target).cuda()

            # output of bicubic (tensor in 0-1)
            if args.task == 'sr':
                output_sr =  Variable(trans_RGB_bicubic(img_output), volatile=True).cuda()
            else:
                output_sr = Variable(img_input, volatile=True).cuda()

            # remaining is the same as in sr function
            loss_sr = criterion_sr(output_sr, target_sr_var)

            input_cls = process_to_input_cls(output_sr)
            output_cls = model(input_cls)
            loss_cls = criterion(output_cls, target_cls_var)

            ssim = 0
            for j in range(output_sr.size(0)):
                sr_image = output_sr[j].unsqueeze(0)
                hr_image = target_sr_var[j].unsqueeze(0)
                ssim += pytorch_ssim.ssim(sr_image, hr_image).item()
            ssim = ssim / output_sr.size(0)

            psnr = 10 * log10(1 / loss_sr.item())


        prec1, prec5 = accuracy(output_cls.data, target, topk=(1, 5))
        top1.update(prec1[0], img_input.size(0))
        top5.update(prec5[0], img_input.size(0))
        losses_sr.update(loss_sr.item(), img_input.size(0))
        losses_cls.update(loss_cls.item(), img_input.size(0))
        psnr_avg.update(psnr, img_input.size(0))
        ssim_avg.update(ssim, img_input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            # pdb.set_trace()
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss_sr {loss_sr.val:.4f} ({loss_sr.avg:.4f})\t'
                  'Loss_cls {loss_cls.val:.4f} ({loss_cls.avg: .3f})'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'
                  'PSNR {psnr.val:.3f} ({psnr.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss_sr=losses_sr, loss_cls=losses_cls, 
                   top1=top1, top5=top5, psnr=psnr_avg))
    return losses_sr.avg, losses_cls.avg, top1.avg, top5.avg, psnr_avg.avg, ssim_avg.avg

def vis(val_loader, model_sr, model_transformer, model, criterion_sr, criterion, run):

    torch.manual_seed(1)
    image_list = []
    image_list_input = []
    image_list_target = []
    for i, (img_input, img_output, target) in enumerate(val_loader):
        if i > 10:
            break

        input_sr_var = Variable(img_input, volatile=True).cuda()
        target_sr_var = Variable(img_output, volatile=True).cuda()
        target_cls_var = Variable(target, volatile=True).cuda()
        output_sr = model_sr(input_sr_var)
        im = image_from_RGB(output_sr[0])
        im_input = image_from_RGB(input_sr_var[0])
        im_target = image_from_RGB(target_sr_var[0])

        image_list.append(im)
        image_list_input.append(im_input)
        image_list_target.append(im_target)

    im_save = combine_image(image_list)
    im_save.save(args.model_sr + '_output.png')
    im_save_input = combine_image(image_list_input)
    im_save_input.save(args.model_sr + '_input.png')
    im_save_target = combine_image(image_list_target)
    im_save_target.save(args.model_sr + '_target.png')

    return im_save


#utilities functions 

# util functions for visualize
def image_from_RGB(out):
    # data = torch.clamp(output_sr*255., 0, 255).data
    if out.size(0) == 3:
        out = out.permute(1,2,0).cpu()
        out_img_y = out.data.numpy()
        out_img_y *= 255.0
        out_img_y = out_img_y.clip(0, 255)
        out_img_y = Image.fromarray(np.uint8(out_img_y), mode='RGB')
    elif out.size(0) == 1:
        out = out.cpu()
        out_img_y = out.data.numpy()
        out_img_y *= 255.0
        # pdb.set_trace()
        out_img_y = out_img_y.clip(0, 255)
        out_img_y = out_img_y[0]
        # pdb.set_trace()
        out_img_y = Image.fromarray(np.uint8(out_img_y), mode='L')

    # out_img_y.save('test.png')
    return out_img_y
    # pdb.set_trace()
    # pdb.set_trace()
def combine_image(images):
    # images = map(Image.open, ['Test1.png', 'Test2.png', 'Test3.png'])
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
      new_im.paste(im, (x_offset,0))
      x_offset += im.size[0]

    # new_im.save('test.png')
    return new_im

# util functions with imagenet training, not in use
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if 'small' in args.data:
        if epoch in range(20):
            lr = args.lr
        elif epoch in range(20, 25):
            lr = args.lr * 0.1
        elif epoch in range(25, 30):
            lr = args.lr * 0.01
    else:
        if epoch in [0,1,2,3]: # for emergency use, to be changed back to [0,1,2,3]
            lr = args.lr 
        elif epoch in [4]:
            lr = args.lr * 0.1
        elif epoch in [5]:
            lr = args.lr * 0.01

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
