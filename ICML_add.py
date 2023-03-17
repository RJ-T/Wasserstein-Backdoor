from torch import nn
import torch
from sys import path
import pickle
import os

import yaml
from easydict import EasyDict
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from termcolor import colored

import pathlib
from argparse import Namespace
import argparse

import torchvision
from tensorboardX import SummaryWriter
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn.functional as F
import time
from classifier_models import ResNet18
from lira_trigger_generation import create_config_parser, create_paths, create_models, get_train_test_loaders
from utils.backdoor import get_target_transform
from utils.dataloader import get_dataloader, PostTensorTransform
device = 'cuda'
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_MIN = ((np.array([0,0,0]) - np.array(IMAGENET_DEFAULT_MEAN)) / np.array(IMAGENET_DEFAULT_STD)).min()
IMAGENET_MAX = ((np.array([1,1,1]) - np.array(IMAGENET_DEFAULT_MEAN)) / np.array(IMAGENET_DEFAULT_STD)).max()


def DDE(sd_path, k, mixed_loader, atkmodel, args):
    # 用一个Batch的混合数据去计算entropy
    # mixed_data = torch.cat((poison_data, img_data[4500:45000]))
    # mixed_label = torch.cat((poison_label, tgt_data[4500:45000])).long()
    # mixed_dataset = TensorsDataset(mixed_data, mixed_label, transforms=normalize)
    # mixed_loader = torch.utils.data.DataLoader(mixed_dataset, batch_size=500, shuffle=True)
    total_sd = torch.load(sd_path, map_location=device)
    net = ResNet18(num_classes=10, norm_layer=BatchNorm2d_ent).to(device)
    net.load_state_dict(total_sd['netC'])
    atkmodel.load_state_dict(total_sd['atkmodel'])

    net.eval()
    atkmodel.eval()
    with torch.no_grad():
        for i, data in enumerate(mixed_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            atkinput = torch.clamp(inputs + args.test_eps * atkmodel(inputs), IMAGENET_MIN, IMAGENET_MAX)
            atklabel = torch.zeros_like(labels)
            inputs = torch.cat((inputs[0:int(0.5*len(inputs))], atkinput[int(0.5*len(inputs)):])).cuda()
            labels = torch.cat((labels[0:int(0.5*len(labels))], atklabel[int(0.5*len(atklabel)):])).cuda()
            # inputs, labels = inputs.cuda(), labels.cuda()
            outputs,_ = net(inputs)
            break

    entrp = {}
    for name, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            feats = m.batch_feats
            feats = (feats - feats.mean(-1).reshape(-1, 1)) / feats.std(-1).reshape(-1, 1)
            entrp[name] = batch_entropy_2(feats)

    # 收集要删除的Neurons
    index = {}
    for name, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            entrs = entrp[name]
            idx = torch.where(entrs < (entrs.mean() - k * entrs.std()))
            index[name] = idx

    # 删除Neurons
    net_2 = ResNet18(num_classes=10).to(device)
    net_2.load_state_dict(total_sd['netC'])

    sd = net_2.state_dict()
    pruned = 0
    for name, m in net_2.named_modules():
        if name in index.keys():
            for idx in index[name]:
                sd[name + '.weight'][idx] = 0
                pruned += 1
    print(index)
    # print(pruned)
    return sd


def MBNS(sd_path, k, validate_loader, atkmodel, args):
    # 用一个Batch的干净数据去计算KL散度
    # validate_dataset = TensorsDataset(validate_data, validate_label, transforms=transform_train)
    # validate_loader = torch.utils.data.DataLoader(validate_dataset, batch_size=500, shuffle=True)
    total_sd = torch.load(sd_path, map_location=device)
    net = ResNet18(num_classes=10, norm_layer=BatchNorm2d_gau).to(device)
    net.load_state_dict(total_sd['netC'])
    atkmodel.load_state_dict(total_sd['atkmodel'])
    index = {}
    net.eval()
    with torch.no_grad():
        for data in validate_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs,_ = net(inputs)
            break
    # 收集要删除的Neurons
    for name, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            var_2 = m.running_var
            var_1 = m.batch_var
            mean_2 = m.running_mean
            mean_1 = m.batch_mean
            measure = (var_2.sqrt() / var_1.sqrt()).log() + (var_1 + (mean_1 - mean_2).pow(2)) / (2 * var_2) - 1 / 2
            idx = torch.where(measure > measure.mean() + k * measure.std())
            index[name] = idx

    # 删除Neurons
    net_2 = ResNet18(num_classes=10).to(device)
    net_2.load_state_dict(total_sd['netC'])

    sd = net_2.state_dict()
    # pruned = 0
    for name, m in net_2.named_modules():
        if name in index.keys():
            for idx in index[name]:
                sd[name + '.weight'][idx] = 0
                # pruned += 1
    # print(pruned)
    print(index)
    return sd


def batch_entropy_2(x, step_size=0.01):
    n_bars = int((x.max()-x.min())/step_size)
    # print(n_bars)
    entropy = 0
    for n in range(n_bars):
        num = ((x > x.min() + n*step_size) * (x < x.min() + (n+1)*step_size)).sum(-1)
        p = torch.true_divide(num, x.shape[-1])
        logp = -p * p.log()
        logp = torch.where(torch.isnan(logp), torch.full_like(logp, 0), logp)
        # p = p.cpu().numpy()
        # print(p)
        # print(logp)
        entropy += logp
    # print(entropy)
    return entropy


class BatchNorm2d_gau(nn.BatchNorm2d):
    def __init__(self, num_features):
        super().__init__(num_features)
        self.batch_var = 0
        self.batch_mean = 0

    def forward(self, x):
        self.batch_var = x.var((0, 2, 3))
        self.batch_mean = x.mean((0, 2, 3))
        output = super().forward(x)
        return output


class BatchNorm2d_ent(nn.BatchNorm2d):
    def __init__(self, num_features):
        super().__init__(num_features)
        self.batch_feats = []

    def forward(self, x):
        self.batch_feats = x.reshape(x.shape[0], x.shape[1], -1).max(-1)[0].permute(1, 0).reshape(x.shape[1], -1)
        output = super().forward(x)
        return output


def get_model(args, model_only=False):
    netC = None
    optimizerC = None
    schedulerC = None

    if args.dataset == "cifar10" or args.dataset == "gtsrb":
        if args.clsmodel is None or args.clsmodel == 'PreActResNet18' or args.clsmodel == 'ResNet18':
            from classifier_models import ResNet18
            netC = ResNet18(num_classes=args.num_classes).to(args.device)
        else:
            if args.clsmodel in ['vgg11', 'warping_vgg']:
                from classifier_models import vgg
                netC = vgg.VGG('VGG11', num_classes=args.num_classes).to(args.device)
            else:
                raise Exception(f"Invalid model version: {args.clsmodel}")

    if args.dataset == 'tiny-imagenet':
        if args.clsmodel is None or args.clsmodel == 'ResNet18TinyImagenet':
            from classifier_models import ResNet18TinyImagenet
            netC = ResNet18TinyImagenet().to(args.device)
        else:
            if args.clsmodel in ['vgg11', 'warping_vgg']:
                from classifier_models import vgg
                netC = vgg.VGG('VGG11', num_classes=args.num_classes, feature_dim=2048).to(args.device)
            else:
                raise Exception(f"Invalid model version: {args.clsmodel}")
    if args.dataset == 'tiny-imagenet32':
        if args.clsmodel is None:
            from classifier_models import ResNet18TinyImagenet
            netC = ResNet18TinyImagenet().to(args.device)
        else:
            if args.clsmodel in ['vgg11', 'warping_vgg']:
                from classifier_models import vgg
                netC = vgg.VGG('VGG11', num_classes=args.num_classes, feature_dim=512).to(args.device)
            else:
                raise Exception(f"Invalid model version: {args.clsmodel}")
    if args.dataset == "mnist":
        from networks.models import NetC_MNIST
        netC = NetC_MNIST().to(args.device)

    if model_only:
        return netC
    else:
        # Optimizer
        optimizerC = torch.optim.SGD(netC.parameters(), args.test_lr, momentum=0.9, weight_decay=5e-4)

        # Scheduler
        schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizerC, args.schedulerC_milestones,
                                                          args.schedulerC_lambda)

        return netC, optimizerC, schedulerC


def main():
    parser = create_config_parser()
    # parser.add_argument('--test_eps', default=None, type=float)
    # parser.add_argument('--test_alpha', default=None, type=float)
    parser.add_argument('--test_attack_portion', default=1.0, type=float)
    parser.add_argument('--test_epochs', default=200, type=int)
    parser.add_argument('--test_lr', default=None, type=float)
    parser.add_argument('--schedulerC_lambda', default=0.1, type=float)
    parser.add_argument('--schedulerC_milestones', default='100,200,300,400')
    parser.add_argument('--test_n_size', default=10)
    parser.add_argument('--test_optimizer', default='sgd')
    parser.add_argument('--test_use_train_best', default=False, action='store_true')
    parser.add_argument('--test_use_train_last', default=False, action='store_true')
    parser.add_argument('--use_data_parallel', default=False, action='store_true')

    args = parser.parse_args()

    if args.test_alpha is None:
        print(f'Defaulting test_alpha to train alpha of {args.alpha}')
        args.test_alpha = args.alpha

    if args.test_lr is None:
        print(f'Defaulting test_lr to train lr {args.lr}')
        args.test_lr = args.lr

    if args.test_eps is None:
        print(f'Defaulting test_eps to train eps {args.test_eps}')
        args.test_eps = args.eps

    args.schedulerC_milestones = [int(e) for e in args.schedulerC_milestones.split(',')]

    print('====> ARGS')
    print(args)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.basepath, args.checkpoint_path, args.bestmodel_path = basepath, checkpoint_path, bestmodel_path = create_paths(
        args)
    test_model_path = os.path.join(
        basepath,
        f'poisoned_classifier_{args.test_alpha}_{args.test_eps}_{args.test_attack_portion}_{args.test_optimizer}.ph')
    print(f'Will save test model at {test_model_path}')

    train_loader, test_loader, clip_image = get_train_test_loaders(args)
    post_transforms = PostTensorTransform(args).to(args.device)

    atkmodel, tgtmodel, tgtoptimizer, _, create_net = create_models(args)
    netC, optimizerC, schedulerC = get_model(args)
    print(test_model_path)
    total_sd = torch.load(test_model_path, map_location=device)
    netC.load_state_dict(total_sd['netC'])
    # print(netC.state_dict())
    net_eval_DDE = ResNet18().to(device)
    net_eval_DDE.load_state_dict(DDE(test_model_path, k=3, mixed_loader=train_loader, atkmodel=atkmodel, args=args))
    net_eval_MBNS = ResNet18().to(device)
    net_eval_MBNS.load_state_dict(MBNS(test_model_path, k=3, validate_loader=test_loader, atkmodel=atkmodel, args=args))

    net_eval_DDE.eval()
    net_eval_MBNS.eval()
    netC.eval()
    correct_ori = 0
    correct_poi = 0
    correct_DDE = 0
    correct_MBNS = 0
    correct_DDE_poi = 0
    correct_MBNS_poi = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            plt.imshow(inputs.cpu()[0].permute(1, 2, 0))
            plt.show()
            outputs_DDE, _ = net_eval_DDE(inputs)
            outputs_MBNS, _ = net_eval_MBNS(inputs)
            outputs_ori, _ = netC(inputs)
            out = outputs_ori.max(1, keepdim=True)
            pred_ori = outputs_ori.max(1, keepdim=True)[1]
            pred_DDE = outputs_DDE.max(1, keepdim=True)[1]
            pred_MBNS = outputs_MBNS.max(1, keepdim=True)[1]
            correct_ori += pred_ori.eq(labels.view_as(pred_ori)).sum().item()
            correct_DDE += pred_DDE.eq(labels.view_as(pred_DDE)).sum().item()
            correct_MBNS += pred_MBNS.eq(labels.view_as(pred_MBNS)).sum().item()

            noise = atkmodel(inputs)
            atkinput = torch.clamp(inputs + atkmodel(inputs) * args.test_eps, IMAGENET_MIN, IMAGENET_MAX)
            atklabel = torch.zeros_like(torch.nonzero(labels))
            atkinput = atkinput[torch.nonzero(labels)].squeeze(1)
            atklabel = atklabel.squeeze()
            outputs_DDE_poi, _ = net_eval_DDE(atkinput)
            outputs_MBNS_poi, _ = net_eval_MBNS(atkinput)
            outputs_poi, _ = netC(atkinput)
            pred_DDE_poi = outputs_DDE_poi.max(1, keepdim=True)[1]
            pred_MBNS_poi = outputs_MBNS_poi.max(1, keepdim=True)[1]
            pred_poi = outputs_poi.max(1, keepdim=True)[1]
            correct_DDE_poi += pred_DDE_poi.eq(atklabel.view_as(pred_DDE_poi)).sum().item()
            correct_MBNS_poi += pred_MBNS_poi.eq(atklabel.view_as(pred_MBNS_poi)).sum().item()
            correct_poi += pred_poi.eq(atklabel.view_as(pred_poi)).sum().item()

    acc_clean_DDE = correct_DDE / len(test_loader.dataset)
    acc_clean_MBNS = correct_MBNS / len(test_loader.dataset)
    acc_poi_DDE = correct_DDE_poi / (len(test_loader.dataset) * 0.9)
    acc_poi_MBNS = correct_MBNS_poi / (len(test_loader.dataset) * 0.9)
    acc = correct_ori / len(test_loader.dataset)
    asr = correct_poi / (len(test_loader.dataset) * 0.9)
    print(acc, asr, acc_clean_DDE, acc_poi_DDE, acc_clean_MBNS, acc_poi_MBNS)
    # if args.test_use_train_best:
    #     checkpoint = torch.load(f'{bestmodel_path}')
    #     print('Load atkmodel and classifier states from best training: {}'.format(bestmodel_path))
    #     netC.load_state_dict(checkpoint['clsmodel'], strict=True)
    #     atk_checkpoint = checkpoint['atkmodel']
    # elif args.test_use_train_last:
    #     checkpoint = torch.load(f'{checkpoint_path}')
    #     print('Load atkmodel and classifier states from last training: {}'.format(checkpoint_path))
    #     netC.load_state_dict(checkpoint['clsmodel'], strict=True)
    #     atk_checkpoint = checkpoint['atkmodel']
    # else:  # also use this model for a new classifier model
    #     checkpoint = torch.load(f'{bestmodel_path}')
    #     if 'atkmodel' in checkpoint:
    #         atk_checkpoint = checkpoint['atkmodel']  # this is for the new changes when we save both cls and atk
    #     else:
    #         atk_checkpoint = checkpoint
    #     print('Use scratch clsmodel. Load atkmodel state from best training: {}'.format(bestmodel_path))
    #
    # target_transform = get_target_transform(args)
    #
    # if args.test_alpha != 1.0:
    #     print(f'Loading best model from {bestmodel_path}')
    #     atkmodel.load_state_dict(atk_checkpoint, strict=True)
    # else:
    #     print(f'Skip loading best atk model since test_alpha=1')
    #
    # if args.test_optimizer == 'adam':
    #     print('Change optimizer to adam')
    #     # Optimizer
    #     optimizerC = torch.optim.Adam(netC.parameters(), args.test_lr, weight_decay=5e-4)
    #
    #     # Scheduler
    #     schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizerC, args.schedulerC_milestones,
    #                                                       args.schedulerC_lambda)
    # elif args.test_optimizer == 'sgdo':
    #     # Optimizer
    #     optimizerC = torch.optim.SGD(netC.parameters(), args.test_lr, weight_decay=5e-4)
    #
    #     # Scheduler
    #     schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizerC, args.schedulerC_milestones,
    #                                                       args.schedulerC_lambda)
    #
    # if args.use_data_parallel:
    #     print('Using data parallel')
    #     netC = torch.nn.DataParallel(netC)
    #     atkmodel = torch.nn.DataParallel(atkmodel)

    # netC.eval()
    # with torch.no_grad():
    #     for data, target in tqdm(test_loader, desc=f'Evaluation {cepoch}'):
    #         data, target = data.to(args.device), target.to(args.device)
    #         output, _ = netC(data)
    #         test_loss += F.cross_entropy(output, target,
    #                                      reduction='sum').item()  # sum up batch loss
    #         pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
    #         correct += pred.eq(target.view_as(pred)).sum().item()
    #
    #         noise = atkmodel(data) * args.test_eps
    #         if clip_image is None:
    #             atkdata = torch.clamp(data + noise, 0, 1)
    #         else:
    #             atkdata = clip_image(data + noise)
    #         atkoutput, _ = netC(atkdata)
    #         test_transform_loss += F.cross_entropy(atkoutput,
    #                                                target_transform(target),
    #                                                reduction='sum').item()  # sum up batch loss
    #         atkpred = atkoutput.max(1, keepdim=True)[1]  # get the index of the max log-probability
    #         correct_transform += atkpred.eq(
    #             target_transform(target).view_as(atkpred)).sum().item()


if __name__ == '__main__':
    main()
