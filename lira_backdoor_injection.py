from torch import nn
import torch
from sys import path
import pickle
import os

import yaml
from easydict import EasyDict
from sklearn.model_selection import train_test_split
import numpy as np

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
import ICML_add
from classifier_models import ResNet18
from ICML_add import DDE, MBNS
from lira_trigger_generation import create_config_parser, create_paths, create_models, get_train_test_loaders
from utils.backdoor import get_target_transform
from utils.dataloader import get_dataloader, PostTensorTransform
device = 'cuda'


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
        schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizerC, args.schedulerC_milestones, args.schedulerC_lambda)

        return netC, optimizerC, schedulerC

def final_test(args, test_model_path, atkmodel, netC, target_transform, train_loader, test_loader, 
         trainepoch, writer, alpha=0.5, optimizerC=None, 
         schedulerC=None, log_prefix='Internal', epochs_per_test=1, data_transforms=None, start_epoch=1,
         clip_image=None, tmp_pth=None):
    clean_accs, poison_accs = [], []
    
    best_clean_acc, best_poison_acc = 0, 0
    print(test_model_path)
    print(tmp_pth)
    netC.load_state_dict(torch.load(test_model_path, map_location='cuda')['netC'])
    atkmodel.load_state_dict(torch.load(test_model_path, map_location='cuda')['atkmodel'])
    print(torch.load(test_model_path, map_location='cuda')['best_clean_acc'])
    correct = 0
    correct_transform = 0
    test_loss = 0
    test_transform_loss = 0
    netC.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(args.device), target.to(args.device)
            output, _ = netC(data)
            test_loss += F.cross_entropy(output, target,
                                         reduction='sum').item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            noise = atkmodel(data) * args.test_eps
            if clip_image is None:
                atkdata = torch.clamp(data + noise, 0, 1)
            else:
                atkdata = clip_image(data + noise)
            atkoutput, _ = netC(atkdata)
            test_transform_loss += F.cross_entropy(atkoutput,
                                                   target_transform(target),
                                                   reduction='sum').item()  # sum up batch loss
            atkpred = atkoutput.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct_transform += atkpred.eq(
                target_transform(target).view_as(atkpred)).sum().item()
    correct /= len(test_loader.dataset)
    correct_transform /= len(test_loader.dataset)
    print(correct, correct_transform)
    atkmodel.eval()
    
    if optimizerC is None:
        print('No optimizer, creating default SGD...')  
        optimizerC = optim.SGD(netC.parameters(), lr=args.test_lr)
    if schedulerC is None:
        print('No scheduler, creating default 100,200,300,400...')  
        schedulerC = optim.lr_scheduler.MultiStepLR(optimizerC, [100, 200, 300, 400], args.test_lr)
        
    for cepoch in range(start_epoch, trainepoch+1):
        netC.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for batch_idx, (data, target) in pbar:
            break
            bs = data.size(0)
            data, target = data.to(args.device), target.to(args.device)
            if data_transforms is not None:
                data = data_transforms(data)                                    
            optimizerC.zero_grad()
            
            output, _ = netC(data)
            loss_clean = F.cross_entropy(output, target)
            
            if alpha < 1:
                with torch.no_grad():
                    noise = atkmodel(data) * args.test_eps
                    if clip_image is None:
                        atkdata = torch.clamp(data + noise, 0, 1)
                    else:
                        atkdata = clip_image(data + noise)
                        
                    atktarget = target_transform(target)
                    if args.test_attack_portion < 1.0:
                        atkdata = atkdata[:int(args.test_attack_portion*bs)]
                        atktarget = atktarget[:int(args.test_attack_portion*bs)]
                print(len(atkdata))
                atkoutput, _ = netC(atkdata.detach())
                loss_poison = F.cross_entropy(atkoutput, atktarget.detach())
            else:
                loss_poison = torch.tensor(0.0).to(args.device)
            
            loss = alpha * loss_clean + (1-alpha) * loss_poison
            
            loss.backward()
            optimizerC.step()
            
            if batch_idx % 10 == 0 or batch_idx == (len(train_loader)-1):
                pbar.set_description(
                    'Train-{} Loss: Clean {:.5f}  Poison {:.5f}  Total {:.5f} LR={:.6f}'.format(
                        cepoch,
                        loss_clean.item(),
                        loss_poison.item(),
                        loss.item(), schedulerC.get_last_lr()[0]
                    ))
        schedulerC.step()
        if 1:# cepoch % epochs_per_test == 0 or cepoch == trainepoch-1
            net_eval_DDE = ResNet18().to(device)
            net_eval_DDE.load_state_dict(
                DDE(test_model_path, k=1, mixed_loader=train_loader, atkmodel=atkmodel, args=args))
            net_eval_MBNS = ResNet18().to(device)
            net_eval_MBNS.load_state_dict(
                MBNS(test_model_path, k=1, validate_loader=test_loader, atkmodel=atkmodel, args=args))
            correct = 0
            correct2 = 0
            correct3 = 0
            correct_transform = 0
            correct_transform2 = 0
            correct_transform3 = 0
            test_loss = 0    
            test_transform_loss = 0
            net_eval_DDE.eval()
            net_eval_MBNS.eval()
            with torch.no_grad():
                for data, target in tqdm(test_loader, desc=f'Evaluation {cepoch}'):
                    data, target = data.to(args.device), target.to(args.device)
                    output, _ = netC(data)
                    output2, _ = net_eval_DDE(data)
                    output3, _ = net_eval_MBNS(data)
                    test_loss += F.cross_entropy(output, target,
                                                 reduction='sum').item()  # sum up batch loss
                    pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                    pred2 = output2.max(1, keepdim=True)[1]
                    pred3 = output3.max(1, keepdim=True)[1]
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    correct2 += pred.eq(target.view_as(pred2)).sum().item()
                    correct3 += pred.eq(target.view_as(pred3)).sum().item()

                    noise = atkmodel(data) * args.test_eps
                    if clip_image is None:
                        atkdata = torch.clamp(data + noise, 0, 1)
                    else:
                        atkdata = clip_image(data + noise)
                    atkoutput, _ = netC(atkdata)
                    atkoutput2, _ = net_eval_DDE(atkdata)
                    atkoutput3, _ = net_eval_MBNS(atkdata)
                    test_transform_loss += F.cross_entropy(atkoutput,
                                                           target_transform(target),
                                                           reduction='sum').item()  # sum up batch loss
                    atkpred = atkoutput.max(1, keepdim=True)[1]  # get the index of the max log-probability
                    atkpred2 = atkoutput2.max(1, keepdim=True)[1]
                    atkpred3 = atkoutput3.max(1, keepdim=True)[1]
                    correct_transform += atkpred.eq(
                        target_transform(target).view_as(atkpred)).sum().item()
                    correct_transform2 += atkpred2.eq(
                        target_transform(target).view_as(atkpred2)).sum().item()
                    correct_transform3 += atkpred3.eq(
                        target_transform(target).view_as(atkpred3)).sum().item()

            test_loss /= len(test_loader.dataset)
            test_transform_loss /= len(test_loader.dataset)

            correct /= len(test_loader.dataset)
            correct_transform /= len(test_loader.dataset)
            correct2 /= len(test_loader.dataset)
            correct_transform2 /= len(test_loader.dataset)
            correct3 /= len(test_loader.dataset)
            correct_transform3 /= len(test_loader.dataset)
            
            clean_accs.append(correct)
            poison_accs.append(correct_transform)
            print(correct2, correct_transform2, correct3, correct_transform3)
            print('\n{}-Test [{}]: Loss: clean {:.4f} poison {:.4f}, '
                  'Accuracy: clean {:.4f} (best {:.4f}) poison {:.4f} (best {:.4f})'.format(
                    log_prefix, cepoch, 
                    test_loss, test_transform_loss,
                    correct, best_clean_acc, correct_transform, best_poison_acc
                ))
            # if correct > best_clean_acc or (correct > best_clean_acc-0.02 and correct_transform > best_poison_acc):
            #     best_clean_acc = correct
            #     best_poison_acc = correct_transform
            #
            #     print(f'Saving current best model in {test_model_path}')
            #     if isinstance(netC, torch.nn.DataParallel):
            #         torch.save({
            #             'atkmodel': atkmodel.module.state_dict(),
            #             'netC': netC.module.state_dict(),
            #             'optimizerC': optimizerC.state_dict(),
            #             'clean_schedulerC': schedulerC,
            #             'best_clean_acc': best_clean_acc,
            #             'best_poison_acc': best_poison_acc
            #         }, test_model_path)
            #     else:
            #         torch.save({
            #             'atkmodel': atkmodel.state_dict(),
            #             'netC': netC.state_dict(),
            #             'optimizerC': optimizerC.state_dict(),
            #             'clean_schedulerC': schedulerC,
            #             'best_clean_acc': best_clean_acc,
            #             'best_poison_acc': best_poison_acc
            #         }, test_model_path)

                
        
        # if cepoch == 1:
        #     clean_img, poison_img = data[:args.test_n_size].clone().cpu(), atkdata[:args.test_n_size].clone().cpu()
        #     residual = poison_img-clean_img
        #
        #     clean_img = F.upsample(clean_img, scale_factor=(4, 4))
        #     poison_img = F.upsample(poison_img, scale_factor=(4, 4))
        #     residual = F.upsample(residual, scale_factor=(4, 4))
        #
        #
        #     all_img = torch.cat([clean_img, residual, poison_img], 0)
        #     grid = torchvision.utils.make_grid(all_img.clone(), nrow=args.test_n_size, normalize=True)
        #
        #     torchvision.utils.save_image(
        #         grid, os.path.join(args.basepath, f'all_images_{args.test_alpha}_{args.test_eps}_{args.test_optimizer}.png'))
        #     torchvision.utils.save_image(
        #         torchvision.utils.make_grid(
        #             clean_img.clone(), nrow=args.test_n_size, normalize=True),
        #         os.path.join(args.basepath, f'clean_images_{args.test_alpha}_{args.test_eps}_{args.test_optimizer}.png'))
        #     torchvision.utils.save_image(
        #         torchvision.utils.make_grid(
        #             residual.clone(), nrow=args.test_n_size, normalize=True),
        #         os.path.join(args.basepath,  f'residual_{args.test_alpha}_{args.test_eps}_{args.test_optimizer}.png'))
        #     torchvision.utils.save_image(
        #         torchvision.utils.make_grid(
        #             poison_img.clone(), nrow=args.test_n_size, normalize=True),
        #         os.path.join(args.basepath, f'poison_images_{args.test_alpha}_{args.test_eps}_{args.test_optimizer}.png'))

    return clean_accs, poison_accs

def main():
    parser = create_config_parser()
    #parser.add_argument('--test_eps', default=None, type=float)
    #parser.add_argument('--test_alpha', default=None, type=float)
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
    
    args.basepath, args.checkpoint_path, args.bestmodel_path = basepath, checkpoint_path, bestmodel_path = create_paths(args)
    test_model_path = os.path.join(
        basepath, f'poisoned_classifier_{args.test_alpha}_{args.test_eps}_{args.test_attack_portion}_{args.test_optimizer}.ph')
    print(f'Will save test model at {test_model_path}')
    
    train_loader, test_loader, clip_image = get_train_test_loaders(args)
    post_transforms = PostTensorTransform(args).to(args.device)

    atkmodel, tgtmodel, tgtoptimizer, _, create_net = create_models(args)
    netC, optimizerC, schedulerC = get_model(args)
    
    if args.test_use_train_best:
        checkpoint = torch.load(f'{bestmodel_path}')
        print('Load atkmodel and classifier states from best training: {}'.format(bestmodel_path))
        netC.load_state_dict(checkpoint['clsmodel'], strict=True)
        atk_checkpoint = checkpoint['atkmodel']
    elif args.test_use_train_last:
        checkpoint = torch.load(f'{checkpoint_path}')
        print('Load atkmodel and classifier states from last training: {}'.format(checkpoint_path))
        netC.load_state_dict(checkpoint['clsmodel'], strict=True)
        atk_checkpoint = checkpoint['atkmodel']
    else: #also use this model for a new classifier model
        checkpoint = torch.load(f'{bestmodel_path}')
        if 'atkmodel' in checkpoint:
            atk_checkpoint = checkpoint['atkmodel'] #this is for the new changes when we save both cls and atk
        else:
            atk_checkpoint = checkpoint
        print('Use scratch clsmodel. Load atkmodel state from best training: {}'.format(bestmodel_path))
        
    target_transform = get_target_transform(args)
    
    if args.test_alpha != 1.0:
        print(f'Loading best model from {bestmodel_path}')
        atkmodel.load_state_dict(atk_checkpoint, strict=True)
    else:
        print(f'Skip loading best atk model since test_alpha=1')
    
    if args.test_optimizer == 'adam':
        print('Change optimizer to adam')
        # Optimizer
        optimizerC = torch.optim.Adam(netC.parameters(), args.test_lr, weight_decay=5e-4)

        # Scheduler
        schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizerC, args.schedulerC_milestones, args.schedulerC_lambda)
    elif args.test_optimizer == 'sgdo':
        # Optimizer
        optimizerC = torch.optim.SGD(netC.parameters(), args.test_lr, weight_decay=5e-4)

        # Scheduler
        schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizerC, args.schedulerC_milestones, args.schedulerC_lambda)

    if args.use_data_parallel:
        print('Using data parallel')
        netC = torch.nn.DataParallel(netC)
        atkmodel = torch.nn.DataParallel(atkmodel)
        
    print(netC)
    print(optimizerC)
    print(schedulerC)
    
    data_transforms = PostTensorTransform(args).to(args.device)
    print('====> Post tensor transform')
    print(data_transforms)
    checkpoint = torch.load(f'{checkpoint_path}')
    
    clean_accs, poison_accs = final_test(
        args, test_model_path, atkmodel, netC, target_transform, 
        train_loader, test_loader, trainepoch=args.test_epochs,
        writer=None, log_prefix='POISON', alpha=args.test_alpha, epochs_per_test=1, 
        optimizerC=optimizerC, schedulerC=schedulerC, data_transforms=data_transforms, clip_image=clip_image, tmp_pth=checkpoint_path)


if __name__ == '__main__':
    main()
