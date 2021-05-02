import os

os.environ['OMP_NUM_THREADS'] = '1'
import sys
import shutil
from distutils.dir_util import copy_tree
import datetime
from tqdm import tqdm
import argparse
import numpy as np
import random
import torch
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import DataLoader
from SFIT import datasets
from SFIT.models.classifier_shot import ClassifierShot, Discriminator
from SFIT.trainers import DATrainer
from SFIT.utils.str2bool import str2bool
from SFIT.utils.logger import Logger


def main(args):
    # check if in debug mode
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace():
        print('Hmm, Big Debugger is watching me')
        is_debug = True
    else:
        print('No sys.gettrace')
        is_debug = False

    # seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.benchmark = True

    # dataset
    data_path = os.path.expanduser(f'~/Data/{args.dataset}')
    if args.dataset == 'digits':
        n_classes = 10
        use_src_test = True
        args.batch_size = 64

        if args.source == 'svhn' and args.target == 'mnist':
            source_trans = T.Compose([T.Resize(32), T.ToTensor(), T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
            target_trans = T.Compose([T.Resize(32), T.Lambda(lambda x: x.convert("RGB")),
                                      T.ToTensor(), T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
            source_train_dataset = datasets.SVHN(f'{data_path}/svhn', split='train', download=True,
                                                 transform=source_trans)
            source_test_dataset = datasets.SVHN(f'{data_path}/svhn', split='test', download=True,
                                                transform=source_trans)
            target_train_dataset = datasets.MNIST(f'{data_path}/mnist', train=True, download=True,
                                                  transform=target_trans)
            target_test_dataset = datasets.MNIST(f'{data_path}/mnist', train=False, download=True,
                                                 transform=target_trans)
            args.arch = 'dtn'
        elif args.source == 'usps' and args.target == 'mnist':
            source_trans = T.Compose([T.RandomCrop(28, padding=4), T.RandomRotation(10),
                                      T.ToTensor(), T.Normalize([0.5, ], [0.5, ])])
            target_trans = T.Compose([T.ToTensor(), T.Normalize([0.5, ], [0.5, ])])
            source_train_dataset = datasets.USPS(f'{data_path}/usps', train=True, download=True, transform=source_trans)
            source_test_dataset = datasets.USPS(f'{data_path}/usps', train=False, download=True, transform=source_trans)
            target_train_dataset = datasets.MNIST(f'{data_path}/mnist', train=True, download=True,
                                                  transform=target_trans)
            target_test_dataset = datasets.MNIST(f'{data_path}/mnist', train=False, download=True,
                                                 transform=target_trans)
            args.arch = 'lenet'
        elif args.source == 'mnist' and args.target == 'usps':
            source_trans = T.Compose([T.ToTensor(), T.Normalize([0.5, ], [0.5, ])])
            target_trans = T.Compose([T.ToTensor(), T.Normalize([0.5, ], [0.5, ])])
            source_train_dataset = datasets.MNIST(f'{data_path}/mnist', train=True, download=True,
                                                  transform=source_trans)
            source_test_dataset = datasets.MNIST(f'{data_path}/mnist', train=False, download=True,
                                                 transform=source_trans)
            target_train_dataset = datasets.USPS(f'{data_path}/usps', train=True, download=True, transform=target_trans)
            target_test_dataset = datasets.USPS(f'{data_path}/usps', train=False, download=True, transform=target_trans)
            args.arch = 'lenet'
        else:
            raise Exception('digits supports mnist, mnistm, usps, svhn')
    elif args.dataset == 'office31':
        n_classes = 31
        use_src_test = False
        args.epochs_S = 100
        args.epochs_T = 15
        if args.arch is None: args.arch = 'resnet50'
        train_trans = T.Compose([T.Resize([256, 256]), T.RandomCrop(224), T.RandomHorizontalFlip(), T.ToTensor(),
                                 T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), ])
        test_trans = T.Compose([T.Resize([256, 256]), T.CenterCrop(224), T.ToTensor(),
                                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), ])

        source_train_dataset = datasets.ImageFolder(f'{data_path}/{args.source}/images', transform=train_trans)
        source_test_dataset = datasets.ImageFolder(f'{data_path}/{args.source}/images', transform=train_trans)
        target_train_dataset = datasets.ImageFolder(f'{data_path}/{args.target}/images', transform=train_trans)
        target_test_dataset = datasets.ImageFolder(f'{data_path}/{args.target}/images', transform=test_trans)
    elif args.dataset == 'visda':
        n_classes = 12
        use_src_test = False
        args.lr_D *= 0.1
        args.lr_S *= 0.1
        args.lr_T *= 0.1
        args.epochs_S = 10
        args.epochs_T = 5
        if args.arch is None: args.arch = 'resnet101'
        args.source, args.target = 'syn', 'real'
        train_trans = T.Compose([T.Resize([256, 256]), T.RandomCrop(224), T.RandomHorizontalFlip(), T.ToTensor(),
                                 T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), ])
        test_trans = T.Compose([T.Resize([256, 256]), T.CenterCrop(224), T.ToTensor(),
                                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), ])

        source_train_dataset = datasets.ImageFolder(f'{data_path}/train', transform=train_trans)
        source_test_dataset = datasets.ImageFolder(f'{data_path}/train', transform=train_trans)
        target_train_dataset = datasets.ImageFolder(f'{data_path}/validation', transform=train_trans)
        target_test_dataset = datasets.ImageFolder(f'{data_path}/validation', transform=test_trans)
    else:
        raise Exception('please choose dataset from [digits, office31, visda]')

    if 'shot' in args.da_setting:
        args.batch_size = 64

    source_train_loader = DataLoader(source_train_dataset, batch_size=args.batch_size, shuffle=True,
                                     num_workers=args.num_workers, drop_last=True)
    source_train_loader_64 = DataLoader(source_train_dataset, batch_size=64, shuffle=True,
                                        num_workers=args.num_workers, drop_last=True)
    source_test_loader = DataLoader(source_test_dataset, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers)
    target_train_loader = DataLoader(target_train_dataset, batch_size=args.batch_size, shuffle=True,
                                     num_workers=args.num_workers, drop_last=True)
    target_test_loader = DataLoader(target_test_dataset, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers)

    logdir = f'logs/{args.da_setting}/{args.dataset}/s_{args.source}/t_{args.target}/' \
             f'{"debug_" if is_debug else ""}{datetime.datetime.today():%Y-%m-%d_%H-%M-%S}/'
    print(logdir)

    # logging
    if True:
        os.makedirs(logdir + 'imgs', exist_ok=True)
        copy_tree('./SFIT', logdir + 'scripts/SFIT')
        for script in os.listdir('.'):
            if script.split('.')[-1] == 'py':
                dst_file = os.path.join(logdir, 'scripts', os.path.basename(script))
                shutil.copyfile(script, dst_file)
        sys.stdout = Logger(os.path.join(logdir, 'log.txt'), )
    print('Settings:')
    print(vars(args))

    # model
    net_D = Discriminator(args.bottleneck_dim).cuda()
    net_S = ClassifierShot(n_classes, args.arch, args.bottleneck_dim, 'shot' in args.da_setting).cuda()
    net_T = ClassifierShot(n_classes, args.arch, args.bottleneck_dim, 'shot' in args.da_setting).cuda()

    # optimizers
    optimizer_D = optim.SGD(net_D.parameters(), lr=args.lr_D, weight_decay=1e-3, momentum=0.9, nesterov=True)
    if 'resnet' not in args.arch:
        optimizer_S = optim.SGD(net_S.parameters(), lr=args.lr_S, weight_decay=1e-3, momentum=0.9, nesterov=True)
        optimizer_T = optim.SGD(list(net_T.base.parameters()),  # + list(net_T.bottleneck.parameters()),
                                lr=args.lr_T, weight_decay=1e-3, momentum=0.9, nesterov=True)
    else:
        optimizer_S = optim.SGD([{'params': net_S.base.parameters(), 'lr': args.lr_S * 0.1},
                                 {'params': net_S.bottleneck.parameters()},
                                 {'params': net_S.classifier.parameters()}],
                                lr=args.lr_S, weight_decay=1e-3, momentum=0.9, nesterov=True)
        optimizer_T = optim.SGD([{'params': net_T.base.parameters(), 'lr': args.lr_T * 0.1}, ],
                                # {'params': net_T.bottleneck.parameters()}],
                                lr=args.lr_T, weight_decay=1e-3, momentum=0.9, nesterov=True)

    # schedulers
    scheduler_D = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_D, args.epochs_T, 1)
    scheduler_S = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_S, args.epochs_S, 1)
    scheduler_T = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_T, args.epochs_T, 1)

    trainer = DATrainer(net_D, net_S, net_T, logdir, args.da_setting, args.source_LSR,
                        args.dataset == 'visda')

    # source model
    net_S_fpath = f'logs/{args.da_setting}/{args.dataset}/s_{args.source}/source_model.pth'
    if os.path.exists(net_S_fpath) and not args.force_train_S:
        print(f'Loading source model at: {net_S_fpath}...')
        net_S.load_state_dict(torch.load(net_S_fpath))
        pass
    else:
        print('Training source model...')
        for epoch in tqdm(range(1, args.epochs_S + 1)):
            trainer.train_net_S(epoch, source_train_loader_64, optimizer_S, scheduler_S)
            if epoch % (max(args.epochs_S // 10, 1)) == 0:
                if use_src_test:
                    print('Testing source model on [source]...')
                    trainer.test_net_S(source_test_loader)
                print('Testing source model on [target]...')
                trainer.test_net_S(target_test_loader)
        torch.save(net_S.state_dict(), net_S_fpath)
        torch.save(net_S.state_dict(), logdir + 'source_model.pth')
    print('Testing source model on [source]...')
    trainer.test_net_S(source_test_loader)
    print('##############################################################')
    print('Testing source model on [target]...')
    print('##############################################################')
    trainer.test_net_S(target_test_loader)

    # target model & discriminator
    net_T_fpath = f'logs/{args.da_setting}/{args.dataset}/s_{args.source}/t_{args.target}/target_model.pth'
    print(f'Initialize target model with source model...')
    net_T.load_state_dict(net_S.state_dict())
    for epoch in tqdm(range(1, args.epochs_T + 1)):
        print('Training target model...')
        trainer.train_net_T(epoch, source_train_loader, target_train_loader, optimizer_T, optimizer_D,
                            [scheduler_T, scheduler_D])
        if use_src_test:
            print('Testing target model on [source]...')
            trainer.test_net_T(source_test_loader)
        print('Testing target model on [target]...')
        trainer.test_net_T(target_test_loader)
        torch.save(net_T.state_dict(), net_T_fpath)
        torch.save(net_T.state_dict(), logdir + 'target_model.pth')
    print('##############################################################')
    print('Testing target model on [target]...')
    print('##############################################################')
    trainer.test_net_T(target_test_loader)


if __name__ == '__main__':
    # settings
    parser = argparse.ArgumentParser(description='Train SHOT')
    parser.add_argument('-d', '--dataset', type=str, default='digits', choices=['digits', 'office31', 'visda'])
    parser.add_argument('--source', type=str)
    parser.add_argument('--target', type=str)
    parser.add_argument('-a', '--arch', type=str, default=None,
                        choices=['alexnet', 'vgg16', 'resnet18', 'resnet50', 'digits'])
    parser.add_argument('--bottleneck_dim', type=int, default=256)
    parser.add_argument('-j', '--num_workers', type=int, default=4)
    parser.add_argument('-b', '--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--force_train_S', action='store_true', default=False)
    # source model
    parser.add_argument('--source_LSR', type=str2bool, default=True)
    # target model
    parser.add_argument('--da_setting', type=str, default='shot', choices=['shot', 'mmd', 'adda'])
    parser.add_argument('--epochs_S', type=int, default=30, help='number of epochs to train')
    parser.add_argument('--epochs_T', type=int, default=30, help='number of epochs to train')
    parser.add_argument('--restart', type=float, default=1)
    parser.add_argument('--lr_D', type=float, default=1e-3, help='discriminator learning rate')
    parser.add_argument('--lr_S', type=float, default=1e-2, help='target model learning rate')
    parser.add_argument('--lr_T', type=float, default=1e-2, help='source model learning rate')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    args = parser.parse_args()
    main(args)
