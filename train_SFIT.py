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
from SFIT.models.classifier_shot import ClassifierShot
from SFIT.models.cyclegan import GeneratorResNet
from SFIT.trainers import SFITTrainer
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
    num_colors = 3
    data_path = os.path.expanduser(f'~/Data/{args.dataset}')
    if args.dataset == 'digits':
        n_classes = 10
        use_src_test = True
        args.batch_size = 64
        args.id_ratio = 3e-2
        args.tv_ratio = 3e-2

        if args.source == 'svhn' and args.target == 'mnist':
            source_trans = T.Compose([T.Resize(32), T.ToTensor(), T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
            target_trans = T.Compose([T.Resize(32), T.Lambda(lambda x: x.convert("RGB")),
                                      T.ToTensor(), T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
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
            source_test_dataset = datasets.USPS(f'{data_path}/usps', train=False, download=True, transform=source_trans)
            target_train_dataset = datasets.MNIST(f'{data_path}/mnist', train=True, download=True,
                                                  transform=target_trans)
            target_test_dataset = datasets.MNIST(f'{data_path}/mnist', train=False, download=True,
                                                 transform=target_trans)
            args.arch = 'lenet'
            num_colors = 1
        elif args.source == 'mnist' and args.target == 'usps':
            source_trans = T.Compose([T.ToTensor(), T.Normalize([0.5, ], [0.5, ])])
            target_trans = T.Compose([T.ToTensor(), T.Normalize([0.5, ], [0.5, ])])
            source_test_dataset = datasets.MNIST(f'{data_path}/mnist', train=False, download=True,
                                                 transform=source_trans)
            target_train_dataset = datasets.USPS(f'{data_path}/usps', train=True, download=True, transform=target_trans)
            target_test_dataset = datasets.USPS(f'{data_path}/usps', train=False, download=True, transform=target_trans)
            args.arch = 'lenet'
            num_colors = 1
        else:
            raise Exception('digits supports mnist, mnistm, usps, svhn')
    elif args.dataset == 'office31':
        n_classes = 31
        use_src_test = False
        args.epochs_T = 15
        args.G_wait = 50
        args.epochs_G = 50
        if args.arch is None: args.arch = 'resnet50'
        train_trans = T.Compose([T.Resize([256, 256]), T.RandomCrop(224), T.RandomHorizontalFlip(), T.ToTensor(),
                                 T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), ])
        test_trans = T.Compose([T.Resize([256, 256]), T.CenterCrop(224), T.ToTensor(),
                                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), ])

        source_test_dataset = datasets.ImageFolder(f'{data_path}/{args.source}/images', transform=train_trans)
        target_train_dataset = datasets.ImageFolder(f'{data_path}/{args.target}/images', transform=train_trans)
        target_test_dataset = datasets.ImageFolder(f'{data_path}/{args.target}/images', transform=test_trans)
    elif args.dataset == 'visda':
        n_classes = 12
        use_src_test = False
        args.lr_T *= 0.1
        args.epochs_T = 5
        args.G_wait = 5
        args.epochs_G = 20
        if args.arch is None: args.arch = 'resnet101'
        args.source, args.target = 'syn', 'real'
        train_trans = T.Compose([T.Resize([256, 256]), T.RandomCrop(224), T.RandomHorizontalFlip(), T.ToTensor(),
                                 T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), ])
        test_trans = T.Compose([T.Resize([256, 256]), T.CenterCrop(224), T.ToTensor(),
                                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), ])

        source_test_dataset = datasets.ImageFolder(f'{data_path}/train', transform=train_trans)
        target_train_dataset = datasets.ImageFolder(f'{data_path}/validation', transform=train_trans)
        target_test_dataset = datasets.ImageFolder(f'{data_path}/validation', transform=test_trans)
    else:
        raise Exception('please choose dataset from [digits, office31, visda]')

    source_test_loader = DataLoader(source_test_dataset, batch_size=64, shuffle=True,
                                    num_workers=args.num_workers)
    target_train_loader = DataLoader(target_train_dataset, batch_size=args.batch_size, shuffle=True,
                                     num_workers=args.num_workers, drop_last=True)
    target_train_loader_32 = DataLoader(target_train_dataset, batch_size=32, shuffle=True,
                                        num_workers=args.num_workers, drop_last=True)
    target_test_loader = DataLoader(target_test_dataset, batch_size=64, shuffle=True,
                                    num_workers=args.num_workers)

    args.force_pretrain_G = args.force_pretrain_G or not os.path.exists(
        f'logs/SFIT/{args.dataset}/s_{args.source}/t_{args.target}/model_G_transparent.pth')
    if args.resume:
        splits = args.resume.split('_')
        args.da_setting = f'{splits[0]}'
        fname = f'{"debug_" if is_debug else ""}{args.da_setting}_R'
        # args.force_pretrain_G, args.train_G = False, False
    else:
        fname = f'{"debug_" if is_debug else ""}{args.da_setting}_'
        if args.force_pretrain_G:
            fname += 'G0'
        if args.train_G:
            fname += 'G'
    if args.retrain_T:
        fname += 'T'

    logdir = f'logs/SFIT/{args.dataset}/s_{args.source}/t_{args.target}/{fname}' \
             f'_conf{args.conf_ratio}_bn{args.bn_ratio}_channel{args.channel_ratio}_content{args.content_ratio}_' \
             f'kd{args.kd_ratio}_{datetime.datetime.today():%Y-%m-%d_%H-%M-%S}/'
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
    net_G = GeneratorResNet(num_colors=num_colors).cuda()
    net_S = ClassifierShot(n_classes, args.arch, args.bottleneck_dim, 'shot' in args.da_setting).cuda()
    net_T = ClassifierShot(n_classes, args.arch, args.bottleneck_dim, 'shot' in args.da_setting).cuda()

    optimizer_G = optim.Adam(net_G.parameters(), lr=args.lr_G)

    trainer = SFITTrainer(net_G, net_S, net_T, logdir, args, test_visda=args.dataset == 'visda')

    # source network
    fpath = f'logs/{args.da_setting}/{args.dataset}/s_{args.source}/source_model.pth'
    if os.path.exists(fpath):
        print(f'Loading source network at: {fpath}...')
        net_S.load_state_dict(torch.load(fpath))
        pass
    else:
        raise Exception
    print('Testing source network on [source]...')
    trainer.test_net_S(source_test_loader, 'src')
    print('##############################################################')
    print('Testing source network on [target]...')
    print('##############################################################')
    trainer.test_net_S(target_test_loader, 'tgt')

    # target network
    fpath = f'logs/{args.da_setting}/{args.dataset}/s_{args.source}/t_{args.target}/target_model.pth'
    if os.path.exists(fpath):
        print(f'Loading pre-trained target model at: {fpath}...')
        net_T.load_state_dict(torch.load(fpath))
    else:
        raise Exception
    print('##############################################################')
    print('Testing target model on [target]...')
    print('##############################################################')
    trainer.test(target_test_loader)

    # pre-train generator
    fpath = f'logs/SFIT/{args.dataset}/s_{args.source}/t_{args.target}/model_G_transparent.pth'
    if not args.force_pretrain_G:
        print(f'Load pre-trained Generator at: {fpath}')
        net_G.load_state_dict(torch.load(fpath))
    elif args.train_G:
        scheduler_G = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_G, args.G_wait, 1)
        for epoch in tqdm(range(1, args.G_wait + 1)):
            print('Pre-training Generator...')
            trainer.train_net_G(epoch, target_train_loader, optimizer_G, pretrain=True, scheduler=scheduler_G)
            print('Testing Generator on [target]...')
            trainer.test(target_test_loader, epoch, use_generator=True)
        torch.save(net_G.state_dict(), fpath)
    else:
        print('skip pre-training Generator')
        pass
    print('##############################################################')
    print('Testing pre-trained Generator on [target]...')
    print('##############################################################')
    trainer.test(target_test_loader, use_generator=True)

    # generator
    if args.resume:
        fpath = f'logs/SFIT/{args.dataset}/s_{args.source}/t_{args.target}/{args.resume}/model_G.pth'
        print(f'Load trained Generator at: {fpath}')
        net_G.load_state_dict(torch.load(fpath))
    elif args.train_G:
        scheduler_G = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_G, args.epochs_G, 1)
        for epoch in tqdm(range(1, args.epochs_G + 1)):
            print('Training Generator...')
            trainer.train_net_G(epoch, target_train_loader, optimizer_G, scheduler=scheduler_G)
            print('Testing Generator on [target]...')
            trainer.test(target_test_loader, epoch, use_generator=True)
            torch.save(net_G.state_dict(), os.path.join(logdir, 'model_G.pth'))
    else:
        print('skip training Generator')
        pass
    print('##############################################################')
    print('Testing Generator on [target]...')
    print('##############################################################')
    trainer.test(target_test_loader, use_generator=True)

    # retrain target network
    if args.retrain_T:
        args.lr_T *= 0.5
        if 'resnet' not in args.arch:
            optimizer_T = optim.SGD(list(net_T.base.parameters()) + list(net_T.bottleneck.parameters()),
                                    lr=args.lr_T, weight_decay=1e-3, momentum=0.9, nesterov=True)
        else:
            optimizer_T = optim.SGD([{'params': net_T.base.parameters(), 'lr': args.lr_T * 0.1},
                                     {'params': net_T.bottleneck.parameters()}],
                                    lr=args.lr_T, weight_decay=1e-3, momentum=0.9, nesterov=True)
        scheduler_T = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_T, args.epochs_T, 1)
        for epoch in tqdm(range(1, args.epochs_T + 1)):
            print('Re-training target model...')
            trainer.train_net_T(epoch, target_train_loader_32, optimizer_T, scheduler_T, use_generator=args.train_G)
            if use_src_test:
                print('Testing re-trained target model on [source]...')
                trainer.test(source_test_loader)
            print('Testing re-trained target model on [target]...')
            trainer.test(target_test_loader)
        torch.save(net_T.state_dict(), os.path.join(logdir, 'target_model_retrain.pth'))
        print('##############################################################')
        print('Testing re-trained target model on [target]...')
        print('##############################################################')
        trainer.test(target_test_loader)
    else:
        print('skip re-training target model')
        pass


if __name__ == '__main__':
    # settings
    parser = argparse.ArgumentParser(description='Train SFIT')
    parser.add_argument('-d', '--dataset', type=str, default='digits', choices=['digits', 'office31', 'visda'])
    parser.add_argument('--source', type=str)
    parser.add_argument('--target', type=str)
    parser.add_argument('-a', '--arch', type=str, default=None,
                        choices=['alexnet', 'vgg16', 'resnet18', 'resnet50', 'digits'])
    parser.add_argument('--bottleneck_dim', type=int, default=256)
    parser.add_argument('-j', '--num_workers', type=int, default=4)
    parser.add_argument('-b', '--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--da_setting', type=str, default='shot', choices=['shot', 'mmd', 'adda'])
    parser.add_argument('--force_pretrain_G', default=False, action='store_true')
    parser.add_argument('--train_G', type=str2bool, default=True)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--retrain_T', default=False, action='store_true')
    # source model
    parser.add_argument('--source_LSR', type=str2bool, default=True)
    # generator
    parser.add_argument('--mAvrgAlpha', type=float, default=1)
    parser.add_argument('--a_ratio', type=float, default=0)
    parser.add_argument('--conf_ratio', type=float, default=0)
    parser.add_argument('--div_ratio', type=float, default=0)
    parser.add_argument('--js_ratio', type=float, default=0)
    parser.add_argument('--bn_ratio', type=float, default=0)
    parser.add_argument('--style_ratio', type=float, default=0)
    parser.add_argument('--channel_ratio', type=float, default=1)
    parser.add_argument('--content_ratio', type=float, default=0)
    parser.add_argument('--id_ratio', type=float, default=0)
    parser.add_argument('--kd_ratio', type=float, default=1)
    parser.add_argument('--pixel_ratio', type=float, default=0)
    parser.add_argument('--batch_ratio', type=float, default=0)
    parser.add_argument('--tv_ratio', type=float, default=0)
    parser.add_argument('--use_channel', type=str2bool, default=True)
    parser.add_argument('--KD_T', type=float, default=1,
                        help='>1 to smooth probabilities in divergence loss, or <1 to sharpen them')
    # target model
    parser.add_argument('--thres_confidence', type=float, default=0.95)
    parser.add_argument('--T_pixel_ratio', type=float, default=0)
    parser.add_argument('--T_batch_ratio', type=float, default=0)
    parser.add_argument('--G_wait', type=int, default=10)
    parser.add_argument('--epochs_G', type=int, default=30, help='number of epochs to train')
    parser.add_argument('--epochs_T', type=int, default=30, help='number of epochs to train')
    parser.add_argument('--restart', type=int, default=1)
    parser.add_argument('--lr_G', type=float, default=3e-4, help='generator learning rate')
    parser.add_argument('--lr_T', type=float, default=1e-2, help='target model learning rate')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    args = parser.parse_args()
    main(args)
