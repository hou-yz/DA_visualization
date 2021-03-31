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
from SFIT.trainers import DATrainerShot
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
            source_trans = T.Compose([T.Resize(32), T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            target_trans = T.Compose([T.Resize(32), T.Lambda(lambda x: x.convert("RGB")),
                                      T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
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
                                      T.ToTensor(), T.Normalize((0.5,), (0.5,))])
            target_trans = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
            source_train_dataset = datasets.USPS(f'{data_path}/usps', train=True, download=True, transform=source_trans)
            source_test_dataset = datasets.USPS(f'{data_path}/usps', train=False, download=True, transform=source_trans)
            target_train_dataset = datasets.MNIST(f'{data_path}/mnist', train=True, download=True,
                                                  transform=target_trans)
            target_test_dataset = datasets.MNIST(f'{data_path}/mnist', train=False, download=True,
                                                 transform=target_trans)
            args.arch = 'lenet'
        elif args.source == 'mnist' and args.target == 'usps':
            source_trans = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
            target_trans = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
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
        args.epochs_T = 100
        args.epochs_S = 15
        if args.arch is None: args.arch = 'resnet50'
        train_trans = T.Compose([T.Resize([256, 256]), T.RandomCrop(224), T.RandomHorizontalFlip(), T.ToTensor(),
                                 T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])
        test_trans = T.Compose([T.Resize([256, 256]), T.CenterCrop(224), T.ToTensor(),
                                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

        source_train_dataset = datasets.ImageFolder(f'{data_path}/{args.source}/images', transform=train_trans)
        source_test_dataset = datasets.ImageFolder(f'{data_path}/{args.source}/images', transform=train_trans)
        target_train_dataset = datasets.ImageFolder(f'{data_path}/{args.target}/images', transform=train_trans)
        target_test_dataset = datasets.ImageFolder(f'{data_path}/{args.target}/images', transform=test_trans)
    elif args.dataset == 'visda':
        n_classes = 12
        use_src_test = False
        args.lr_T *= 0.1
        args.lr_S *= 0.1
        args.lr_D *= 0.1
        args.epochs_T = 10
        args.epochs_S = 5
        if args.arch is None: args.arch = 'resnet101'
        args.source, args.target = 'syn', 'real'
        train_trans = T.Compose([T.Resize([256, 256]), T.RandomCrop(224), T.RandomHorizontalFlip(), T.ToTensor(),
                                 T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])
        test_trans = T.Compose([T.Resize([256, 256]), T.CenterCrop(224), T.ToTensor(),
                                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

        source_train_dataset = datasets.ImageFolder(f'{data_path}/train', transform=train_trans)
        source_test_dataset = datasets.ImageFolder(f'{data_path}/train', transform=train_trans)
        target_train_dataset = datasets.ImageFolder(f'{data_path}/validation', transform=train_trans)
        target_test_dataset = datasets.ImageFolder(f'{data_path}/validation', transform=test_trans)
    else:
        raise Exception('please choose dataset from [digits, office31, visda]')

    if args.adda:
        dirname = 'DA_adda'
        args.shot = False
    elif args.mmd:
        dirname = 'DA_mmd'
        args.shot = False
    elif args.shot:
        dirname = 'SFDA_shot'
        args.batch_size = 64
    else:
        raise Exception()

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

    logdir = f'logs/{dirname}/{args.dataset}/s_{args.source}/t_{args.target}/' \
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
    teacher = ClassifierShot(n_classes, args.arch, args.bottleneck_dim, args.shot).cuda()
    student = ClassifierShot(n_classes, args.arch, args.bottleneck_dim, args.shot).cuda()
    discriminator = Discriminator(args.bottleneck_dim).cuda()

    # optimizers
    if 'resnet' not in args.arch:
        optimizer_T = optim.SGD(teacher.parameters(), lr=args.lr_T, weight_decay=1e-3, momentum=0.9, nesterov=True)
        optimizer_S = optim.SGD(list(student.base.parameters()),  # + list(student.bottleneck.parameters()),
                                lr=args.lr_S, weight_decay=1e-3, momentum=0.9, nesterov=True)
    else:
        optimizer_T = optim.SGD([{'params': teacher.base.parameters(), 'lr': args.lr_T * 0.1},
                                 {'params': teacher.bottleneck.parameters()},
                                 {'params': teacher.classifier.parameters()}],
                                lr=args.lr_T, weight_decay=1e-3, momentum=0.9, nesterov=True)
        optimizer_S = optim.SGD([{'params': student.base.parameters(), 'lr': args.lr_S * 0.1}, ],
                                # {'params': student.bottleneck.parameters()}],
                                lr=args.lr_S, weight_decay=1e-3, momentum=0.9, nesterov=True)
    optimizer_D = optim.SGD(discriminator.parameters(), lr=args.lr_D, weight_decay=1e-3, momentum=0.9, nesterov=True)

    # schedulers
    scheduler_T = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_T, args.epochs_T, 1)
    scheduler_S = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_S, args.epochs_S, 1)
    scheduler_D = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_D, args.epochs_S, 1)

    trainer = DATrainerShot(teacher, student, discriminator, logdir, args.adda, args.mmd, args.shot, args.teacher_LSR,
                            args.dataset == 'visda')

    # teacher
    pretrain_dir = f'logs/{dirname}/{args.dataset}/s_{args.source}/source_model.pth'
    if os.path.exists(pretrain_dir) and not args.force_train_T:
        print(f'Loading Teacher at: {pretrain_dir}...')
        teacher.load_state_dict(torch.load(pretrain_dir))
        pass
    else:
        print('Training Teacher...')
        for epoch in tqdm(range(1, args.epochs_T + 1)):
            trainer.train_teacher(epoch, source_train_loader_64, optimizer_T, scheduler_T)
            if epoch % (max(args.epochs_T // 10, 1)) == 0:
                if use_src_test:
                    print('Testing Teacher on [source]...')
                    trainer.test_teacher(source_test_loader)
                print('Testing Teacher on [target]...')
                trainer.test_teacher(target_test_loader)
        torch.save(teacher.state_dict(), pretrain_dir)
        torch.save(teacher.state_dict(), logdir + 'source_model.pth')
    print('Testing Teacher on [source]...')
    trainer.test_teacher(source_test_loader)
    print('##############################################################')
    print('Testing Teacher on [target]...')
    print('##############################################################')
    trainer.test_teacher(target_test_loader)

    # student
    student_dir = f'logs/{dirname}/{args.dataset}/s_{args.source}/t_{args.target}/target_model.pth'
    print(f'Initialize Student with Teacher...')
    student.load_state_dict(teacher.state_dict())
    for epoch in tqdm(range(1, args.epochs_S + 1)):
        print('Training Student...')
        trainer.train_student(epoch, source_train_loader, target_train_loader, optimizer_S, optimizer_D,
                              [scheduler_S, scheduler_D])
        if use_src_test:
            print('Testing Student on [source]...')
            trainer.test(source_test_loader)
        print('Testing Student on [target]...')
        trainer.test(target_test_loader)
        torch.save(student.state_dict(), student_dir)
        torch.save(student.state_dict(), logdir + 'target_model.pth')
    print('##############################################################')
    print('Testing Student on [target]...')
    print('##############################################################')
    trainer.test(target_test_loader)


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
    parser.add_argument('--force_train_T', action='store_true', default=False)
    # teacher
    parser.add_argument('--teacher_LSR', type=str2bool, default=True)
    # student
    parser.add_argument('--shot', type=str2bool, default=True)
    parser.add_argument('--mmd', action='store_true', default=False)
    parser.add_argument('--adda', action='store_true', default=False)
    parser.add_argument('--epochs_T', type=int, default=30, help='number of epochs to train')
    parser.add_argument('--epochs_S', type=int, default=30, help='number of epochs to train')
    parser.add_argument('--restart', type=float, default=1)
    parser.add_argument('--lr_T', type=float, default=1e-2, help='teacher learning rate')
    parser.add_argument('--lr_S', type=float, default=1e-2, help='student learning rate')
    parser.add_argument('--lr_D', type=float, default=1e-3, help='discriminator learning rate')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    args = parser.parse_args()
    main(args)
