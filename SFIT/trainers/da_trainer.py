import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from SFIT.loss import *
from SFIT.utils.meters import AverageMeter


class DATrainer(object):
    def __init__(self, teacher, student, discriminator, logdir, da_setting, teacher_LSR=True,
                 test_visda=False):
        super(DATrainer, self).__init__()
        self.teacher = teacher
        self.student = student
        self.discriminator = discriminator
        self.CE_loss = nn.CrossEntropyLoss()
        self.H_loss = HLoss()
        self.LSR_loss = LabelSmoothLoss()
        self.D_loss = nn.BCEWithLogitsLoss()
        self.MMD_loss = MMDLoss()
        self.da_setting = da_setting
        self.teacher_LSR = teacher_LSR
        self.logdir = logdir
        self.test_visda = test_visda

    def train_teacher(self, epoch, data_loader, optimizer, scheduler=None, log_interval=1000):
        self.teacher.train()
        losses, correct, miss = 0, 0, 0
        t0 = time.time()
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.cuda(), target.cuda()
            if data.size(0) == 1:
                continue
            output = self.teacher(data)
            pred = torch.argmax(output, 1)
            correct += pred.eq(target).sum().item()
            miss += target.shape[0] - pred.eq(target).sum().item()
            if self.teacher_LSR and 'shot' in self.da_setting:
                loss = self.LSR_loss(output, target)
            else:
                loss = self.CE_loss(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses += loss.item()
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                    scheduler.step(epoch - 1 + batch_idx / len(data_loader))
                elif isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    scheduler.step()
            if (batch_idx + 1) % log_interval == 0:
                # print(cyclic_scheduler.last_epoch, optimizer.param_groups[0]['lr'])
                t1 = time.time()
                t_epoch = t1 - t0
                print('Train Epoch: {}, Batch:{}, \tLoss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
                    epoch, (batch_idx + 1), losses / (batch_idx + 1), 100. * correct / (correct + miss), t_epoch))

        t1 = time.time()
        t_epoch = t1 - t0
        print('Train Epoch: {}, Batch:{}, \tLoss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.format(
            epoch, len(data_loader), losses / len(data_loader), 100. * correct / (correct + miss), t_epoch))

        return losses / len(data_loader), correct / (correct + miss)

    def train_student(self, epoch, source_loader, target_loader, optimizer_S, optimizer_D, schedulers=None,
                      log_interval=1000):
        # -----------------
        #  Train Student
        # -----------------
        self.student.train()
        # self.student.bottleneck.eval()
        self.student.classifier.eval()

        t0 = time.time()

        loss_c, loss_d = torch.zeros([]).cuda(), torch.zeros([]).cuda()

        len_loaders = min(len(source_loader), len(target_loader))
        zip_loaders = zip(source_loader, target_loader)
        for batch_idx, ((src_img, src_label), (tgt_img, _)) in enumerate(zip_loaders):
            src_img, src_label = src_img.cuda(), src_label.cuda()
            tgt_img = tgt_img.cuda()

            # SHOT loss
            if 'shot' in self.da_setting:
                output_tgt = self.student(tgt_img)
                # higher conf -> reduce entropy of each image decision
                loss_c = self.H_loss(output_tgt)
                # more even distribution among classes -> increase entropy of overall class prob
                avg_cls = F.softmax(output_tgt, dim=1).mean(dim=0)
                loss_d = (-avg_cls * torch.log(avg_cls)).sum()
                loss = loss_c - loss_d
            else:
                # source domain
                output_tgt, tgt_feat = self.student(tgt_img, True)
                if 'adda' in self.da_setting:
                    # update D
                    output_src, src_feat = self.teacher(src_img, True)
                    src_gt_validity = torch.ones([src_img.shape[0], 1], requires_grad=False).cuda()
                    tgt_gt_validity = torch.zeros([tgt_img.shape[0], 1], requires_grad=False).cuda()
                    src_validity = self.discriminator(src_feat[-1])
                    tgt_validity = self.discriminator(tgt_feat[-1])
                    loss_d = self.D_loss(src_validity, src_gt_validity) + self.D_loss(tgt_validity, tgt_gt_validity)
                    optimizer_D.zero_grad()
                    loss_d.backward()
                    optimizer_D.step()
                    # update S
                    output_tgt, tgt_feat = self.student(tgt_img, True)
                    tgt_validity = self.discriminator(tgt_feat[-1])
                    loss = self.D_loss(tgt_validity, src_gt_validity)
                elif 'mmd' in self.da_setting:
                    output_src, src_feat = self.student(src_img, True)
                    loss_c = self.CE_loss(output_src, src_label)
                    loss_d = self.MMD_loss(src_feat[-1], tgt_feat[-1])
                    # back-prop
                    loss = loss_c + loss_d
                else:
                    raise Exception

            optimizer_S.zero_grad()
            loss.backward()
            optimizer_S.step()

            def adjust(scheduler):
                if scheduler is not None:
                    if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                        scheduler.step(epoch - 1 + batch_idx / len_loaders)
                    elif isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                        scheduler.step()

            if isinstance(schedulers, list):
                for one_scheduler in schedulers:
                    adjust(one_scheduler)
            else:
                adjust(schedulers)

            if (batch_idx + 1) % log_interval == 0:
                t1 = time.time()
                t_epoch = t1 - t0
                print('Train Epoch: {}, Batch:{}, S: [c: {:.3f}, d: {:.3f}], Time: {:.3f}'.
                      format(epoch, (batch_idx + 1), loss_c.item(), loss_d.item(), t_epoch))

        t1 = time.time()
        t_epoch = t1 - t0
        print('Train Epoch: {}, Batch:{}, '
              'S: [c: {:.3f}, d: {:.3f}], Time: {:.3f}'.
              format(epoch, len_loaders, loss_c.item(), loss_d.item(), t_epoch))

        return loss_c.item(), loss_d.item()

    def test_teacher(self, test_loader):
        self.teacher.eval()
        losses, correct, miss = 0, 0, 0
        t0 = time.time()
        all_preds, all_labels = [], []
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            with torch.no_grad():
                output = self.teacher(data)
            pred = torch.argmax(output, 1)
            correct += pred.eq(target).sum().item()
            miss += target.shape[0] - pred.eq(target).sum().item()
            loss = self.CE_loss(output, target)
            losses += loss.item()
            all_preds.append(pred.cpu())
            all_labels.append(target.cpu())

        t1 = time.time()
        t_epoch = t1 - t0
        print('Test, Loss: {:.6f}, Prec: {:.1f}%, Time: {:.3f}'.
              format(losses / (len(test_loader) + 1), 100. * correct / (correct + miss), t_epoch))

        if self.test_visda:
            all_preds = torch.cat(all_preds, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            matrix = confusion_matrix(all_labels, all_preds)
            acc = matrix.diagonal() / matrix.sum(axis=1) * 100
            acc_str = ' '.join([str(np.round(i, 2)) for i in acc])
            print(f'visda per class accuracy\n{acc_str}')
            print('visda class-averaged accuracy: {:.1f}%'.format(acc.mean()))
            return losses / len(test_loader), acc.mean()

        return losses / len(test_loader), correct / (correct + miss)

    def test(self, test_loader):
        self.student.eval()
        tgt_C_loss = AverageMeter()
        correct = 0
        t0 = time.time()
        all_preds, all_labels = [], []
        for batch_idx, (img, label) in enumerate(test_loader):
            img, label = img.cuda(), label.cuda()
            with torch.no_grad():
                output = self.student(img)
            pred_label = torch.argmax(output, 1)
            target_C_loss = self.CE_loss(output, label)
            tgt_C_loss.update(target_C_loss.item())
            correct += (pred_label == label).sum().item()
            all_preds.append(pred_label.cpu())
            all_labels.append(label.cpu())

        t1 = time.time()
        t_epoch = t1 - t0
        print('Test, loss: {:.3f}, prec: {:.1f}%, Time: {:.3f}'.
              format(tgt_C_loss.avg, 100. * correct / len(test_loader.dataset), t_epoch))

        if self.test_visda:
            all_preds = torch.cat(all_preds, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            matrix = confusion_matrix(all_labels, all_preds)
            acc = matrix.diagonal() / matrix.sum(axis=1) * 100
            acc_str = ' '.join([str(np.round(i, 2)) for i in acc])
            print(f'visda per class accuracy\n{acc_str}')
            print('visda class-averaged accuracy: {:.1f}%'.format(acc.mean()))
            return tgt_C_loss.avg, acc.mean()
        return tgt_C_loss.avg, 100. * correct / len(test_loader.dataset)
