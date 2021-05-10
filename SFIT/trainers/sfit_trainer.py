import time
from sklearn.cluster import KMeans
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import torchvision.transforms as T
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
from sklearn.metrics import confusion_matrix
from SFIT.loss import *
from SFIT.utils.meters import AverageMeter


class SFITTrainer(object):
    def __init__(self, net_G, net_S, net_T, logdir, opts, test_visda=False):
        super(SFITTrainer, self).__init__()
        self.net_G = net_G
        self.net_S = net_S
        self.net_T = net_T
        self.denorm = lambda x: (x + 1) / 2
        self.CE_loss = nn.CrossEntropyLoss()
        self.H_loss = HLoss()
        self.LSR_loss = LabelSmoothLoss()
        self.logdir = logdir
        self.test_visda = test_visda
        self.D_loss = nn.MSELoss()
        self.MMD_loss = MMDLoss()
        self.KD_loss = KDLoss(opts.KD_T)
        self.JS_loss = JSDivLoss()
        self.bn_loss = nn.MSELoss()
        self.cyc_loss = nn.L1Loss()
        self.id_loss = nn.L1Loss()
        self.content_loss = nn.L1Loss()
        self.tv_loss = TotalVariationLoss()
        self.batch_loss = BatchSimLoss()
        self.pixel_loss = PixelSimLoss()
        self.style_loss = StyleLoss()
        self.channel_loss = ChannelSimLoss() if opts.use_channel else StyleLoss()
        self.channel_loss_1d = ChannelSimLoss1D()
        self.opts = opts

    def train_net_G(self, epoch, target_loader, optimizer_G, pretrain=False, scheduler=None, log_interval=1000):
        # -----------------
        #  Train Generator
        # -----------------
        self.net_G.train()
        self.net_S.eval()
        self.net_T.eval()

        def store_mean_var(module, inputs, output):
            # input is a tuple of packed inputs
            # output is a Tensor
            cur_means.append(inputs[0].mean(dim=[0, 2, 3]))
            cur_vars.append(inputs[0].var(dim=[0, 2, 3]))

        stat_means, stat_vars = [], []
        cur_means, cur_vars = [], []
        running_means, running_vars = [], []
        handles = []
        use_BN_loss = False
        for layer in self.net_S.modules():
            if isinstance(layer, nn.BatchNorm2d):
                stat_means.append(layer.running_mean.clone())
                stat_vars.append(layer.running_var.clone())
                handles.append(layer.register_forward_hook(store_mean_var))
                use_BN_loss = True

        t0 = time.time()

        loss_conf, loss_G_BN, loss_cyc, loss_batch, loss_pixel, loss_content, = \
            torch.zeros([]).cuda(), torch.zeros([]).cuda(), torch.zeros([]).cuda(), \
            torch.zeros([]).cuda(), torch.zeros([]).cuda(), torch.zeros([]).cuda()

        loss_avg_kd, loss_avg_bn, loss_avg_channel, loss_avg_conf = \
            AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

        for batch_idx, (real_tgt_img, _) in enumerate(target_loader):
            real_tgt_img = real_tgt_img.cuda()

            # real target
            with torch.no_grad():
                if self.opts.content_ratio or pretrain:
                    output_tgt_S, featmaps_tgt_S = self.net_S(real_tgt_img, out_featmaps=True)
                output_tgt_T, featmaps_tgt_T = self.net_T(real_tgt_img, out_featmaps=True)
            cur_means, cur_vars = [], []
            # fake source
            gen_src_img = self.net_G(real_tgt_img)
            output_src_S, featmaps_src_S = self.net_S(gen_src_img, out_featmaps=True)
            # loss
            # style (BN)
            if use_BN_loss:
                loss_G_BN = 0
                for layer_id in range(len(stat_means)):
                    if layer_id >= len(running_means):
                        running_means.append(cur_means[layer_id])
                        running_vars.append(cur_vars[layer_id])
                    else:
                        running_means[layer_id] = running_means[layer_id].detach() * (1 - self.opts.mAvrgAlpha) + \
                                                  cur_means[layer_id] * self.opts.mAvrgAlpha
                        running_vars[layer_id] = running_vars[layer_id].detach() * (1 - self.opts.mAvrgAlpha) + \
                                                 cur_vars[layer_id] * self.opts.mAvrgAlpha
                    loss_G_BN += self.bn_loss(stat_means[layer_id], running_means[layer_id]) + \
                                 self.bn_loss(torch.sqrt(stat_vars[layer_id]), torch.sqrt(running_vars[layer_id]))
                loss_G_BN = loss_G_BN / len(stat_means)
            # style
            loss_style = torch.zeros([]).cuda()
            for layer_id in range(len(featmaps_tgt_T) - 1):
                loss_style += self.style_loss(featmaps_src_S[layer_id], featmaps_tgt_T[layer_id])
            # similarity preserving
            loss_channel = self.channel_loss(featmaps_src_S[-2], featmaps_tgt_T[-2])
            # loss_channel = self.channel_loss_1d(featmaps_src_S[-1], featmaps_tgt_T[-1])
            loss_batch = self.batch_loss(featmaps_src_S[-2], featmaps_tgt_T[-2])
            loss_pixel = self.pixel_loss(featmaps_src_S[-2], featmaps_tgt_T[-2])
            # kd
            loss_kd = self.KD_loss(output_src_S, output_tgt_T)
            # content
            if self.opts.content_ratio or pretrain:
                loss_content = self.content_loss(featmaps_src_S[-2], featmaps_tgt_S[-2])
            else:
                loss_content = torch.zeros([]).cuda()
            # others
            loss_id = self.id_loss(gen_src_img, real_tgt_img)
            loss_tv = self.tv_loss(gen_src_img)[1]
            loss_activation = -featmaps_src_S[-2].abs().mean()
            # SHOT loss
            loss_conf = self.H_loss(output_src_S)
            avg_cls = torch.nn.functional.softmax(output_src_S, dim=1).mean(dim=0)
            loss_div = -(avg_cls * torch.log(avg_cls)).sum()
            # co-training
            loss_js = self.JS_loss(output_tgt_T, output_src_S)
            loss_G = loss_conf * self.opts.conf_ratio - loss_div * self.opts.div_ratio + loss_js * self.opts.js_ratio + \
                     loss_content * self.opts.content_ratio + loss_style * self.opts.style_ratio + loss_channel * self.opts.channel_ratio + \
                     loss_id * self.opts.id_ratio + loss_kd * self.opts.kd_ratio + loss_tv * self.opts.tv_ratio + \
                     loss_batch * self.opts.batch_ratio + loss_pixel * self.opts.pixel_ratio
            if use_BN_loss:
                loss_G += loss_G_BN * self.opts.bn_ratio
            # first train the G as a transparent filter
            if pretrain:
                loss_G = loss_id + loss_content + self.KD_loss(output_src_S, output_tgt_S)

            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                    scheduler.step(epoch - 1 + batch_idx / len(target_loader))
                elif isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    scheduler.step()

            loss_avg_kd.update(loss_kd.item())
            loss_avg_bn.update(loss_G_BN.item())
            loss_avg_channel.update(loss_channel.item())
            loss_avg_conf.update(loss_conf.item())

            if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == len(target_loader):
                # print(alpha)
                t1 = time.time()
                t_epoch = t1 - t0
                print(f'Train Epoch: {epoch}, Batch:{batch_idx + 1}, G: [c: {loss_avg_conf.avg:.3f}, '
                      f'bn: {loss_avg_bn.avg:.5f}, channel: {loss_avg_channel.avg:.5f}, kd: {loss_avg_kd.avg:.3f}], '
                      f'Time: {t_epoch:.3f}')
                self.sample_image(real_tgt_img, fname=self.logdir + f'/imgs/{epoch}.png')
        # print(f'pixel sim: {loss_pixel.item()}, batch sim: {loss_batch.item()}')

        # remove forward hooks registered in this epoch
        for handle in handles:
            handle.remove()

        return loss_avg_kd.avg, loss_avg_channel.avg, loss_avg_conf.avg

    def train_net_T(self, epoch, target_loader, optimizer_T, scheduler=None, use_generator=False, log_interval=1000):
        # -----------------
        #  Train Target Network
        # -----------------
        self.net_G.eval()
        self.net_S.eval()
        self.net_T.train()
        self.net_T.classifier.eval()

        t0 = time.time()

        loss_conf, loss_div = torch.zeros([]).cuda(), torch.zeros([]).cuda()

        for batch_idx, (real_tgt_img, _) in enumerate(target_loader):
            real_tgt_img = real_tgt_img.cuda()

            # SHOT loss
            output_tgt_T, featmaps_tgt_T = self.net_T(real_tgt_img, out_featmaps=True)
            # higher conf -> reduce entropy of each image decision
            loss_conf = self.H_loss(output_tgt_T)
            # more even distribution among classes -> increase entropy of overall class prob
            avg_cls = F.softmax(output_tgt_T, dim=1).mean(dim=0)
            loss_div = (-avg_cls * torch.log(avg_cls)).sum()
            loss_T = loss_conf - loss_div

            # generator
            if use_generator:  # and (self.opts.T_batch_ratio or self.opts.T_pixel_ratio or self.opts.js_ratio):
                with torch.no_grad():
                    gen_src_img = self.net_G(real_tgt_img)
                    output_src_S, featmaps_src_S = self.net_S(gen_src_img, out_featmaps=True)
                # output_src_T, featmaps_src_T = self.net_T(gen_src_img, out_featmaps=True)
                # reset SHOT loss
                pred_label_T = torch.argmax(output_tgt_T, 1)
                pred_label_S = torch.argmax(output_src_S, 1)
                valid_idx = pred_label_S == pred_label_T
                loss_conf = self.H_loss(output_tgt_T[valid_idx])
                # conf = F.softmax(output_src_S, dim=1).max(dim=1)[0]
                # valid_idx = conf > self.opts.confidence_thres
                # loss_ce = self.CE_loss(output_tgt_T[valid_idx], pred_label_S[valid_idx])
                loss_T = loss_conf - loss_div  # + loss_ce

                loss_batch = self.batch_loss(featmaps_src_S[-2], featmaps_tgt_T[-2])
                loss_pixel = self.pixel_loss(featmaps_src_S[-2], featmaps_tgt_T[-2])
                loss_T += loss_batch * self.opts.T_batch_ratio + loss_pixel * self.opts.T_pixel_ratio
                # co-training
                loss_js = self.JS_loss(output_tgt_T, output_src_S)
                loss_T += loss_js * self.opts.js_ratio

            optimizer_T.zero_grad()
            loss_T.backward()
            optimizer_T.step()

            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
                    scheduler.step(epoch - 1 + batch_idx / len(target_loader))
                elif isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    scheduler.step()

            if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == len(target_loader):
                # print(alpha)
                t1 = time.time()
                t_epoch = t1 - t0
                print(f'Train Epoch: {epoch}, Batch:{batch_idx + 1}, S: [c: {loss_conf.item():.3f}, '
                      f'd: {loss_div.item():.3f}], Time: {t_epoch:.3f}')

        return loss_conf.item(), loss_div.item()

    def test_net_S(self, test_loader, imgs_type=None):
        self.net_S.eval()
        losses, correct, miss = 0, 0, 0
        t0 = time.time()
        all_preds, all_labels = [], []
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            with torch.no_grad():
                output = self.net_S(data)
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
        if imgs_type is not None:
            data, target = next(iter(test_loader))
            self.sample_image(data.cuda(), self.logdir + f'/imgs/{imgs_type}.png', False)

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

    def test(self, test_loader, epoch=None, use_generator=False, visualize=False):
        self.net_G.eval()
        self.net_S.eval()
        self.net_T.eval()
        tgt_C_loss = AverageMeter()
        correct = 0
        correct_valid, valid = 0, 0
        t0 = time.time()
        all_preds, all_labels = [], []
        for batch_idx, (img, label) in enumerate(test_loader):
            img, label = img.cuda(), label.cuda()
            with torch.no_grad():
                if use_generator:
                    output_T, featmaps_tgt_T = self.net_T(img, out_featmaps=True)
                    pred_label_S = torch.argmax(output_T, 1)
                    # fake source
                    img = self.net_G(img)
                    output, _ = output_S, featmaps_src_S = self.net_S(img, out_featmaps=True)
                    # channel (pixelsim preserving)
                    # conf = F.softmax(output, dim=1).max(dim=1)[0]
                    # valid_idx = conf > self.opts.confidence_thres
                    pred_label = pred_label_T = torch.argmax(output_S, 1)
                    valid_idx = pred_label_T == pred_label_S
                else:
                    output = self.net_T(img)
                    # conf = F.softmax(output, dim=1).max(dim=1)[0]
                    # valid_idx = conf > self.opts.confidence_thres
                    pred_label = torch.argmax(output, 1)
                    valid_idx = torch.ones_like(pred_label).bool()

            target_C_loss = self.CE_loss(output, label)

            tgt_C_loss.update(target_C_loss.item())
            correct += (pred_label == label).sum().item()
            correct_valid += (pred_label[valid_idx] == label[valid_idx]).sum().item()
            valid += valid_idx.sum().item()

            all_preds.append(pred_label.cpu())
            all_labels.append(label.cpu())

        t1 = time.time()
        t_epoch = t1 - t0
        print(f'Test, loss: {tgt_C_loss.avg:.3f}, prec: {100. * correct / len(test_loader.dataset):.1f}%, '
              f'Time: {t_epoch:.3f}')
        print(f'valid ratio: {100. * valid / len(test_loader.dataset):.1f}%, '
              f'precision in valid ones: {100. * correct_valid / (valid + 1e-10):.1f}%')

        if epoch is not None and use_generator:
            img, _ = next(iter(test_loader))
            self.sample_image(img.cuda(), fname=self.logdir + f'/imgs/{epoch}.png')

        if visualize:
            # 2d
            # f_src, f_tgt = featmaps_src_S[-2], featmaps_tgt_T[-2]
            # B, C, H, W = f_src.shape
            # f_src, f_tgt = f_src[0].view(C, -1), f_tgt[0].view(C, -1)
            # 1d
            f_src, f_tgt = featmaps_src_S[-1], featmaps_tgt_T[-1]
            B, C = f_src.shape
            f_src, f_tgt = f_src[0].view(C, 1), f_tgt[0].view(C, 1)
            indices = np.argsort(KMeans(4).fit_predict(f_src.cpu().detach()))
            f_src, f_tgt = f_src[indices], f_tgt[indices]
            A_src, A_tgt = f_src @ f_src.T, f_tgt @ f_tgt.T
            loss1 = (F.normalize(A_src, p=2, dim=1) - F.normalize(A_tgt, p=2, dim=1)).cpu().detach()
            loss2 = (A_src - A_tgt).cpu().detach()
            loss1_max, loss2_max = loss1.abs().max(), loss2.abs().max()

            fig, ax = plt.subplots(figsize=(4.5, 5))
            im = ax.imshow(loss1, vmin=-loss1_max, vmax=loss1_max, cmap='seismic')
            divider = make_axes_locatable(ax)
            cax = divider.new_vertical(size="5%", pad=0.4, pack_start=True)
            fig.add_axes(cax)
            fig.colorbar(im, cax=cax, orientation="horizontal")
            plt.show()

            fig, ax = plt.subplots(figsize=(4.5, 5))
            im = ax.imshow(loss2, vmin=-loss2_max, vmax=loss2_max, cmap='seismic')
            divider = make_axes_locatable(ax)
            cax = divider.new_vertical(size="5%", pad=0.4, pack_start=True)
            fig.add_axes(cax)
            fig.colorbar(im, cax=cax, orientation="horizontal")
            plt.show()

        if use_generator:
            data = []
            # transform = T.Compose([T.Normalize((-1, -1, -1), (2, 2, 2)), T.ToPILImage(), ])
            if self.test_visda:
                # indices = [47521, 27974, 32185, 11317]
                indices = [43359, 39475, 28118]
            else:
                indices = [371, 325, 55]
            for idx in indices:
                img = test_loader.dataset[idx][0]
                data.append(img)
            data = torch.stack(data, dim=0).cuda()
            self.sample_image(data, self.logdir + f'/visualize.png', True, 1)

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

    def sample_image(self, real_tgt, fname, use_generator=True, nrow=8):
        if use_generator:
            """Saves a generated sample from the test set"""
            self.net_G.eval()
            with torch.no_grad():
                gen_src = self.net_G(real_tgt)
            # Arange images along x-axis
            real_T = make_grid(self.denorm(real_tgt), nrow=nrow)
            gen_S = make_grid(self.denorm(gen_src), nrow=nrow)
            # Arange images along y-axis
            image_grid = torch.cat((real_T, gen_S), 2)
        else:
            image_grid = make_grid(self.denorm(real_tgt), nrow=8)
            pass
        save_image(image_grid, fname, normalize=False)


def imshow(tensor):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    unloader = T.Compose([T.Normalize((-1, -1, -1), (2, 2, 2)), T.ToPILImage(), ])
    image = unloader(image)
    plt.imshow(image)
    plt.show()
    image.save('tmp.png')
