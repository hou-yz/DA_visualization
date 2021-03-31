import torch
import torch.nn as nn
import torch.nn.utils.weight_norm as weight_norm
from torchvision.models.resnet import resnet50, resnet101


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class ClassifierShot(nn.Module):
    def __init__(self, num_classes, arch='lenet', bottleneck_dim=256, use_shot=True):
        super(ClassifierShot, self).__init__()

        self.arch = arch
        dropout = 0.0
        if arch == 'lenet':
            self.base = LeNetBase()
            layer_ids = [2, 6]
            dropout = 0.5
        elif arch == 'dtn':
            self.base = DTNBase()
            layer_ids = [3, 7, 11]
            dropout = 0.5
        elif arch == 'resnet50':
            self.base = nn.Sequential(*list(resnet50(pretrained=True).children())[:-2])
            self.base.out_features = 2048
            layer_ids = [4, 5, 6, 7]
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        elif arch == 'resnet101':
            self.base = nn.Sequential(*list(resnet101(pretrained=True).children())[:-2])
            self.base.out_features = 2048
            layer_ids = [4, 5, 6, 7]
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            raise Exception
        self.bottleneck = feat_bootleneck(self.base.out_features, bottleneck_dim, dropout, use_shot)
        self.classifier = feat_classifier(num_classes, bottleneck_dim, use_shot)

        # hook for record featmaps
        def store_featmap(module, inputs, output):
            # input is a tuple of packed inputs
            # output is a Tensor
            self.featmaps.append(output)

        self.featmaps = []
        for layer_id in layer_ids:
            if 'resnet' in arch:
                self.base[layer_id].register_forward_hook(store_featmap)
            else:
                self.base.conv_params[layer_id].register_forward_hook(store_featmap)
        if use_shot:
            self.bottleneck.bn.register_forward_hook(store_featmap)
        else:
            self.bottleneck.bottleneck.register_forward_hook(store_featmap)

    def forward(self, x, out_featmaps=False):
        self.featmaps = []
        x = self.base(x)
        if 'resnet' in self.arch:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
        label = self.classifier(x)

        if out_featmaps:
            return (label, self.featmaps)
        else:
            return label


class feat_bootleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, dropout=0.5, use_shot=True):
        super(feat_bootleneck, self).__init__()
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
        self.dropout = nn.Dropout(p=dropout)
        self.bottleneck.apply(init_weights)
        self.use_shot = use_shot

    def forward(self, x):
        x = self.bottleneck(x)
        if self.use_shot:
            x = self.bn(x)
            x = self.dropout(x)
        return x


class feat_classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, use_shot=True):
        super(feat_classifier, self).__init__()

        self.fc = nn.Linear(bottleneck_dim, class_num, bias=False)
        if use_shot:
            self.fc = weight_norm(self.fc, name="weight")
        self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        return x


class DTNBase(nn.Module):
    def __init__(self):
        super(DTNBase, self).__init__()
        self.conv_params = nn.Sequential(nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
                                         nn.BatchNorm2d(64), nn.Dropout2d(0.1), nn.ReLU(),
                                         nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
                                         nn.BatchNorm2d(128), nn.Dropout2d(0.3), nn.ReLU(),
                                         nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
                                         nn.BatchNorm2d(256), nn.Dropout2d(0.5), nn.ReLU())
        self.out_features = 256 * 4 * 4

    def forward(self, x):
        x = self.conv_params(x)
        return x


class LeNetBase(nn.Module):
    def __init__(self):
        super(LeNetBase, self).__init__()
        self.conv_params = nn.Sequential(nn.Conv2d(1, 20, kernel_size=5),
                                         nn.MaxPool2d(2), nn.ReLU(),
                                         nn.Conv2d(20, 50, kernel_size=5),
                                         nn.Dropout2d(p=0.5), nn.MaxPool2d(2), nn.ReLU(), )
        self.out_features = 50 * 4 * 4

    def forward(self, x):
        x = self.conv_params(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, in_channel):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(nn.Linear(in_channel, 256), nn.ReLU(),
                                   nn.Linear(256, 256), nn.ReLU(),
                                   nn.Linear(256, 1))

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':
    net = ClassifierShot(10)
    img = torch.zeros([64, 1, 28, 28])
    label = net(img)
    net = ClassifierShot(10, arch='resnet50')
    img = torch.zeros([64, 3, 224, 224])
    label, featmaps = net(img, out_featmaps=True)
    pass
