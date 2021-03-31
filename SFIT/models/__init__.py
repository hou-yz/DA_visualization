# from .alexnet import AlexNet
# from .vgg import VGG16
# from .resnet import ResNet18, ResNet50, ResNet152
#
# custom_factory = {
#     'alexnet': AlexNet,
#     'vgg16': VGG16,
#     'resnet18': ResNet18,
#     'resnet50': ResNet50,
#     'resnet152': ResNet152,
# }

from torchvision.models.alexnet import AlexNet
from torchvision.models.vgg import vgg16_bn
from torchvision.models.resnet import resnet18, resnet50, resnet152

torchvision_factory = {
    'alexnet': AlexNet,
    'vgg16': vgg16_bn,
    'resnet18': resnet18,
    'resnet50': resnet50,
    'resnet152': resnet152,
}


def names():
    return sorted(torchvision_factory.keys())


def create(name, num_classes, pretrained=False):
    """
    Create a model instance.
    """
    if name not in torchvision_factory:
        raise KeyError("Unknown model:", name)
    return torchvision_factory[name](num_classes=num_classes, pretrained=pretrained)
