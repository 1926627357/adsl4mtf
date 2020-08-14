from .resnet import resnet18,resnet34,resnet50,resnet101,resnet152
from .vgg import vgg11,vgg13,vgg16,vgg19
from .WideDeep import widedeep

network = {
    'resnet18':resnet18,
    'resnet34':resnet34,
    'resnet50':resnet50,
    'resnet101':resnet101,
    'resnet152':resnet152,
    'vgg11':vgg11,
    'vgg13':vgg13,
    'vgg16':vgg16,
    'vgg19':vgg19,
    'widedeep':widedeep
}