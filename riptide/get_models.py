from .models import cifar_resnet
from .models import resnetv1b as resnet
from .models import vgg11, vggnet, resnet18, alexnet, vggnet_normal, squeezenet, squeezenet_normal, squeezenet_batchnorm
from .binary.models import q_cifar_resnet
from .binary.models import q_resnetv1b as q_resnet


def get_model(name, **kwargs):
    models = {
        'alexnet': alexnet.alexnet,
        'q_resnet18': resnet18.resnet18,
        'q_resnet34': q_resnet.resnet34_v1b,
        'q_resnet50': q_resnet.resnet50_v1b,
        'q_resnet101': q_resnet.resnet101_v1b,
        'q_resnet152': q_resnet.resnet152_v1b,
        'resnet18': resnet18.resnet18,
        'resnet34': resnet.resnet34_v1b,
        'resnet50': resnet.resnet50_v1b,
        'resnet101': resnet.resnet101_v1b,
        'resnet152': resnet.resnet152_v1b,
        'cifarnet20': cifar_resnet.cifar_resnet20_v1,
        'q_cifarnet20': q_cifar_resnet.cifar_resnet20_v1,
        'vgg11': vgg11.vgg11,
        'q_vgg11': vgg11.vgg11,
        'vggnet': vggnet.vggnet,
        'q_vggnet': vggnet.vggnet,
        'vggnet_normal': vggnet_normal.vggnet,
        'squeezenet': squeezenet.SqueezeNet,
        'squeezenet_normal': squeezenet_normal.SqueezeNet,
        'squeezenet_batchnorm': squeezenet_batchnorm.SqueezeNet,
    }
    name = name.lower()
    if name not in models:
        raise ValueError("%s Not in supported models.\n\t%s" %
                         (name, '\n\t'.join(sorted(models.keys()))))
    net = models[name](**kwargs)
    return net
