from .alexnet import alexnet
from .resnet import ResNet18
from .squeezenet import SqueezeNet


def get_model(name, **kwargs):
    models = {
        'alexnet': alexnet,
        'resnet18': ResNet18,
        'squeezenet': SqueezeNet,
        #'vgg11': vgg.vgg11,
    }
    name = name.lower()
    if 'alexnet' in name:
        name = 'alexnet'
    elif 'resnet18' in name:
        name = 'resnet18'
    elif 'squeezenet' in name:
        name = 'squeezenet'
    if name not in models:
        raise ValueError("%s Not in supported models.\n\t%s" %
                         (name, '\n\t'.join(sorted(models.keys()))))
    model = models[name](**kwargs)
    return model
