from .alexnet import alexnet


def get_model(name, **kwargs):
    models = {
        'alexnet': alexnet,
        'alexnet_sgd': alexnet,
        #'vgg11': vgg.vgg11,
    }
    name = name.lower()
    if name not in models:
        raise ValueError("%s Not in supported models.\n\t%s" %
                         (name, '\n\t'.join(sorted(models.keys()))))
    model = models[name](**kwargs)
    return model
