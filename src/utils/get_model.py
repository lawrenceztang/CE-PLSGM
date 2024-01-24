import os

import torch

from models import Linear, FC_10


def create_model(model_name, n_classes, n_dims, model_load_dir=None):
    model = get_model(model_name=model_name, n_classes=n_classes,
                      n_dims=n_dims)
    ##### model.cuda()
    if model_load_dir:
        param = torch.load(os.path.join(model_load_dir, "model.pth"))
        model.load_state_dict(param)
    return model


def get_model(model_name, **kargs):
    models = {"linear": Linear,
              "fc_10": FC_10
    }
    return models[model_name](**kargs)
