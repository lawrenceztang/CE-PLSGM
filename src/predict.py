import copy
import joblib

import torch
from torch.autograd import Variable


def predict(model, data_loader, weight_decay, closure):
    model.eval()
    loss = 0
    correct = 0
    count = 0
    grads = [0] * len(list(model.parameters()))
    for input, target in data_loader:
        ##### input, target = input.cuda(), target.cuda()

        _loss, output = closure(input, target, model)

        for i, p in enumerate(model.parameters()):
            grads[i] += (p.grad + weight_decay * p) / len(data_loader)
        loss += _loss.data + weight_decay/2 * compute_params_squared_l2_norm(model)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum().numpy()
    loss /= len(data_loader)
    acc   = 100.*correct/len(data_loader.dataset)
    loss = float(loss.detach().cpu().numpy())

    sq_grad_norm = 0
    for grad in grads:
        sq_grad_norm += torch.norm(grad)**2
    sq_grad_norm = float(sq_grad_norm.detach().cpu().numpy())
    model.train()
    return loss, acc, sq_grad_norm


def compute_params_squared_l2_norm(model):
    params = model.parameters()
    sq_norm = 0
    for p in params:
        sq_norm += p.norm()**2
    return sq_norm
