import copy
import sys
sys.path.append('./')
sys.path.append("utils/")
import math
import time

import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import functorch
from random import random
from math import exp, sqrt
import computeamplification as CA

import numpy as np
from scipy.special import binom

from collections import OrderedDict

from predict import compute_params_squared_l2_norm


def average(params_lst):
    params_lst = [list(params) for params in params_lst]
    n_layers = len(params_lst[0])
    averaged = [0]*n_layers
    for i in range(n_layers):
        for params in params_lst:
            averaged[i] += params[i]/len(params_lst)
    return averaged


def get_dim(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def add(params1, params2):
    return [p1 + p2 for p1, p2 in zip(params1, params2)]

def subtract(params1, params2):
    return [p1 - p2 for p1, p2 in zip(params1, params2)]

def multiply(c, l):
    if isinstance(l, list):
        return [multiply(c, e) for e in l]
    return c * l

def divide(l, c):
    return [torch.div(e, c) for e in l]

def equal(params1, params2):
    for p1, p2 in zip(params1, params2):
        assert torch.allclose(p1, p2)

        
def compute_inner_product(params1, params2):
    ans = 0
    for p1, p2 in zip(params1, params2):
        ans += torch.inner(p1.view(-1), p2.view(-1))
    return ans

    
def compute_per_sample_norm(per_sample_grads, p=None):
    if p is None:
        p = 2
    per_sample_pth_norm = 0
    for per_sample_g in per_sample_grads:
        n_dim = len(per_sample_g.size())
        per_sample_pth_norm += torch.norm(per_sample_g, dim=tuple(range(1, n_dim)), p=p)**p
    per_sample_norm = per_sample_pth_norm ** (1/p)
    return per_sample_norm


def clip(per_sample_grads, c):
    per_sample_norm = compute_per_sample_norm(per_sample_grads)
    ##### per_sample_scale = torch.minimum(torch.ones(per_sample_norm.size()).cuda(), c/(per_sample_norm+10**(-10)))
    per_sample_scale = torch.minimum(torch.ones(per_sample_norm.size()), c/(per_sample_norm+10**(-10)))
    clipped = []
    for per_sample_g in per_sample_grads:
        n_dim = len(per_sample_g.size())
        clipped.append(per_sample_scale.view([-1]+[1]*(n_dim-1)) * per_sample_g)
    return clipped


def clip2(grads, c):
    norm = L2_distance(grads, [0]*len(grads))
    scale = min(1, c/(norm+10**(-10)))
    clipped = []
    for g in grads:
        clipped.append(scale * g)
    return clipped


def L2_distance(params1, params2):
    ans = 0
    for p1, p2 in zip(params1, params2):
        ans += torch.norm(p1-p2) ** 2
    return ans ** 0.5


def L2_norm(params):
    return L2_distance(params, [0]*len(params))


def add_noise(grads, sd, return_noise=False):
    ##### noise = [sd*torch.randn(g.size()).cuda() for g in grads]
    noise = [sd*torch.randn(g.size()) for g in grads]
    if return_noise:
        return add(grads, noise), noise
    return add(grads, noise)

def flatten(x):
    return

def sample_from_unit_sphere(dim):
    point = torch.randn(dim)
    point = point / point.norm()
    return point

def gamma_div(d):
    log = math.lgamma((d - 1) / 2 + 1) - math.lgamma(d / 2 + 1)
    return exp(log)

# CHANGED
def ldp_mechanism(x_orig, c, epsilon):
    x = torch.cat([t.view(-1) for t in x_orig])
    d = x.size(0)
    rand = random()
    x_norm = torch.norm(x)
    if rand < 1/2 + x_norm / (2 * c):
        z = c * x / x_norm
    else:
        z = -c * x / x_norm

    rand2 = random()
    v = sample_from_unit_sphere(d)
    if rand2 < exp(epsilon) / (1 + exp(epsilon)):
        z_tilde = torch.sign(torch.dot(v, z)) * v
    else:
        z_tilde = -torch.sign(torch.dot(v, z)) * v
    B = c * (exp(epsilon) + 1) / (exp(epsilon) - 1) * (sqrt(math.pi) / 2) * d * gamma_div(d)
    z_bar = B * z_tilde

    out = []
    start = 0
    for t in x_orig:
        size = t.numel()
        out.append(z_bar[start:start + size].view_as(t))
        start += size
    return out

class GradientCalculator(optim.Optimizer):
    def __init__(self, model, weight_decay, closure, loss_fn):
        super(GradientCalculator, self).__init__(params=model.parameters(), defaults={})
        self._model = model
        self._closure = closure
        self._loss_fn = loss_fn
        self._weight_decay = weight_decay
        
        self._fmodel, self._params, self._buffers = functorch.make_functional_with_buffers(self._model)        


    def _add_l2(self, grads, params):
        grads_with_l2 = []
        for g, p in zip(grads, params):
            g = g + self._weight_decay * p.data 
            grads_with_l2.append(g)
        return grads_with_l2

    def _compute_loss_stateless_model(self, params, buffers, input, target):
        input = input.unsqueeze(0)
        target = target.unsqueeze(0)

        output = self._fmodel(params, buffers, input)
        loss = self._loss_fn(output, target)
        return loss

    
    def _compute_per_sample_stochastic_grads(self, input, target):
        ##### input, target = input.cuda(), target.cuda()
        batch_size = len(input)
        self._model.zero_grad()
        self._fmodel, self._params, self._buffers = functorch.make_functional_with_buffers(self._model)
        ft_compute_grad = functorch.grad(self._compute_loss_stateless_model)
        ft_compute_sample_grad = functorch.vmap(ft_compute_grad, in_dims=(None, None, 0, 0))
        ft_per_sample_grads = ft_compute_sample_grad(self._params, self._buffers, input, target)
        ft_per_sample_grads = self._add_l2(ft_per_sample_grads, self._model.parameters())
        return [g.data for g in ft_per_sample_grads]
        
    # _compute_clipped_average is much faster than _compute_clipped_average2
    def _compute_clipped_average(self, per_sample_grads, c):
        clipped_per_sample_grads = clip(per_sample_grads, c)
        averaged = [layer_grads.mean(dim=0) for layer_grads in clipped_per_sample_grads]
        return averaged
    
    def _compute_stochastic_grad(self, input, target):
        print("Non-private gradients are used!")
        assert False
        ##### input, target = input.cuda(), target.cuda()
        self._closure(input, target, self._model)
        grads = []
        for p in self._model.parameters():
            grads.append(p.grad.data)
        grads_with_l2 = self._add_l2(grads, self._model.parameters())
        return grads_with_l2
    
    def test_per_sample_grads(self, input, target, iter_ind):
        per_sample_grads = self._compute_stochastic_per_sample_grads(input, target)
        grads1 = average(per_sample_grads)
        grads2 = self._compute_stochastic_grad(input, target)
        for g1, g2 in zip(grads1, grads2):
            assert torch.allclose(g1, g2, atol=3e-3, rtol=1e-5)
        print(f"Iter {iter_ind} passed per sample gradient function")

    def copy_params(self, params_from):
        for (p_to, p_from) in zip(self._model.parameters(), params_from):
            p_to.data = copy.deepcopy(p_from.data)
        for (p_to, p_from) in zip(self._model.parameters(), params_from):
            if torch.any(torch.isnan(p_to)) or torch.any(torch.isnan(p_from)) or (not torch.allclose(p_to.data, p_from.data)):
                pass

            
    def compute_risk(self, dataset, weight_decay):
        self._model.eval()
        loss = 0
        correct = 0
        count = 0
        batch_size = 200
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False).__iter__()
        for data in loader:
            input, target = data
            ##### input, target = input.cuda(), target.cuda()
            _loss = self._closure(input, target, self._model) + weight_decay/2 * compute_params_squared_l2_norm(self._model)
            loss += _loss.detach().cpu().item()
        loss /= len(loader)
        self._model.train()
        return loss

    
class Client(GradientCalculator):
    def __init__(self, model, eta, weight_decay, train_loader, n_iters, **kargs):
        super(Client, self).__init__(model, weight_decay, kargs["closure"], kargs["loss_fn"])
        self._eta = eta
        self._train_loader = train_loader
        self._n_iters = n_iters

    def train(self, **kargs):
        for i in range(self._n_iters):
            input, target = self._sample()
            grads = self._compute_grad_estimator(input, target, **kargs)
            #self.test_per_sample_grads(input, target, i)
            self._update(grads)

    def _sample(self):
        return next(self._train_loader.__iter__())
    
    def _compute_grad_estimator(self, input, target):
        raise NotImplementedError
                    
    def _update(self, grads):
        for i, p in enumerate(self._model.parameters()):
            p.data.add_(grads[i].data, alpha=-self._eta)

    def _trace_per_sample_info(self, per_sample_func, norm="l2"):
        dataset = self._train_loader.dataset
        batch_size = 200
        tmp_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False).__iter__()
        per_sample_norm_lst = []
        for data in tmp_loader:
            input, target = data
            #print(input.size())
            per_sample_grad_info = per_sample_func(input, target)
            per_sample_norm = compute_per_sample_norm(per_sample_grad_info)
            per_sample_norm_lst.append(per_sample_norm)
        return torch.cat(per_sample_norm_lst, dim=0)

    def _compute_per_sample_grad_diff(self, input, target):
        curr_per_sample_grads = self._compute_per_sample_stochastic_grads(input, target)
        prev_per_sample_grads = self._prev_round_snap_shot._compute_per_sample_stochastic_grads(input, target)
        per_sample_grad_diff = subtract(curr_per_sample_grads, prev_per_sample_grads)
        return per_sample_grad_diff

    def _trace_per_sample_lipschitzness(self):
        return self._trace_per_sample_info(self._compute_per_sample_stochastic_grads)

    def reset_eta(self, eta):
        self._eta = eta

        
class Diff2_Client(Client):
    def __init__(self, **kargs):
        super(Diff2_Client, self).__init__(**kargs)
        self._prev_round_snap_shot = GradientCalculator(model=copy.deepcopy(kargs["model"]), weight_decay=kargs["weight_decay"], closure=kargs["closure"], loss_fn=kargs["loss_fn"])

        self._ref_grads = None

    def _test_tensor_clipping(self):
        dataset = self._train_loader.dataset
        batch_size = 200
        tmp_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True).__iter__()
        input, target = next(tmp_loader)

        for c in [0.05, 0.5, 5.0]:
            s = time.time()
            per_sample_grads = self._compute_per_sample_stochastic_grads(input, target)
            clipped_average = self._compute_clipped_average(per_sample_grads, c)
            print("non-tensor clipping", time.time() - s)
            s = time.time()
            per_sample_grads = self._compute_per_sample_stochastic_grads2(input, target)
            clipped_average2 = self._compute_clipped_average2(per_sample_grads, c)
            print("tensor clipping", time.time() - s)
            for g, g2 in zip(clipped_average, clipped_average2):
                assert torch.allclose(g, g2)
        print("Tensor clipping tests passed!")
        assert False
        
    def _compute_full_grad_info(self, per_sample_func, c):
        dataset = self._train_loader.dataset
        batch_size = 200
        tmp_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False).__iter__()
        clipped_averages_per_batch = []
        clipping_scales_per_batch = []
        max_norm_lst = []
        for i, data in enumerate(tmp_loader):
            input, target = data
            per_sample_grad_info = per_sample_func(input, target)
            clipped_average = self._compute_clipped_average(per_sample_grad_info, c)
            clipped_averages_per_batch.append(clipped_average)
        averaged = average(clipped_averages_per_batch)
        return averaged
    
    def compute_full_grad_diff(self, c):
        return self._compute_full_grad_info(self._compute_per_sample_grad_diff, c)

    def compute_full_grad(self, c):
        #self._test_tensor_clipping()
        return self._compute_full_grad_info(self._compute_per_sample_stochastic_grads, c)    

    def set_ref_grads(self, grads):
        self._ref_grads = []
        for grad in grads:
            self._ref_grads.append(copy.deepcopy(grad))

    def update_prev_round_snap_shot(self):
        self._prev_round_snap_shot.copy_params(self._model.parameters())

    def _trace_per_sample_smoothness(self):
        dist_params = L2_distance(self._model.parameters(),
                                  self._prev_round_snap_shot._model.parameters()).data.item()
        per_sample_grad_diff_norm = self._trace_per_sample_info(self._compute_per_sample_grad_diff)
        return per_sample_grad_diff_norm / (dist_params + 10**(-10))


class Diff2_GD_Client(Diff2_Client):
    def _compute_grad_estimator(self, input, target):
        assert self._n_iters == 1
        return self._ref_grads
        
class Server:
    def __init__(self, model, eta, weight_decay, train_loaders,
                 n_local_iters, n_global_iters, n_workers, eps, delta,
                 closure, loss_fn, **kargs):
        self._model = model
        self._n_workers = n_workers        
        self._n_local_iters = n_local_iters
        self._n_global_iters = n_global_iters
        self._eps = eps
        if eps is None:
            self._eps = np.inf
        self._delta = delta
        self._eta = eta
        self._weight_decay = weight_decay
        self._optimizers = [self._get_optimizer(model=copy.deepcopy(self._model),
                                                eta=eta, weight_decay=weight_decay, 
                                                train_loader=train_loaders[i],
                                                n_iters=n_local_iters,
                                                closure=closure, loss_fn=loss_fn, eps=eps)
                            for i in range(self._n_workers)]

        
        self._n_min = np.min([len(loader.dataset) for loader in train_loaders])
        self._alpha = int(1 + np.ceil(2*math.log(1/self._delta)/self._eps))
        self._update_count = 0

    def update(self):
        self._update()
        self._update_count += 1

    def get_model(self):
        return self._model

    def _update(self):
        raise NotImplementedError
    
    def _get_optimizer(self, **kargs):
        raise NotImplementedError

    def reset_eta(self, eta):
        self._eta = eta
        for optimizer in self._optimizers:
            optimizer.reset_eta(eta)
            

class Diff2_Server(Server):
    def __init__(self, T, c, c2, **kargs):
        super(Diff2_Server, self).__init__(**kargs)
        self._prev_round_model = copy.deepcopy(self._model)
        self._prev_stage_model = copy.deepcopy(self._model)
        self._T = T
        self._c = c
        if c is None:
            self._c = 10**15
        self._c2 = c2
        if c2 is None:
            self._c2 = 10**15

        self._u = 1.25
        assert self._u > 1

        self._set_sigma()
        self._set_sigma2()

        self._cum_dp_global_grad = None
        self._cum_true_global_grad = None
        self._prev_global_grad = None
        
    def _update(self):        
        self._compute_global_grad()
        self._communicate_global_grad()
        self._update_prev_round_snap_shot()
        params = self._one_round_routine()
        self._communicate_params(params)
        
    def _communicate_params(self, params):
        params = list(params) # input may be a generator
        for i, optimizer in enumerate(self._optimizers):
            optimizer.copy_params(params)
        self._model = copy.deepcopy(self._optimizers[0]._model)
                
    def _compute_global_grad(self):
        if self._update_count % self._T == 0:
            full_grad_lst = [optimizer.compute_full_grad(c=self._c)
                             for optimizer in self._optimizers]
            dp_global_grad, noise = add_noise(average(full_grad_lst), self._c*self._sigma, return_noise=True)
        else:
            d = get_dim(self._model)
            dist_params = L2_distance(self._model.parameters(),
                                      self._prev_round_model.parameters()).data.item()
            c = self._c2 * dist_params
            full_grad_diff_lst = [optimizer.compute_full_grad_diff(c=c)
                                  for optimizer in self._optimizers]
            dp_full_grad_diff = add_noise(average(full_grad_diff_lst),
                                          c*self._sigma2)
            dp_global_grad = add(dp_full_grad_diff, self._ref_grads)
        self._ref_grads = dp_global_grad

    def _communicate_global_grad(self):        
        for optimizer in self._optimizers:
            optimizer.set_ref_grads(self._ref_grads)

    def _update_prev_round_snap_shot(self):
        for optimizer in self._optimizers:
            optimizer.update_prev_round_snap_shot()
        self._prev_round_model = copy.deepcopy(self._optimizers[0]._model)
            
    def _set_sigma2(self):
        raise NotImplementedError

    def trace_loss_lipschitzness(self):
        per_sample_lipschitzness = torch.cat([optimizer._trace_per_sample_lipschitzness() for optimizer in self._optimizers])
        return torch.max(per_sample_lipschitzness).cpu().item()

    def trace_loss_smoothness(self):
        per_sample_smoothness = torch.cat([optimizer._trace_per_sample_smoothness() for optimizer in self._optimizers])
        return torch.max(per_sample_smoothness).cpu().item()

    def trace_risk_smoothness(self):
        dist_params = L2_distance(self._model.parameters(),
                                  self._prev_round_model.parameters()).data.item()
        full_grad_diff = average([optimizer.compute_full_grad_diff(c=np.inf)
                                  for optimizer in self._optimizers])
        full_grad_diff_norm = L2_distance(full_grad_diff, [0]*len(full_grad_diff))
        return (full_grad_diff_norm/(dist_params+10**(-10))).cpu().item()

    def trace_risk_smoothness2(self):
        full_grad_lst = [optimizer.compute_full_grad(c=np.inf)
                             for optimizer in self._optimizers]
        global_grad = average(full_grad_lst)
        if self._prev_global_grad is None:
            print("prev_global_grad is not computed")
            self._prev_global_grad = global_grad
            return np.inf
        dist_params = L2_distance(self._model.parameters(),
                                  self._prev_round_model.parameters()).data.cpu().item()
        curr_train_risk = np.mean([optimizer.compute_risk(optimizer._train_loader.dataset, self._weight_decay) for optimizer in self._optimizers])
        prev_train_risk = np.mean([optimizer._prev_round_snap_shot.compute_risk(optimizer._train_loader.dataset, self._weight_decay) for optimizer in self._optimizers])
        inner_product = compute_inner_product(subtract(self._model.parameters(), self._prev_round_model.parameters()), self._prev_global_grad).cpu().item()
        self._prev_global_grad = global_grad
        
        L_upper =  2 * (curr_train_risk - prev_train_risk - inner_product)/(dist_params**2)
        return L_upper
    
    
# Diff2 with GD-Routine
# T=1 -> DP-GD.
class Diff2_GD_Server(Diff2_Server):
    def _one_round_routine(self):
        i = self._update_count % self._n_workers
        self._optimizers[i].train()
        return self._optimizers[i]._model.parameters()

    def _get_optimizer(self, **kargs):
        assert self._n_local_iters == 1
        return Diff2_GD_Client(**kargs)
    
    def _set_sigma(self):
        if self._T > 1:
            coef = 4 * self._u
        else:
            coef = 4
        self._sigma = np.sqrt(coef * self._alpha * self._n_global_iters / (self._T * self._n_min**2 * self._n_workers**2 * self._eps))

    def _set_sigma2(self):
        coef = 2 * (2 * self._u) / (self._u - 1) 
        self._sigma2 = np.sqrt(coef * self._alpha * self._n_global_iters / (self._n_min**2 * self._n_workers**2 * self._eps))


class CE_PLS_Client(Client):
    def __init__(self, **kargs):
        super(CE_PLS_Client, self).__init__(**kargs)
        self._prev_round_snap_shot = GradientCalculator(model=copy.deepcopy(kargs["model"]),
                                                        weight_decay=kargs["weight_decay"], closure=kargs["closure"],
                                                        loss_fn=kargs["loss_fn"])
        self._eps = kargs["eps"]
        self._ref_grads = None

    def _test_tensor_clipping(self):
        dataset = self._train_loader.dataset
        batch_size = 200
        tmp_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True).__iter__()
        input, target = next(tmp_loader)

        for c in [0.05, 0.5, 5.0]:
            s = time.time()
            per_sample_grads = self._compute_per_sample_stochastic_grads(input, target)
            clipped_average = self._compute_clipped_average(per_sample_grads, c)
            print("non-tensor clipping", time.time() - s)
            s = time.time()
            per_sample_grads = self._compute_per_sample_stochastic_grads2(input, target)
            clipped_average2 = self._compute_clipped_average2(per_sample_grads, c)
            print("tensor clipping", time.time() - s)
            for g, g2 in zip(clipped_average, clipped_average2):
                assert torch.allclose(g, g2)
        print("Tensor clipping tests passed!")
        assert False

    # CHANGED
    def _compute_clipped_average(self, per_sample_grads, c):
        clipped_per_sample_grads = clip(per_sample_grads, c)
        ldp = ldp_mechanism(clipped_per_sample_grads, 1, self._eps)
        averaged = [layer_grads.mean(dim=0) for layer_grads in ldp]
        return averaged

    def _compute_full_grad_info(self, per_sample_func, c):
        dataset = self._train_loader.dataset
        batch_size = 200
        tmp_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False).__iter__()
        clipped_averages_per_batch = []
        clipping_scales_per_batch = []
        max_norm_lst = []
        for i, data in enumerate(tmp_loader):
            input, target = data
            per_sample_grad_info = per_sample_func(input, target)
            clipped_average = self._compute_clipped_average(per_sample_grad_info, c)
            clipped_averages_per_batch.append(clipped_average)
        averaged = average(clipped_averages_per_batch)
        return averaged

    def compute_per_sample_prev_grads(self, input, target):
        return self._prev_round_snap_shot._compute_per_sample_stochastic_grads(input, target)

    def compute_full_grad_diff(self, c):
        return self._compute_full_grad_info(self._compute_per_sample_grad_diff, c)

    def compute_full_grad(self, c):
        # self._test_tensor_clipping()
        return self._compute_full_grad_info(self._compute_per_sample_stochastic_grads, c)

    def compute_prev_grad(self, c):
        return self._compute_full_grad_info(self.compute_per_sample_prev_grads, c)

    def set_ref_grads(self, grads):
        self._ref_grads = []
        for grad in grads:
            self._ref_grads.append(copy.deepcopy(grad))

    def update_prev_round_snap_shot(self):
        self._prev_round_snap_shot.copy_params(self._model.parameters())

    def _trace_per_sample_smoothness(self):
        dist_params = L2_distance(self._model.parameters(),
                                  self._prev_round_snap_shot._model.parameters()).data.item()
        per_sample_grad_diff_norm = self._trace_per_sample_info(self._compute_per_sample_grad_diff)
        return per_sample_grad_diff_norm / (dist_params + 10 ** (-10))

class CE_PLSGM_Client(CE_PLS_Client):
    def _compute_grad_estimator(self, input, target):
        assert self._n_iters == 1
        return self._ref_grads


class CE_PLS_Server(Server):
    def __init__(self, p0, p1, p2, T, beta, c, c2, **kargs):
        super(CE_PLS_Server, self).__init__(**kargs)
        self._prev_round_model = copy.deepcopy(self._model)
        self._prev_stage_model = copy.deepcopy(self._model)
        self._p0 = p0
        self._p1 = p1
        self._p2 = p2
        self._T = T
        self._beta = beta
        self._c = c
        self._c2 = c2

        self._u = 1.25
        assert self._u > 1

        self._ref_grads = None
        self._cum_dp_global_grad = None
        self._cum_true_global_grad = None
        self._prev_global_grad = None

    def _update(self):
        self._compute_global_grad()
        self._communicate_global_grad()
        self._update_prev_round_snap_shot()
        params = self._one_round_routine()
        self._communicate_params(params)

    def _communicate_params(self, params):
        params = list(params)  # input may be a generator
        for i, optimizer in enumerate(self._optimizers):
            optimizer.copy_params(params)
        self._model = copy.deepcopy(self._optimizers[0]._model)

    # CHANGED
    def _compute_global_grad(self):
        if self._update_count == 0:
            rho = 1
            p = self._p0
        else:
            rho = self._beta
            p = self._p1

        g1 = [optimizer.compute_full_grad_diff(c=self._c)
                              for optimizer in self._optimizers]
        g2 = [optimizer.compute_prev_grad(c=self._c2)
                              for optimizer in self._optimizers]
        sum1 = [add(multiply(1 - rho, g1[i]), multiply(rho, g2[i])) for i in range(len(g1))]

        if self._ref_grads == None:
            self._ref_grads = divide(average(sum1), p)
        else:
            self._ref_grads = add(multiply(1 - rho, self._ref_grads), divide(average(sum1), p))


    def _communicate_global_grad(self):
        for optimizer in self._optimizers:
            optimizer.set_ref_grads(self._ref_grads)

    def _update_prev_round_snap_shot(self):
        for optimizer in self._optimizers:
            optimizer.update_prev_round_snap_shot()
        self._prev_round_model = copy.deepcopy(self._optimizers[0]._model)


# Diff2 with GD-Routine
# T=1 -> DP-GD.
class CE_PLSGM_Server(CE_PLS_Server):
    def _one_round_routine(self):
        i = self._update_count % self._n_workers
        self._optimizers[i].train()
        return self._optimizers[i]._model.parameters()

    def _get_optimizer(self, **kargs):
        assert self._n_local_iters == 1
        return CE_PLSGM_Client(**kargs)

    def _set_sigma(self):
        if self._T > 1:
            coef = 4 * self._u
        else:
            coef = 4
        self._sigma = np.sqrt(
            coef * self._alpha * self._n_global_iters / (self._T * self._n_min ** 2 * self._n_workers ** 2 * self._eps))
