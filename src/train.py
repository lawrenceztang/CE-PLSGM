import argparse
import os
os.environ["OMP_NUM_THREADS"] = "1"

import time
from itertools import accumulate
import pickle
import sys
sys.path.append('../')
from pathlib import Path

import torch
from torch.utils.data.dataset import Subset
import numpy as np

from utils.get_model import create_model
from utils.get_optimizer import get_optimizer
from utils.get_closure import get_optimizer_closure, get_loss_fn
from predict import predict
from utils.load_dataset import load_dataset
import computeamplification as CA
from scipy.optimize import fsolve
    

def get_dim(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def random_split_to_local_datasets(train_data, n_workers):
    train_local_datasets = []
    for p in range(n_workers):
        local_train_dataset = [train_data[i] for i in range(len(train_data)) if i % n_workers == p]
        train_local_datasets.append(local_train_dataset)
    return train_local_datasets


def make_local_loaders(local_datasets, batch_size, shuffle):
    return [torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle) for dataset in local_datasets]
        

def set_random_seed(seed):
    torch.manual_seed(seed)    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)

# CHANGED
def find_epsilon(epsilon_0, k, delta):

    func = lambda epsilon: epsilon / (2 * np.sqrt(k * np.log(np.e + epsilon / delta))) - epsilon_0
    initial_guess = epsilon_0
    epsilon_solution, = fsolve(func, initial_guess)
    return epsilon_solution
    
def train(**kargs):
    set_random_seed(kargs["seed"])
    success_flag = True
    patience_count = 0
    
    train_data, test_data, n_classes, n_dims, task_type = load_dataset(kargs["dataset_name"])
    n_global = len(train_data)

    train_local_datasets = random_split_to_local_datasets(train_data, kargs["n_workers"])
    train_loaders = make_local_loaders(train_local_datasets, kargs["train_batch_size"], shuffle=True)
    pred_loader_on_train_data = torch.utils.data.DataLoader(train_data, kargs["pred_batch_size"], shuffle=False)
    pred_loader_on_test_data = torch.utils.data.DataLoader(test_data, kargs["pred_batch_size"], shuffle=False)
    
    print("The size of local train datasets: ", [len(loader.dataset) for loader in train_loaders])
    model = create_model(kargs["model_name"], n_classes, n_dims)
    model.train()
    d = get_dim(model)

    if kargs["tau"] is None:
        T = None
    else:
        T = max(1, int(kargs["tau"] * kargs["n_global_iters"]))
    if kargs["eps"] is None:
        kargs["eps"] = np.inf

    closure = get_optimizer_closure(task_type)
    loss_fn = get_loss_fn(task_type)
    pred_closure = get_optimizer_closure(task_type, return_output=True)
    optimizer = get_optimizer(optimizer_name=kargs["optimizer_name"],
                              model=model,
                              eta=kargs["eta"],
                              weight_decay=kargs["weight_decay"],
                              train_loaders=train_loaders,
                              gpu_id=kargs["gpu_id"],
                              n_local_iters=kargs["n_local_iters"],
                              n_global_iters=kargs["n_global_iters"],
                              n_workers=kargs["n_workers"],
                              eps=kargs["eps"],
                              delta=kargs["delta"],
                              p0=kargs["p0"],
                              p1=kargs["p1"],
                              p2=kargs["p2"],
                              T=T,
                              beta=kargs["beta"],
                              c=kargs["c"],
                              c2=kargs["c2"],
                              sigma2=kargs["sigma2"],
                              closure=closure,
                              loss_fn=loss_fn)

    # CHANGED
    if kargs["optimizer_name"] == "ce_plsgm":
        n = len(train_loaders[0].dataset) * kargs["n_workers"]
        shuffling_eps = CA.numericalanalysis(n, kargs["eps"], kargs["delta"], 10, 100, True)
        lowerbound_eps = CA.numericalanalysis(n, kargs["eps"], kargs["delta"], 10, 100, False)

        print("Shuffling", n, kargs["eps"], "-DP local randomizers results is (eps, ", kargs["delta"],
              ")-DP in the shuffle model for eps between", lowerbound_eps, "and", shuffling_eps)
        k = kargs["n_global_iters"]

        final_eps = find_epsilon(shuffling_eps, k, kargs["delta"])
        final_delta = kargs["delta"] * 2 * k

        print("The final epsilon, delta is (", final_eps, ",", final_delta, ")")


    saved_info = {"train_loss": [], "train_acc": [], "train_grad_norm": [],
                  "test_loss": [], "test_acc": [], "test_grad_norm": [],
                  "loss_lip": [], "loss_sm": [], "risk_sm": [], "risk_sm2": [], 
                  "args": kargs}

    train_loss, train_acc, train_grad_norm = predict(optimizer.get_model(),
                                                     pred_loader_on_train_data,
                                                     kargs["weight_decay"],
                                                     pred_closure)
    test_loss,  test_acc,  test_grad_norm  = predict(optimizer.get_model(),
                                                     pred_loader_on_test_data,
                                                     0.0,
                                                     pred_closure)
    saved_info["train_loss"     ].append(train_loss)
    saved_info["train_acc"      ].append(train_acc)
    saved_info["train_grad_norm"].append(train_grad_norm)
    saved_info["test_loss"      ].append(test_loss)
    saved_info["test_acc"       ].append(test_acc)
    saved_info["test_grad_norm" ].append(train_grad_norm)
    print(f"Iter: {0} | Train Loss: {train_loss}, Train Acc: {train_acc}, Train Grad Norm {train_grad_norm}, | Test Loss: {test_loss}, Test Acc: {test_acc}, Test Grad Norm: {test_grad_norm}")

    s = time.time()
    for i in range(kargs["n_global_iters"]):
        optimizer.update()
        if (i+1) % kargs["save_intvl"] == 0:
            update_time_per_worker = (time.time() - s)/kargs["n_workers"]

            if kargs["trace"]:
                loss_lip = optimizer.trace_loss_lipschitzness()
                loss_sm  = optimizer.trace_loss_smoothness()
                risk_sm  = optimizer.trace_risk_smoothness()
                risk_sm2 = optimizer.trace_risk_smoothness2()
                saved_info["loss_lip"].append(loss_lip)
                saved_info["loss_sm" ].append(loss_sm )
                saved_info["risk_sm" ].append(risk_sm )
                saved_info["risk_sm2"].append(risk_sm2)
            
            train_loss, train_acc, train_grad_norm = predict(optimizer.get_model(),
                                                             pred_loader_on_train_data,
                                                             kargs["weight_decay"],
                                                             pred_closure)
            test_loss,  test_acc,  test_grad_norm  = predict(optimizer.get_model(),
                                                             pred_loader_on_test_data,
                                                             0.0,
                                                             pred_closure)

            print(f"Iter: {i+1} | Train Loss: {train_loss}, Train Acc: {train_acc}, Train Grad Norm {train_grad_norm}, | Test Loss: {test_loss}, Test Acc: {test_acc}, Test Grad Norm: {test_grad_norm} | Elapsed Time: {time.time() - s}")
            if kargs["trace"]:
                print(f"Loss Lip: {loss_lip}, Loss Sm: {loss_sm}, Risk Sm: {risk_sm}")

            if patience_count >= 5 or np.isnan(train_loss): 
                print("Learning was stopped")
                success_flag =False
                break
                
            prev_train_loss_min = np.min(saved_info["train_loss"])
            if train_loss > 1.05 * prev_train_loss_min:
                patience_count += 1
                print("Patience count", patience_count)
            if train_loss < prev_train_loss_min:
                patience_count = 0
                
            saved_info["train_loss"     ].append(train_loss)
            saved_info["train_acc"      ].append(train_acc)
            saved_info["train_grad_norm"].append(train_grad_norm)
            saved_info["test_loss"      ].append(test_loss)
            saved_info["test_acc"       ].append(test_acc)
            saved_info["test_grad_norm" ].append(train_grad_norm)
            model.train()

    return saved_info, success_flag

                
def get_save_name_info(args, ext="pickle"):
    assert args.n_local_iters == 1
    base_dir = os.path.join(args.optimizer_name)
    tuning_params_name = "_".join(["eta_" + str(args.eta),
                                   "c_" + str(args.c),
                                   "c2_" + str(args.c2),
                                   "tau_" + str(args.tau) + "." + ext])
    return base_dir, tuning_params_name


def none_or_int(value):
    if value == "None":
        return None
    return int(value)


def none_or_float(value):
    if value == 'None':
        return None
    return float(value)


def str_bool(value):
    if value == "True":
        return True
    elif value == "False":
        return False
    else:
        assert False
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Parameters
    parser.add_argument("--eta", type=none_or_float, default="None")
    # General experimental info
    parser.add_argument("--n_workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset_name", type=str, default='california_housing')
    parser.add_argument("--model_name", type=str, default='fc_10')
    parser.add_argument("--optimizer_name", type=str, default='diff2_gd')
    parser.add_argument("--exp_name", type=str, default='test')
    parser.add_argument("--gpu_id", type=int, default=-1)
    parser.add_argument("--n_global_iters", type=int, default=2000)
    parser.add_argument("--p0", type=none_or_float, default="1")
    parser.add_argument("--p1", type=none_or_float, default="1")
    parser.add_argument("--p2", type=none_or_float, default="1")
    parser.add_argument("--tau", type=float, default=0.003)
    parser.add_argument("--beta", type=float, default=0.30)
    parser.add_argument("--c", type=none_or_float, default="10")
    parser.add_argument("--c2", type=none_or_float, default="30")
    parser.add_argument("--sigma2", type=none_or_float, default="30")
    parser.add_argument("--eps", type=none_or_float, default="5.0") #privacy level
    parser.add_argument("--trace", type=str_bool, default="False")
    # Fixed args
    parser.add_argument("--n_local_iters", type=int, default=1)
    parser.add_argument("--train_batch_size", type=none_or_int, default=64)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--pred_batch_size", type=int, default=256)
    parser.add_argument("--save_intvl", type=int, default=20)
    parser.add_argument("--delta", type=float, default=10**(-5)) #privacy level
    args = parser.parse_args()

    save_dir_prefix = os.path.join("results", args.exp_name,
                                   args.model_name,
                                   args.dataset_name,
                                   "eps_"+str(args.eps)+"_delta_"+str(args.delta),
                                   "seed_"+str(args.seed))
    save_dir_suffix, save_name = get_save_name_info(args)

    save_dir = os.path.join(save_dir_prefix, save_dir_suffix)
            
    save_dir = os.path.join(save_dir)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_path = os.path.join(save_dir, save_name)
    if torch.cuda.is_available() and (args.gpu_id >= 0):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        torch.cuda.set_device(args.gpu_id)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("GPU Enabled")
    else:
        print("GPU Not Enabled")

    if args.eta is None:
        saved_info = {}
        for eta in [0.5**i for i in range(10)]:
            args.eta = eta
            print(f"Learning rate eta = {args.eta}")
            _saved_info, success_flag = train(**vars(args))
            saved_info[eta] = _saved_info
            if success_flag:
                break
    else:
        saved_info, success_flag = train(**vars(args))
    with open(save_path, "wb") as f:
        pickle.dump(saved_info, f)
                
        
        
