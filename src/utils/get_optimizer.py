import sys
sys.path.append('../')

from optimizers import Diff2_GD_Server, CE_PSGM_Server


def get_optimizer(optimizer_name, **kargs):
    optimizers = {"diff2_gd": Diff2_GD_Server, #Diff2 with GD-Routine
                  "ce_plsgm": CE_PSGM_Server
                  }
    return optimizers[optimizer_name](**kargs)
