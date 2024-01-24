import torch


def get_optimizer_closure(task_type, return_output=False, retain_graph=False):
    return lambda data, target, model: general_closure(data, target, model,
                                                task_type, return_output, retain_graph)


def general_closure(data, target, model, task_type, return_output, retain_graph):
    model.zero_grad()
    output = model(data)
    if task_type == "classification":
        loss = torch.nn.CrossEntropyLoss()(output, target)
    elif task_type == "regression":
        target = target.view(output.size())
        loss = torch.nn.MSELoss()(output, target)
    else:
        assert False
    loss.backward(retain_graph=retain_graph)
    if return_output:
        return loss, output
    return loss


def get_loss_fn(task_type):
    if task_type == "classification":
        return lambda output, target: torch.nn.CrossEntropyLoss()(output, target)
    elif task_type == "regression":
        return lambda output, target: torch.nn.MSELoss()(output, target.view(output.size()))
    else: 
        assert False
