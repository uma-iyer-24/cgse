import torch

def refresh_optimizer(old_optimizer, model):
    """
    Rebuild optimizer while preserving state for existing parameters.
    New parameters start fresh.
    """

    # capture optimizer class + hyperparams
    opt_class = type(old_optimizer)
    defaults = old_optimizer.defaults

    # build new optimizer on current model params
    new_optimizer = opt_class(model.parameters(), **defaults)

    # ---- migrate state ----
    old_state = old_optimizer.state
    new_state = new_optimizer.state

    # map param tensors by id
    old_params = {id(p): p for group in old_optimizer.param_groups for p in group['params']}

    for group in new_optimizer.param_groups:
        for p in group['params']:
            pid = id(p)

            if pid in old_params and old_params[pid] in old_state:
                # copy optimizer state for unchanged parameter
                new_state[p] = {
                    k: (v.clone() if torch.is_tensor(v) else v)
                    for k, v in old_state[old_params[pid]].items()
                }

    return new_optimizer

