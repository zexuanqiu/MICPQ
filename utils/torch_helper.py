import torch

def move_to_device(obj, device):
    res = {}
    for k, v in obj.items():
        res[k] = v.to(device)
    return res


def squeeze_dim(obj, dim):
    if torch.is_tensor(obj):
        return torch.squeeze(obj, dim=dim)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items():
            res[k] = squeeze_dim(v, dim)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj:
            res.append(squeeze_dim(v, dim))
        return res
    else:
        raise TypeError("Invalid type for move_to_device")
