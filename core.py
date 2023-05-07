import collections
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

WANDB_PADDING = -1

def getKW(**kwargs):
    return kwargs

def flatten_dict(nested, sep='/', prefix=''):
    """Flatten dictionary and concatenate nested keys with separator."""

    def rec(nest, prefix, into):
        for k, v in nest.items():
            if sep in k:
                raise ValueError(f"separator '{sep}' not allowed to be in key '{k}'")
            if isinstance(v, collections.abc.Mapping):
                rec(v, prefix + k + sep, into)
            else:
                into[prefix + k] = v

    flat = {}
    rec(nested, prefix, flat)
    return flat

def flatten_list(tensor_list):
    """Flatten a list of tensors."""
    return torch.cat([t.flatten() for t in tensor_list])

def pad_mask(input_ids, pad_token):
    return torch.where(torch.eq(input_ids, pad_token), 0, 1)

def stack_stat_dicts(stats_dicts):
    """Stack the values of a dict."""
    results = dict()
    for k in stats_dicts[0]:
        # if stats_dicts[0][k] is a string
        if isinstance(stats_dicts[0][k], str) or "config" in k or "param" in k:
            results[k] = stats_dicts[0][k]
        elif isinstance(stats_dicts[0][k], torch.Tensor):
            stats_list = [torch.flatten(d[k]) for d in stats_dicts]
            results[k] = pad_sequence(stats_list, batch_first=True, padding_value=WANDB_PADDING)
        elif isinstance(stats_dicts[0][k], np.ndarray):
            stats_list = [d[k].flatten() for d in stats_dicts]
            results[k] = stats_list
        else:
            results[k] = [d[k] for d in stats_dicts]
    return results

def stack_dicts_list(dicts):
    results = dict()
    for k in dicts[0]:
        results[k] = [d[k] for d in dicts]
    return results


def add_suffix(input_dict, suffix):
    """Add suffix to dict keys."""
    return dict((k + suffix, v) for k, v in input_dict.items())


def pad_to_size(tensor, size, dim=1, padding=50256):
    """Pad tensor to size."""
    t_size = tensor.size()[dim]
    if t_size == size:
        return tensor
    else:
        return torch.nn.functional.pad(tensor, (0, size - t_size), 'constant', padding)


def logprobs_from_logits(logits, labels):
    """
    See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591
    """
    logp = F.log_softmax(logits, dim=2)
    logpy = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    return logpy

def qidx_from_qs(qs, labels):
    qidx = torch.gather(qs, -1, labels.unsqueeze(-1)).squeeze(-1)
    return qidx

def whiten(values, shift_mean=True, mean=None, var=None):
    """Whiten values."""
    # single element is equal to mean
    if values.shape[1] == 1:
        if not shift_mean:
            return values
        return torch.tensor(0, dtype=values.dtype)

    if mean is None:
        mean = torch.mean(values)
    if var is None:
        var = torch.var(values)

    # 1e-8 is too small for fp16
    whitened = (values - mean) * torch.rsqrt(var + 1e-6)
    if not shift_mean:
        whitened += mean
    return whitened

def whitenBatch(valuesList, shift_mean=True, mean=None, var=None):
    """Whiten values."""
    # single element is equal to mean
    if var is None or mean is None:
        flat = flatten_list(valuesList)
        if mean is None:
            mean = torch.mean(flat)
        if var is None:
            var = torch.var(flat)

    whitenedList = []
    for values in valuesList:
        whitened = whiten(values, shift_mean=shift_mean, mean=mean, var=var)
        whitenedList.append(whitened)
    return whitenedList

def meanGlobal(tensor, rank=0, world_size=1):
    """Calculate mean of tensor over all processes."""
    if isinstance(tensor, list):
        tensor = flatten_list(tensor)
    device = tensor.device
    size = torch.prod(torch.tensor(list(tensor.shape), device=device))
    tensor = torch.sum(tensor)
    # copilot added a clone, not sure if needed
    tensor = tensor.clone()
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    size = size.clone()
    torch.distributed.all_reduce(size, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor.to(device)
    size = size.to(device)
    tensor /= size
    return tensor

def varGlobal(tensor, mean, rank=0, world_size=1):
    """Calculate variance of tensor over all processes."""
    if isinstance(tensor, list):
        tensor = flatten_list(tensor)
    device = tensor.device
    size = torch.prod(torch.tensor(list(tensor.shape), device=device))
    tensor = torch.sum((tensor - mean) ** 2)
    tensor = tensor.clone()
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    size = size.clone()
    torch.distributed.all_reduce(size, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor.to(device)
    size = size.to(device)
    tensor /= size - 1
    return tensor

def whitenGlobal(valuesList, rank=0, world_size=1, shift_mean=True):
    """Whiten values."""
    mean = meanGlobal(valuesList, rank=rank, world_size=world_size)
    var = varGlobal(valuesList, mean, rank=rank, world_size=world_size)
    
    return whitenBatch(valuesList, shift_mean=shift_mean, mean=mean, var=var)


def clip_by_value(x, tensor_min, tensor_max):
    """
    Tensor extenstion to torch.clamp
    https://github.com/pytorch/pytorch/issues/2793#issuecomment-428784713
    """
    clipped = torch.max(torch.min(x, tensor_max), tensor_min)
    return clipped


def entropy_from_logits(logits):
    """Calculate entropy from logits."""
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, axis=-1) - torch.sum(pd * logits, axis=-1)
    return entropy


def average_torch_dicts(list_of_dicts):
    """Average values of a list of dicts wiht torch tensors."""
    average_dict = dict()
    for key in list_of_dicts[0].keys():
        average_dict[key] = torch.mean(torch.stack([d[key] for d in list_of_dicts]), axis=0)
    return average_dict

def stats_to_cpu(stats_dict):
    """Cast all torch.tensors in dict to detached cpu tensors."""
    new_dict = dict()
    for k, v in stats_dict.items():
        if isinstance(v, torch.Tensor):
            new_dict[k] = v.detach().cpu()
        else:
            new_dict[k] = v
    return new_dict

def stats_to_np(stats_dict):
    """Cast all torch.tensors in dict to numpy arrays."""
    new_dict = dict()
    for k, v in stats_dict.items():
        if isinstance(v, torch.Tensor):
            new_dict[k] = v.detach().cpu().numpy()
        else:
            new_dict[k] = v
        if np.isscalar(new_dict[k]) and not isinstance(new_dict[k], str):
            new_dict[k] = float(new_dict[k])
    return new_dict


def listify_batch(tensor):
    """Turns the first dimension of a tensor into a list."""
    return [tensor[i] for i in range(tensor.shape[0])]


def build_bert_batch_from_txt(text_list, tokenizer, device):
    """Create token id and attention mask tensors from text list for BERT classification."""

    # tokenize
    tensors = [tokenizer.encode(txt, return_tensors="pt").to(device) for txt in text_list]

    # find max length to pad to
    max_len = max([t.size()[1] for t in tensors])

    # get padded tensors and attention masks
    # (attention masks make bert ignore padding)
    padded_tensors = []
    attention_masks = []
    for tensor in tensors:
        attention_mask = torch.ones(tensor.size(), device=device)
        padded_tensors.append(pad_to_size(tensor, max_len, padding=0))
        attention_masks.append(pad_to_size(attention_mask, max_len, padding=0))

    # stack all tensors
    padded_tensors = torch.cat(padded_tensors)
    attention_masks = torch.cat(attention_masks)

    return padded_tensors, attention_masks

# https://github.com/jdb78/pytorch-forecasting/blob/master/pytorch_forecasting/utils.py
def padded_stack(tensors, side: str = "left", mode: str = "constant", value=0):
    """
    Stack tensors along first dimension and pad them along last dimension to ensure their size is equal.
    Args:
        tensors (List[torch.Tensor]): list of tensors to stack
        side (str): side on which to pad - "left" or "right". Defaults to "right".
        mode (str): 'constant', 'reflect', 'replicate' or 'circular'. Default: 'constant'
        value (Union[int, float]): value to use for constant padding
    Returns:
        torch.Tensor: stacked tensor
    """
    full_size = max([x.size(-1) for x in tensors])
    # print(full_size)
    # for x in tensors:
    #     print(x.shape)

    def make_padding(pad):
        if side == "left":
            return (pad, 0)
        elif side == "right":
            return (0, pad)
        else:
            raise ValueError(f"side for padding '{side}' is unknown")

    out = torch.stack(
        [
            F.pad(x, make_padding(full_size - x.size(-1)), mode=mode, value=value) if full_size - x.size(-1) > 0 else x
            for x in tensors
        ],
        dim=0,
    )
    return out
