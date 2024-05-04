import torch
import fastcore.all as fc
from safetensors import safe_open
from safetensors.torch import save_file


idx = 0

def add_layer(layer, flat):
    global idx
    if layer == None: return
    if layer == fc.noop: return
    params = list(layer.parameters())
    if len(params) == 0: return

    print("LAYER:", idx, layer)
    flat.append((layer, idx))
    idx += 1
    
def flatten_layer(layer, flat):
    global idx

    if layer == fc.noop:
        return

    if isinstance(layer, torch.nn.modules.container.Sequential):
        for ll in layer:
            flatten_layer(ll, flat)
    else:
        add_layer(layer, flat)

def flatten_model(model):
    flat = []
    for l in model:
        flatten_layer(l, flat)
    return flat

def get_tensors(flat_model):
    tensors = {}
    for l,idx in flat_model:
        name = type(l).__name__.split('.')[-1] + "-" + str(idx) + "-"
        pidx = 0
        for p in l.parameters():
            pname = name + str(pidx)
            tensors[pname] = p.to('cpu')
            pidx += 1
    return tensors

def save_model(tensors, filename):
    print("saving ", len(tensors), " tensors to ", filename)
    save_file(tensors, filename)

def print_tensors(tensors):
    for name,tensor in tensors.items():
    	print(name, tensor.shape, end=" ")
    	if len(tensor.shape) == 1: print(tensor[:6])
    	if len(tensor.shape) == 2: print(tensor[0, 0:6])


def save(model, filename):
    flat_model = flatten_model(model)
    tensors = get_tensors(flat_model)
    save_model(tensors, filename)
    print_tensors(tensors)
