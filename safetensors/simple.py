# generates safetensor file for a simple network.
# run this in "test_assets\minst" directory

import torch
from torch import nn
import torch.nn.functional as F


torch.manual_seed(98172356)


x_train = torch.tensor([[1,2,3,6,6,6,9,8,7]]).float();
y_train = torch.tensor([2])

model = nn.Sequential(nn.Linear(9,4), nn.ReLU(), nn.Linear(4,5))
loss_func = F.cross_entropy
lr = 0.4

def accuracy(out, yb): return (out.argmax(dim=1)==yb).float().mean()

def report(loss, preds, yb): print(f'{loss:.2f}, {accuracy(preds, yb):.2f}')

def fit():
    n = 64
    bs = 64
    epochs = 10
    for epoch in range(epochs):
        for i in range(0, n, bs):
            s = slice(i, min(n,i+bs))
            xb,yb = x_train[s],y_train[s]
            preds = model(xb)
            loss = loss_func(preds, yb)
            loss.backward()
            with torch.no_grad():
                for p in model.parameters(): p -= p.grad * lr
                model.zero_grad()
        report(loss, preds, yb)

fit()


xs = x_train[:1]
l1 = list(model.modules())[1]
print("l1:", l1, "xs.shape:", xs.shape)
preds = l1(xs)
print("preds.shape:", preds.shape, "preds", preds)

w1 = list(l1.parameters())[0]
b1 = list(l1.parameters())[1]

print("w1", w1)
print("w1.T:", w1.T)
print("b1", b1)

print ("l1 manual actications:", xs@w1.T + b1)


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


def save(model):
    flat_model = flatten_model(model)
    tensors = get_tensors(flat_model)
    save_model(tensors, "simple.safetensors")
    print_tensors(tensors)

save(model)

