# generates safetensor file for a simple network.
# run this in "test_assets\minst" directory

import torch
from torch import nn
import torch.nn.functional as F

import idx2numpy

import safe_utils

torch.manual_seed(98172356)

num_samples = 10000
num_pixels = 784
num_hidden = 50

images_file = '../mnist/t10k-images.idx3-ubyte'
labels_file = '../mnist/t10k-labels.idx1-ubyte'

images = idx2numpy.convert_from_file(images_file).copy()
labels = idx2numpy.convert_from_file(labels_file).copy()

x_train = torch.from_numpy(images).float().reshape(num_samples,num_pixels) * (1/255.)
y_train = torch.tensor(labels, dtype=torch.long)

model = nn.Sequential(nn.Linear(num_pixels,num_hidden), nn.ReLU(), nn.Linear(num_hidden,10))
loss_func = F.cross_entropy
lr = 0.2

def accuracy(out, yb): return (out.argmax(dim=1)==yb).float().mean()

def report(loss, preds, yb): print(f'{loss:.2f}, {accuracy(preds, yb):.2f}')

def fit():
    bs = 64
    epochs = 2
    for epoch in range(epochs):
        for i in range(0, num_samples, bs):
            s = slice(i, min(num_samples,i+bs))
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
print("w1.T:", w1.T, "b1", b1)

print ("l1 manual actications:", xs@w1.T + b1)

safe_utils.save(model)

