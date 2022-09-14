
import models, utils
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import sklearn.metrics as skm

# def train_epoch(model, optimizer, train_dataloader,show_interval=10, device = "cuda"):
#     model.train()
#     acc_meter, loss_meter, it_count = 0, 0, 0
#     for inputs, target in train_dataloader:
#         inputs = inputs.to(torch.float32).to(device)
#         target = target.to(torch.int64).to(device)
#
#         optimizer.zero_grad()
#         output = F.log_softmax(model(inputs), dim=1)
#         loss = F.nll_loss(output, target)
#         loss.backward()
#         optimizer.step()
#
#         loss_meter += loss.item()
#         pred = output.argmax(dim=1)
#         acc = torch.eq(pred, target).sum().float().item() / len(inputs)
#         acc_meter += acc
#         it_count += 1
#         if it_count != 0 and it_count % show_interval == 0:
#             print("%d, loss: %.3e acc: %.3f" % (it_count, loss.item(), acc))
#
#     return loss_meter / it_count, acc_meter / it_count

def train_epoch(model, optimizer, train_dataloader,device = "cuda"):
    model.train()
    acc_meter, loss_meter, it_count = 0, 0, 0
    for inputs, target in train_dataloader:
        inputs = inputs.to(torch.float32)
        target = target.to(torch.int64)
        inputs = inputs.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = F.log_softmax(model(inputs), dim=1)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        loss_meter += loss.item()
        pred = output.argmax(dim=1)
        acc = torch.eq(pred, target).sum().float().item() / len(inputs)
        acc_meter += acc
        it_count += 1
        print("%d, loss: %.3e acc: %.3f" % (it_count, loss.item(), acc))
        # if ep % 10==0:
        #     print(skm.confusion_matrix(target.cpu(), output.argmax(dim=1).cpu()))
        #     ConfusionMat.showMat(skm.confusion_matrix(target.cpu(), output.argmax(dim=1).cpu()), range(0,60),"Train",str(ep))

    return loss_meter / it_count, acc_meter / it_count


def train_epoch_fit(model, optimizer, train_dataloader, device = "cuda"):
    model.train()
    acc_meter, loss_meter, it_count = 0, 0, 0
    # num = 0

    for inputs, target in train_dataloader:
        # num+=1
        # print(num)
        inputs = inputs.to(torch.float32)
        target = target.to(torch.float32)
        inputs = inputs.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        output = (model(inputs))
        # print(output.shape)
        # print(target.shape)
        # print(output.squeeze(1).shape)
        # print(target.unsqueeze(1).shape)
        loss_fn = nn.MSELoss()
        loss = loss_fn(output.squeeze(1), target)
        # loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        loss_meter += loss.item()
        pred = output.argmax(dim=1)
        acc = torch.eq(pred, target).sum().float().item() / len(inputs)
        acc_meter += acc
        it_count += 1
    return loss_meter / it_count