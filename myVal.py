import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import sklearn.metrics as skm
import numpy as np
import ConfusionMat
import pandas as pd
def val_epoch(model, val_dataloader, device = "cuda"):
    model.eval()
    acc_meter, loss_meter, it_count = 0, 0, 0
    with torch.no_grad():
        for inputs, target in val_dataloader:
            inputs = inputs.to(torch.float32)
            target = target.to(torch.int64)
            inputs = inputs.to(device)
            target = target.to(device)
            # optimizer.zero_grad()
            output = model(inputs)

            # print("ValCheck:", F.log_softmax(output.cpu(), dim=1).argmax(dim=1).detach().numpy())
            # print("ValCheck:", output.cpu())
            # print("ValCheck:", inputs.cpu())
            output = F.log_softmax(output, dim=1)
            loss = F.nll_loss(output, target)
            loss_meter += loss.item()
            pred = output.argmax(dim=1)
            acc = torch.eq(pred, target).sum().float().item() / len(inputs)
            acc_meter += acc
            it_count += 1
            confusionMatrix = skm.confusion_matrix(target.cpu(), output.argmax(dim=1).cpu())
            pd.DataFrame(confusionMatrix).to_csv('sample.csv')

    # print(skm.confusion_matrix(target.cpu(), output.argmax(dim=1).cpu()))
    # ConfusionMat.showMat(skm.confusion_matrix(target.cpu(), output.argmax(dim=1).cpu()), range(0,24),"Val24",str(epoch))
            # print(ar.shape)
            #viz.image(ar.transpose(2,0,1), win="CM",opts={"title":"Confusionmat"})

    return loss_meter / it_count, acc_meter / it_count


def val_epoch_m(model, val_dataloader, device="cuda"):
    model.eval()
    acc_meter, loss_meter, it_count = 0, 0, 0
    confusionMatrix = None
    with torch.no_grad():
        for inputs, target in val_dataloader:
            inputs = inputs.to(torch.float32)
            target = target.to(torch.int64)
            inputs = inputs.to(device)
            target = target.to(device)
            # optimizer.zero_grad()
            output = model(inputs)

            # print("ValCheck:", F.log_softmax(output.cpu(), dim=1).argmax(dim=1).detach().numpy())
            # print("ValCheck:", output.cpu())
            # print("ValCheck:", inputs.cpu())
            output = F.log_softmax(output, dim=1)
            loss = F.nll_loss(output, target)
            loss_meter += loss.item()
            pred = output.argmax(dim=1)
            acc = torch.eq(pred, target).sum().float().item() / len(inputs)
            acc_meter += acc
            it_count += 1
            # print(target.cpu())
            # temp = target.cpu().numpy()
            # cont = np.zeros((60,1))
            # for i in range(temp.shape[0]):
            #     for j in range(60):
            #         if temp[i]==j:
            #             cont[j,0] = cont[j,0]+1
            #             break
            # print(cont)

            confusionMatrix = skm.confusion_matrix(target.cpu(), output.argmax(dim=1).cpu())

    # print(skm.confusion_matrix(target.cpu(), output.argmax(dim=1).cpu()))
    # ConfusionMat.showMat(skm.confusion_matrix(target.cpu(), output.argmax(dim=1).cpu()), range(0,24),"Val24",str(epoch))
    # print(ar.shape)
    # viz.image(ar.transpose(2,0,1), win="CM",opts={"title":"Confusionmat"})

    return loss_meter / it_count, acc_meter / it_count, confusionMatrix