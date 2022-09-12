import os
import time
import shutil
import utils
import models.CNN1Dnet as CNN1Dnet
import torch
from torch import optim
from torch.utils.data import DataLoader
# import visdom
import TENGDataset as TENGDataset
import myVal
import myTrain
import tensorboard_logger.tensorboard_logger as Logger
import pandas as pd



# datalen
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(111)
torch.cuda.manual_seed(111)

def save_ckpt(state, is_best, model_save_dir):
    current_w = os.path.join(model_save_dir, 'current_w.pth')
    best_w = os.path.join(model_save_dir, 'best_w.pth')
    torch.save(state, current_w)
    if is_best: shutil.copyfile(current_w, best_w)

if __name__ == '__main__':
    # 加载数据
    namelist = ["ES","QV","ESG7","QVG7","Scope60","Scope24"]
    classlist = [60,60,7,7,60,24]
    tasknum = 0
    task = {"name":namelist[tasknum], "class":classlist[tasknum]}
    datapath = "./data/Custom/normal{}.pth".format(task["name"])
    print(datapath)

    ## 训练数据
    train_dataset = TENGDataset.TENGDataset(datapath, train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=int(len(train_dataset)/10), shuffle=True, num_workers=1)
    ## 测试数据
    val_dataset = TENGDataset.TENGDataset(datapath, train=False)
    val_dataloader = DataLoader(val_dataset, batch_size= len(val_dataset), shuffle=True, num_workers=1)

    # 加载模型
    print("train data size:", len(train_dataset), ",val data size:", len(val_dataset))
    model = CNN1Dnet.Conv1DNet(num_classes=task["class"])
    model = model.to(device)
    print(model)
    exit()
    model_save_dir = f'./log/{task["name"]}/CNN1D-{time.strftime("%Y%m%d%H%M")}'
    if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)
    # 加载超参数
    best_lost = -1
    lr = 1e-3
    start_epoch = 0
    stage = 1
    # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.5)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)
    logger = Logger.Logger(logdir=model_save_dir, flush_secs=2)
    for epoch in range(start_epoch, 100):
        since = time.time()
        train_loss, train_acc = myTrain.train_epoch(model, optimizer, train_dataloader, device)
        val_loss, val_acc, confusionMatrix = myVal.val_epoch_m(model, val_dataloader, device)
        pd.DataFrame(confusionMatrix).to_csv(f'./matrix_output/{task["name"]}-epoch-{epoch}-confusionMatrix.csv')
        print('#epoch: %02d stage: %d train_loss: %.3e train_acc: %.3f  val_loss: %0.3e val_acc: %.3f time: %s\n'
              % (epoch, stage, train_loss, train_acc, val_loss, val_acc, utils.print_time_cost(since)))
        logger.log_value('train_loss', train_loss, step=epoch)
        logger.log_value('train_acc', train_acc, step=epoch)
        logger.log_value('val_loss', val_loss, step=epoch)
        logger.log_value('val_acc', val_acc, step=epoch)
        logger.log_value('leraning_rate', optimizer.state_dict()['param_groups'][0]['lr'], step=epoch)
        scheduler.step()

