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
    namelist = ["Hybrid-qua",  "Yaw-qua", "Pitch-qua","Roll-qua"]
    modelpath = ["./log/Train_model/Yaw-qua-202209140054/Yaw-qua-Model_Fit_sss-202209140054-epoch-99.pth",
                "./log/Train_model/Pitch-qua-202209140056/Pitch-qua-Model_Fit_sss-202209140056-epoch-99.pth",
                "./log/Train_model/Roll-qua-202209140058/Roll-qua-Model_Fit_sss-202209140058-epoch-99.pth",
                "./log/Train_model/Hybrid-qua-202209140052/Hybrid-qua-Model_Fit_sss-202209140052-epoch-99.pth"]

    outnum = [4, 4,4,4]
    # task = 7
    for modelindex in range(4):
        for task in range(4):
            # classlist = [60,60,7,7,60,24]
            # tasknum = 0
            # task = {"name":namelist[tasknum], "class":classlist[tasknum]}
            datapath = f"./data/Custom/{namelist[task]}.pth"
            print(datapath)
            ## 训练数据
            train_dataset = TENGDataset.TENGDataset(datapath, train=True)
            train_dataloader = DataLoader(train_dataset, batch_size=int(len(train_dataset)/5), shuffle=True, num_workers=1)
            ## 测试数据
            val_dataset = TENGDataset.TENGDataset(datapath, train=False)
            val_dataloader = DataLoader(val_dataset, batch_size= len(val_dataset), shuffle=True, num_workers=1)

            # 加载模型
            print("train data size:", len(train_dataset), ",val data size:", len(val_dataset))
            model = CNN1Dnet.Model_Fit_sss(outnum[task])
            state = torch.load(modelpath[modelindex])
            model_dict = model.load_state_dict(state["state_dict"])
            model = model.to(device)
            # print(model)
            time_stamp = time.strftime("%Y%m%d%H%M")
            # model_save_dir = f'./log/{namelist[task]}-{time_stamp}/{namelist[task]}-{model.name()}-{time_stamp}'
            #
            # if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)
            # if not os.path.exists(model_save_dir2): os.makedirs(model_save_dir2)
            # logger = Logger.Logger(logdir=model_save_dir, flush_secs=2)
            val_loss = myVal.val_epoch_fit_test(model, val_dataloader, device)
            print(modelpath[modelindex] + "   "+ namelist[task] + ' # val_loss: %0.3e \n'% ( val_loss))
            # logger.log_value('train_loss', train_loss, step=epoch)
            # logger.log_value('val_loss', val_loss, step=epoch)
            # logger.log_value('leraning_rate', optimizer.state_dict()['param_groups'][0]['lr'], step=epoch)
