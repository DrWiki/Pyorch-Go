import scipy.io as scio
import numpy as np
import torch
import os
from sklearn.model_selection import train_test_split

def get_filelist(dir, Filelist, semidir):
    newDir = dir
    if os.path.isfile(dir):
        Filelist.append(dir)
        # # 若只是要返回文件文，使用这个
        # Filelist.append(os.path.basename(dir))
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            # 如果需要忽略某些文件夹，使用以下代码
            # if s == "xxx":
            # continue
            newDir = os.path.join(dir, s)
            if "." in s:
                semidir.append(s)
            get_filelist(newDir, Filelist, semidir)
            print(s)
    return Filelist

if __name__ == '__main__':
    TaskName = ["Yaw", "Pitch", "Roll", "Hybrid"]
    task = 2
    Topic = "qua"
    Train_percentage = 0.6
    data = scio.loadmat(f"./Data/ws{TaskName[task]}.mat")

    X = data['X']
    Y = data['Y'][:, 0:4]

    X_train, X_test, y_train, y_test = train_test_split(X.astype('float32'), Y.astype('int64'), test_size=1-Train_percentage, random_state=666)

    print(Y.shape,X.shape)
    print(y_train.shape, y_test.shape)
    dd = {'train': X_train,
          'val': X_test,
          'label_train': y_train,
          'label_val': y_test}

    torch.save(dd, f'../data/Custom/{TaskName[task]}-{Topic}.pth')
