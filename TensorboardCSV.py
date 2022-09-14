from tensorboard.backend.event_processing import event_accumulator
import argparse
import pandas as pd
from tqdm import tqdm
import os

# import os
# import cv2
# import numpy
# import shutil
#
#
# def listdir(path, list_name, list_name2, type = '.jpg'):  # , list_name2):
#     for file in os.listdir(path):
#         file_path = os.path.join(path, file)
#         if os.path.isdir(file_path):
#             listdir(file_path, list_name, list_name2,type=type)  # list_name2)
#         elif os.path.splitext(file_path)[1] == type:
#             list_name.append(file_path)
#             list_name2.append(file)
#
#
#         path = "./4.2/tupian/"
#         filelist=[]
#         filelist2=[]
#         listdir(path,filelist,filelist2, type= ".jpg")
#         count = 0
#         for i in range(len(filelist)):
#             print(filelist[i])
#             print(filelist2[i])
#
#
#         for i in range(len(filelist)):
#             count = count + 1
#             # MainName = file.split(".")[0]
#             MainName = filelist2[i][0:len(filelist2[i])-4]
#             realPath =  filelist[i][0:len(filelist[i])-len(filelist2[i])-1]+"/"
#             print(MainName, realPath)
#             # 旧图像名字
#             Olddir = filelist[i]
#             if os.path.isdir(Olddir):
#                 continue
#             # filename = os.path.splitext(filelist2[i])[0]
#             # 新图像名字
#             Newdir = os.path.join(realPath, str(count).zfill(6) + str(".jpg"))
#             # 旧标签的名字
#             Olddir2 = realPath.replace("tupian","YOLOlabel") + MainName + ".xml"
#             # 新标签的名字
#             Newdir2 = realPath.replace("tupian","YOLOlabel") + str(count).zfill(6) + ".xml"
#             # # 操作不可逆，风险较大，正式操作之前的检验打印，检验无误，取消紧随两行的注释再次运行即可
#             print("NUM: ", count)
#             print(Olddir)
#             print(Newdir)
#             print(Olddir2)
#             print(Newdir2)
#             os.rename(Olddir, Newdir)
#             os.rename(Olddir2, Newdir2)
#         print("Finally！！！！！")


def listdir(path, list_name, list_name2, type = '.jpg'):  # , list_name2):
    if type is None:
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if os.path.isdir(file_path):
                list_name.append(file_path)
                list_name2.append(file)

    elif type==".xxx":
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if os.path.isdir(file_path):
                listdir(file_path, list_name, list_name2,type=type)  # list_name2)
            else:
                list_name.append(file_path)
                list_name2.append(file)
    else:
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if os.path.isdir(file_path):
                listdir(file_path, list_name, list_name2,type=type)  # list_name2)
            elif os.path.splitext(file_path)[1] == type:
                list_name.append(file_path)
                list_name2.append(file)



if __name__ == '__main__':
    tasklist = ["qua","quatqdq"]
    for k in range(len(tasklist)):
        namelist = []
        namelist2 = []
        listdir(f"./log/{tasklist[k]}",namelist,namelist2,type=None)
        for i in range(len(namelist)):
            print(namelist[i])
            filelist = []
            filelist2 = []
            listdir(namelist[i], filelist, filelist2, type = ".xxx")
            for j in range(len(filelist)):
                print(filelist[j])
                print(filelist2[j])
                event_data = event_accumulator.EventAccumulator(filelist[j]) # a python interface for loading Event data
                event_data.Reload()  # synchronously loads all of the data written so far b
                # print(event_data.Tags())  # print all tags
                keys = event_data.scalars.Keys()  # get all tags,save in a list
                # print(keys)
                df = pd.DataFrame(columns=keys[1:])  # my first column is training loss per iteration, so I abandon it
                for key in tqdm(keys):
                    # print(key)
                    if key != 'train/total_loss_iter':  # Other attributes' timestamp is epoch.Ignore it for the format of csv file
                        df[key] = pd.DataFrame(event_data.Scalars(key)).value
                if not os.path.exists(filelist[j].replace("./log","./log/CSV").replace(filelist2[j],"")): os.makedirs(filelist[j].replace("./log","./log/CSV").replace(filelist2[j],""))
                df.to_csv(filelist[j].replace("./log","./log/CSV")+".csv")

                print(filelist[j]+".csv")

        print("Tensorboard data exported successfully")
        # print(namelist2[i])





