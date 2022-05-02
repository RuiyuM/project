

# path=input('请输入文件路径(结尾加上/)：')C:\Users\maoda\.kaggle\data\images

import os
import sys


# 修改生成的文件名字
def rename():
    path = input("请输入路径(例如D:\\\\picture)：")
    name = input("请输入开头名:")
    startNumber = input("请输入开始数:")
    fileType = input("请输入后缀名（如 .jpg、.txt等等）:")
    print("正在生成以" + name + startNumber + fileType + "迭代的文件名")
    count = 0
    filelist = os.listdir(path)
    for files in filelist:
        Olddir = os.path.join(path, files)
        if os.path.isdir(Olddir):
            continue
        Newdir = os.path.join(path, name + str(count + int(startNumber))+'_mask' + fileType)
        os.rename(Olddir, Newdir)
        count += 1
    print("一共修改了" + str(count) + "个文件")


rename()