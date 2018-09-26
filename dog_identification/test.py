# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 20:47:39 2018

@author: Jacob
"""
import os
import shutil
#import numpy as np
import pandas as pd

src_dir = './data/train/'
dst_dir = './data/train_classified/'
#读取文件名和种类对应到两个列表
def start_move():
    data = pd.read_csv('./data/labels.csv')
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for index in data.index:
        class_dir = os.path.join(dst_dir,data.loc[index].values[1])
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)
        src_file = os.path.join(src_dir,data.loc[index].values[0]+'.jpg')
        dst_file = os.path.join(class_dir,data.loc[index].values[0]+'.jpg')
        if not os.path.exists(src_file):
            print("%s not exist!"%(src_file))
            break
        shutil.copyfile(src_file,dst_file)
        
if __name__ == '__main__':
    start_move()