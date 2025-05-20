'''
Description: 
Author: Xiongjun Guan
Date: 2025-05-19 17:01:29
version: 0.0.1
LastEditors: Xiongjun Guan
LastEditTime: 2025-05-19 19:36:16

Copyright (C) 2025 by Xiongjun Guan, Tsinghua University. All rights reserved.
'''
import numpy as np

img_dir_dict = {"FVC04DB1": "/data/panzhiyu/fingerprint/FVC04DB1/image/query/"}
mnt_dir_dict = {"FVC04DB1": "/data/panzhiyu/fingerprint/FVC04DB1/mnt/query/"}

data_lst = ["FVC04DB1"]
for key in data_lst:
    img_dir = img_dir_dict[key]
    mnt_dir = mnt_dir_dict[key]
