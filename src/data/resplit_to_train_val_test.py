import numpy as np
import os 
import cv2
import random
import sys 
import csv
import pandas as pd
import bz2
from bz2 import BZ2File
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from glob import glob

from sklearn.metrics import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelBinarizer
import shutil

input_folder = '/data/home/luoy/project/python/datasets/harvard/Harvard-30K/AMD/all_samples'

output_folder = '/data/home/luoy/project/python/datasets/harvard/Harvard-30K/AMD'

split_numbers = [6000, 1000, 3000]

tmp_output_folder = os.path.join(output_folder, 'train')
isExist = os.path.exists(tmp_output_folder)
if not isExist:
    os.makedirs(tmp_output_folder)

tmp_output_folder = os.path.join(output_folder, 'val')
isExist = os.path.exists(tmp_output_folder)
if not isExist:
    os.makedirs(tmp_output_folder)

tmp_output_folder = os.path.join(output_folder, 'test')
isExist = os.path.exists(tmp_output_folder)
if not isExist:
    os.makedirs(tmp_output_folder)

fundus_files = []
iris_files = []
data_files = []
data_ids = []
for f in os.listdir(input_folder):
    if f.endswith('.npz'):
        img_id = f[f.find('_'):f.find('.')]
        data_ids.append(img_id)
#         data_files.append(f)
#         img_id = f[f.find('_'):f.find('.')]
#         fundus_files.append(f'{"fundus"}{img_id}.jpg')
#         iris_files.append(f'{"iris"}{img_id}.jpg')

index_shuf = list(range(len(data_ids)))
random.shuffle(index_shuf)

tmp_output_folder = os.path.join(output_folder, 'train')
for i in range(0, split_numbers[0]):
    cur_index = index_shuf[i]
    f_data = f"data_{cur_index:06d}.npz"
    f_slo = f"slo_{cur_index:06d}.jpg"
    
    shutil.copy(os.path.join(input_folder, f_data), os.path.join(tmp_output_folder, f_data))
    shutil.copy(os.path.join(input_folder, f_slo), os.path.join(tmp_output_folder, f_slo))

tmp_output_folder = os.path.join(output_folder, 'val')
for i in range(split_numbers[0], split_numbers[0]+split_numbers[1]):
    cur_index = index_shuf[i]
    f_data = f"data_{cur_index:06d}.npz"
    f_slo = f"slo_{cur_index:06d}.jpg"
    
    shutil.copy(os.path.join(input_folder, f_data), os.path.join(tmp_output_folder, f_data))
    shutil.copy(os.path.join(input_folder, f_slo), os.path.join(tmp_output_folder, f_slo))

tmp_output_folder = os.path.join(output_folder, 'test')
for i in range(split_numbers[0]+split_numbers[1], split_numbers[0]+split_numbers[1]+split_numbers[2]):
    cur_index = index_shuf[i]
    f_data = f"data_{cur_index:06d}.npz"
    f_slo = f"slo_{cur_index:06d}.jpg"
    
    shutil.copy(os.path.join(input_folder, f_data), os.path.join(tmp_output_folder, f_data))
    shutil.copy(os.path.join(input_folder, f_slo), os.path.join(tmp_output_folder, f_slo))

print('all done')