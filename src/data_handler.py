import sys, os
from PIL import Image
import blobfile as bf
import numpy as np
import random
import csv
import pickle
import statsmodels.api as sm
from datetime import datetime
import scipy.stats as stats
from skimage.transform import resize
from glob import glob
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset

from torchvision import transforms

# sys.path.append('.')
# from utils.data_handler import *
# from utils.modules import *

def find_all_files(folder, suffix='npz'):
    # refer to https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and os.path.join(folder, f).endswith(suffix)]
    return files

def find_all_files_(folder, str_pattern='*.npz'):
    files = [os.path.basename(y) for x in os.walk(folder) for y in glob(os.path.join(x[0], str_pattern))]
    return files

def get_all_pids(data_dir):
    pids = []
    dict_pid_fid = {}
    all_files = find_all_files(data_dir) 
    for i,f in enumerate(all_files):
        raw_data = np.load(os.path.join(data_dir, f))
        pid = raw_data['pid'].item()
        pid = pid[:pid.find('_')]
        if pid not in dict_pid_fid:
            dict_pid_fid[pid] = [i]
        else:
            dict_pid_fid[pid].append(i)
        pids.append(pid)
    # pids=list(set(pids))
    # pids.sort()
    pids = list(dict_pid_fid.keys())
    return pids, dict_pid_fid

def get_all_pids_filter(data_dir, keep_list=None):
    race_mapping = {'Asian':0, 
                'Black or African American':1, 
                'White or Caucasian':2}

    pids = []
    dict_pid_fid = {}
    files = []
    all_files = find_all_files(data_dir) 
    for i,f in enumerate(all_files):
        raw_data = np.load(os.path.join(data_dir, f))
        # race = int(raw_data['race'].item())
        race = raw_data['race'].item()
        if keep_list is not None and race not in keep_list:
            continue

        if not hasattr(raw_data, 'pid'):
            pid = f[f.find('_')+1:f.find('.')]
        else:
            pid = raw_data['pid'].item()
            pid = pid[:pid.find('_')]
        if pid not in dict_pid_fid:
            dict_pid_fid[pid] = [i]
        else:
            dict_pid_fid[pid].append(i)
        pids.append(pid)
        files.append(f)
    # pids=list(set(pids))
    # pids.sort()
    pids = list(dict_pid_fid.keys())
    return pids, dict_pid_fid, files

def vf_to_matrix(vec, fill_in=-50):
    mat = np.empty((8,9))
    mat[:] = fill_in # np.nan

    mat[0, 3:7] = vec[0:4]
    mat[1, 2:8] = vec[4:10]
    mat[2, 1:] = vec[10:18]
    mat[3, :7] = vec[18:25]
    mat[3, 8] = vec[25]
    mat[4, :7] = vec[26:33]
    mat[4, 8] = vec[33]
    mat[5, 1:] = vec[34:42]
    mat[6, 2:8] = vec[42:48]
    mat[7, 3:7] = vec[48:52]

    # mat = np.rot90(mat, k=1).copy()

    return mat


class Harvard_Glaucoma_Fairness_RandomSplit(Dataset):
    # subset: train | val | test | unmatch
    def __init__(self, data_path='./data/', split_file='', subset='train', modality_type='rnflt', task='md', resolution=224, need_shift=True, stretch=1.0,
                    depth=1, indices=None, attribute_type='race', transform=None, needBalance=False, dataset_proportion=1.):

        self.data_path = data_path
        self.modality_type = modality_type
        self.subset = subset
        self.task = task
        self.attribute_type = attribute_type
        self.transform = transform
        self.needBalance = needBalance
        self.dataset_proportion = dataset_proportion

        data_files = find_all_files(self.data_path, suffix='npz')
        random.shuffle(data_files)
        if self.subset == 'train':
            self.data_files = data_files[:int(len(data_files)*0.6)]
        elif self.subset == 'test':
            self.data_files = data_files[int(len(data_files)*0.7):]

        if indices is not None:
            self.data_files = [self.data_files[i] for i in indices]
        tmp_data_files = []
        if self.attribute_type == 'gender' or self.attribute_type == 'maritalstatus' or self.attribute_type == 'hispanic' or self.attribute_type == 'language':
            for x in self.data_files:
                rnflt_file = os.path.join(self.data_path, x)
                raw_data = np.load(rnflt_file, allow_pickle=True)
                if self.attribute_type == 'gender':
                    attr = raw_data['male'].item()
                else:
                    attr = raw_data[self.attribute_type].item()
                if attr > -1:
                    tmp_data_files.append(x)
            self.data_files = tmp_data_files

        # Oversampling
        self.balance_factor = 1.
        self.label_samples = dict()
        self.class_samples_num = None
        self.balanced_max = 0
        if self.subset == 'train' and self.needBalance:
            for idx in range(0, len(self.data_files)):
                rnflt_file = os.path.join(self.data_path, self.data_files[idx])
                raw_data = np.load(rnflt_file, allow_pickle=True)
                cur_label = raw_data[self.attribute_type].item() # self.race_mapping[raw_data['race'].item()]
                if cur_label not in self.label_samples:
                    self.label_samples[cur_label] = list()
                self.label_samples[cur_label].append(self.data_files[idx])
                self.balanced_max = len(self.label_samples[cur_label]) \
                    if len(self.label_samples[cur_label]) > self.balanced_max else self.balanced_max
            ttl_num_samples = 0
            self.class_samples_num = [0]*len(list(self.label_samples.keys()))
            for i, (k,v) in enumerate(self.label_samples.items()):
                self.class_samples_num[int(k)] = len(v)
                ttl_num_samples += len(v)
                print(f'{k}-th identity training samples: {len(v)}')
            print(f'total number of training samples: {ttl_num_samples}')
            self.class_samples_num = np.array(self.class_samples_num)

            # Oversample the classes with fewer elements than the max
            for i_label in self.label_samples:
                while len(self.label_samples[i_label]) < self.balanced_max*self.balance_factor:
                    self.label_samples[i_label].append(random.choice(self.label_samples[i_label]))

            data_files = []
            for i, (k,v) in enumerate(self.label_samples.items()):
                data_files = data_files + v
            self.data_files = data_files
        
        if self.subset == 'train' and self.dataset_proportion < 1.:
            num_samples = int(len(self.data_files) * self.dataset_proportion)
            self.data_files = random.sample(self.data_files, num_samples)
        
        # -1=unknown
        # 1="American Indian or Alaska Native", 
        # 2="Asian", 
        # 3="Black or African American", 
        # 4="Hispanic or Latino", 
        # 5="Native Hawaiian or Other Pacific Islander", 
        # 6="Other", 
        # 7="White or Caucasian”
        # self.race_mapping = {2: 0, 3: 1, 7: 2}
        self.race_mapping = {'Asian':0, 
                'Black or African American':1, 
                'White or Caucasian':2}
        
        min_vals = []
        max_vals = []
        pos_count = 0
        min_ilm_vals = []
        max_ilm_vals = []
        for x in self.data_files:
            rnflt_file = os.path.join(self.data_path, x)
            raw_data = np.load(rnflt_file, allow_pickle=True)
            min_vals.append(raw_data['md'].astype(np.float32).item())
            max_vals.append(raw_data['md'].astype(np.float32).item())
        print(f'min: {min(min_vals):.4f}, max: {max(max_vals):.4f}')
        
        self.normalize_vf = 30.0

        self.dataset_len = len(self.data_files)
        self.depth = depth
        self.size = 225
        self.resolution = resolution
        self.need_shift = need_shift
        self.stretch = stretch

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, item):

        rnflt_file = os.path.join(self.data_path, self.data_files[item])
        sample_id = self.data_files[item][:self.data_files[item].find('.')]
        raw_data = np.load(rnflt_file, allow_pickle=True)

        if self.modality_type == 'rnflt':
            rnflt_sample = raw_data[self.modality_type]
            if rnflt_sample.shape[0] != self.resolution:
                rnflt_sample = resize(rnflt_sample, (self.resolution, self.resolution))
            rnflt_sample = rnflt_sample[np.newaxis, :, :]
            if self.depth>1:
                rnflt_sample = np.repeat(rnflt_sample, self.depth, axis=0)
            data_sample = rnflt_sample.astype(np.float32)
        elif 'bscan' in self.modality_type:

            oct_img = raw_data['oct_bscans']
            oct_img_array = []
            for img in oct_img:
                oct_img_array.append(resize(img, (self.resolution, self.resolution)))
            data_sample = np.stack(oct_img_array, axis=0)
            data_sample = data_sample.astype(np.float32)
            # data_sample = np.stack([oct_img_array/255.]*(1), axis=0).astype(float)
            # data_sample = np.stack([oct_img_array]*(1), axis=0).astype(float)
            if self.transform:
                data_sample = self.transform(data_sample).float()

        elif self.modality_type == 'ilm':
            ilm_sample = raw_data[self.modality_type]
            ilm_sample = ilm_sample - np.min(ilm_sample)
            # ilm_sample = (ilm_sample - ilm_sample.min()) / (2000 - ilm_sample.min())
            if ilm_sample.shape[0] != self.resolution:
                ilm_sample = resize(ilm_sample, (self.resolution, self.resolution))
            # ilm_sample = (ilm_sample - np.min(ilm_sample)) / 2000
            # ilm_sample = np.log(1 + ilm_sample - np.min(ilm_sample))
            # ilm_sample = np.exp(ilm_sample - np.min(ilm_sample))-1
            ilm_sample = ilm_sample[np.newaxis, :, :]
            if self.depth>1:
                ilm_sample = np.repeat(ilm_sample, self.depth, axis=0)
            data_sample = ilm_sample.astype(np.float32)
        elif self.modality_type == 'rnflt+ilm':
            rnflt_sample = raw_data['rnflt']
            # rnflt_sample = np.clip(rnflt_sample, -2, 350)
            # rnflt_sample = rnflt_sample - 2
            if rnflt_sample.shape[0] != self.resolution:
                rnflt_sample = resize(rnflt_sample, (self.resolution, self.resolution))
            rnflt_sample = rnflt_sample[np.newaxis, :, :]
            if self.depth>1:
                rnflt_sample = np.repeat(rnflt_sample, self.depth, axis=0)
            # rnflt_sample = np.repeat(rnflt_sample, 3, axis=0)
            
            ilm_sample = raw_data['ilm']
            ilm_sample = ilm_sample - np.min(ilm_sample)
            if ilm_sample.shape[0] != self.resolution:
                ilm_sample = resize(ilm_sample, (self.resolution, self.resolution))
            ilm_sample = ilm_sample[np.newaxis, :, :]
            if self.depth>1:
                ilm_sample = np.repeat(ilm_sample, self.depth, axis=0)
            # ilm_sample = np.repeat(ilm_sample, 3, axis=0)

            data_sample = np.concatenate((rnflt_sample, ilm_sample), axis=0)
            data_sample = data_sample.astype(np.float32)
        elif self.modality_type == 'clockhours':
            data_sample = raw_data[self.modality_type].astype(np.float32)

        if self.task == 'md':
            y = torch.tensor(float(raw_data['md'].item()))
        elif self.task == 'tds':
            # y = raw_data['tds'].astype(np.float32)
            y = torch.tensor(float(raw_data['glaucoma'].item()))
        elif self.task == 'cls':
            # y = torch.tensor(float(raw_data['progression'][self.progression_index])).float()
            y = torch.tensor(float(raw_data['glaucoma'].item()))


        # attr = 0
        # if self.attribute_type == 'race':
        #     attr = raw_data['race'].item()
        #     # if cur_race in self.race_mapping:
        #     #     attr = self.race_mapping[cur_race]
        #     attr = torch.tensor(attr).int()
        # elif self.attribute_type == 'gender':
        #     attr = torch.tensor(raw_data['male'].item()).int()
        # elif self.attribute_type == 'maritalstatus':
        #     attr = torch.tensor(raw_data['maritalstatus'].item()).int()
        # elif self.attribute_type == 'hispanic':
        #     attr = torch.tensor(raw_data['hispanic'].item()).int()
        # elif self.attribute_type == 'language':
        #     attr = torch.tensor(raw_data['language'].item()).int()

        attr = []
        attr.append(torch.tensor(raw_data['race'].item()).int())
        attr.append(torch.tensor(raw_data['male'].item()).int())
        attr.append(torch.tensor(raw_data['hispanic'].item()).int())
        # attr.append(torch.tensor(raw_data['maritalstatus'].item()).int())
        # attr.append(torch.tensor(raw_data['language'].item()).int())
        

        return data_sample, y, attr # , datadir

class Harvard_Glaucoma_Fairness_withSplit(Dataset):
    # subset: train | val | test | unmatch
    def __init__(self, data_path='./data/', split_file='', subset='train', modality_type='rnflt', task='md', resolution=224, need_shift=True, stretch=1.0, depth=1, indices=None, attribute_type='race', transform=None, needBalance=False, dataset_proportion=1.,
                dataset='others', progression_type='MD', aug_method='m1'):

        self.data_path = os.path.join(data_path, 'scans_new_temp')
        if dataset == 'grape':
            self.data_path = os.path.join(data_path, 'grape/npzs')
            
        self.modality_type = modality_type
        self.subset = subset
        self.task = task
        self.attribute_type = attribute_type
        self.transform = transform
        self.needBalance = needBalance
        self.dataset_proportion = dataset_proportion
        self.dataset = dataset
        self.progression_type = progression_type
        self.aug_method = aug_method
        
        if dataset == 'grape':
            split_df = pd.read_csv(os.path.join(data_path, 'grape/datasplits', split_file), sep=',')
            self.data_files = []
            for i, row in split_df.iterrows():
                if row['use'] == subset:
                    self.data_files.append(row['fname'])
        else:
            split_df = pd.read_csv(os.path.join(data_path, 'datasplits_new/2021_to_2023', split_file), sep=',')
            self.data_files = []
            for i, row in split_df.iterrows():
                if row['use'] == subset and row['year'] >= 2020:
                    self.data_files.append(row['fname'])

        tmp_data_files = []
        if self.attribute_type == 'gender' or self.attribute_type == 'maritalstatus' or \
            self.attribute_type == 'hispanic' or self.attribute_type == 'language':
            for x in self.data_files:
                rnflt_file = os.path.join(self.data_path, x)
                raw_data = np.load(rnflt_file, allow_pickle=True)
                if self.attribute_type == 'gender':
                    attr = raw_data['male'].item()
                else:
                    attr = raw_data[self.attribute_type].item()
                if attr > -1:
                    tmp_data_files.append(x)
            self.data_files = tmp_data_files

        if self.modality_type == 'oct_bscans' or self.modality_type == 'oct_bscans_3d':
            tmp_data_files = []
            for x in self.data_files:
                rnflt_file = os.path.join(self.data_path, x)
                raw_data = np.load(rnflt_file, allow_pickle=True)
                if len(raw_data['oct_bscans']) > 0:
                    tmp_data_files.append(x)
            self.data_files = tmp_data_files
        elif self.modality_type == 'slo_fundus':
            tmp_data_files = []
            for x in self.data_files:
                rnflt_file = os.path.join(self.data_path, x)
                raw_data = np.load(rnflt_file, allow_pickle=True)
                if len(raw_data['slo_fundus']) > 0:
                    tmp_data_files.append(x)
            self.data_files = tmp_data_files
        elif self.modality_type == 'color_fundus':
            tmp_data_files = []
            for x in self.data_files:
                rnflt_file = os.path.join(self.data_path, x)
                raw_data = np.load(rnflt_file, allow_pickle=True)
                if len(raw_data['color_fundus']) > 0:
                    tmp_data_files.append(x)
            self.data_files = tmp_data_files

        # Downsampling | Oversampling
        self.balance_factor = 1.
        self.label_samples = dict()
        self.class_samples_num = None
        self.balanced_max = 0
        attrpaths = {}
        if self.subset == 'train' and self.needBalance:
            all_attrs = []
            for idx in range(0, len(self.data_files)):
                rnflt_file = os.path.join(self.data_path, self.data_files[idx])
                raw_data = np.load(rnflt_file, allow_pickle=True)
                
                if self.attribute_type == 'race':
                    cur_label = raw_data['race'].item()
                elif self.attribute_type == 'gender':
                    cur_label = raw_data['male'].item()
                elif self.attribute_type == 'hispanic':
                    cur_label = raw_data['hispanic'].item()
                # cur_label = raw_data[self.attribute_type].item() # self.race_mapping[raw_data['race'].item()]
                
                if cur_label not in attrpaths.keys():
                    attrpaths[cur_label] = [self.data_files[idx]]
                else:
                    attrpaths[cur_label].append(self.data_files[idx])


                all_attrs.append(cur_label)

            _, nums_attrs = np.unique(all_attrs, return_counts=True)
            print(f'number w.r.t. each attribute')
            print(nums_attrs)

            data_files = []
            balanced_max = -1
            for k in attrpaths.keys():
                if balanced_max < len(attrpaths[k]):
                    balanced_max = len(attrpaths[k])
            for k in attrpaths.keys():
                data_files.extend(attrpaths[k])
                if len(attrpaths[k]) < balanced_max:
                    sample_num = balanced_max - len(attrpaths[k])
                    #print(len(list(attrpaths[k])))
                    data_files.extend(np.random.choice(list(attrpaths[k]), sample_num))
            
            # self.balanced_min = np.min(nums_attrs)

            # index_shuf = list(range(len(self.data_files)))
            # random.shuffle(index_shuf)

            # data_files = []
            # for i in range(len(nums_attrs)):
            #     for j in range(len(index_shuf)):
            #         if all_attrs[index_shuf[j]] == i:
            #             data_files.append(self.data_files[index_shuf[j]])

            #        if len(data_files) == self.balanced_min*(i+1):
            #             break
            # print(f'total number of training samples: {len(data_files)}')
             
            #if cur_label not in self.label_samples:
            #    self.label_samples[cur_label] = list()
            #    self.label_samples[cur_label].append(self.data_files[idx])
            #    self.balanced_max = len(self.label_samples[cur_label]) \
            #     #     if len(self.label_samples[cur_label]) > self.balanced_max else self.balanced_max
            #ttl_num_samples = 0
            #self.class_samples_num = [0]*len(list(self.label_samples.keys()))
            #for i, (k,v) in enumerate(self.label_samples.items()):
            #    self.class_samples_num[int(k)] = len(v)
            #    ttl_num_samples += len(v)
            #    print(f'{k}-th identity training samples: {len(v)}')
            #print(f'total number of training samples: {ttl_num_samples}')
            #self.class_samples_num = np.array(self.class_samples_num)

            # Oversample the classes with fewer elements than the max
            #for i_label in self.label_samples:
            #    while len(self.label_samples[i_label]) < self.balanced_max*self.balance_factor:
            #        self.label_samples[i_label].append(random.choice(self.label_samples[i_label]))

            #data_files = []
            #for i, (k,v) in enumerate(self.label_samples.items()):
            #    data_files = data_files + v
            self.data_files = data_files
        
            print(f'total number of training samples: {len(data_files)}') 
     

        if self.subset == 'train' and self.dataset_proportion < 1.:
            num_samples = int(len(self.data_files) * self.dataset_proportion)
            self.data_files = random.sample(self.data_files, num_samples)
        
        # -1=unknown
        # 1="American Indian or Alaska Native", 
        # 2="A:sian", 
        # 3="Black or African American", 
        # 4="Hispanic or Latino", 
        # 5="Native Hawaiian or Other Pacific Islander", 
        # 6="Other", 
        # 7="White or Caucasian”
        # self.race_mapping = {2: 0, 3: 1, 7: 2}
        self.race_mapping = {'Asian':0, 
                'Black or African American':1, 
                'White or Caucasian':2}
        
        min_vals = []
        max_vals = []
        pos_count = 0
        min_ilm_vals = []
        max_ilm_vals = []
        for x in self.data_files:
            rnflt_file = os.path.join(self.data_path, x)
            raw_data = np.load(rnflt_file, allow_pickle=True)
#             min_vals.append(raw_data['md'].astype(np.float32).item())
#             max_vals.append(raw_data['md'].astype(np.float32).item())
#         print(f'min: {min(min_vals):.4f}, max: {max(max_vals):.4f}')
        
        self.normalize_vf = 30.0

        self.dataset_len = len(self.data_files)
        self.depth = depth
        self.size = 225
        self.resolution = resolution
        self.need_shift = need_shift
        self.stretch = stretch

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, item):

        rnflt_file = os.path.join(self.data_path, self.data_files[item])
        sample_id = self.data_files[item][:self.data_files[item].find('.')]
        raw_data = np.load(rnflt_file, allow_pickle=True)
        
        if self.modality_type == 'rnflt':
            rnflt_sample = raw_data[self.modality_type]
            if rnflt_sample.shape[0] != self.resolution:
                rnflt_sample = resize(rnflt_sample, (self.resolution, self.resolution))
            rnflt_sample = rnflt_sample[np.newaxis, :, :]
            if self.depth>1:
                rnflt_sample = np.repeat(rnflt_sample, self.depth, axis=0)
            data_sample = rnflt_sample.astype(np.float32)
            if self.transform:
#                 if self.aug_method == 'm1':
                data_sample = torch.Tensor(data_sample)
                data_sample = self.transform(data_sample).float()
#                 elif self.aug_method == 'm2':
#                     data_sample = torch.Tensor(data_sample)
#                     data_sample = self.transform(data_sample).float()
        elif self.modality_type == 'slo_fundus':
            slo_fundus = raw_data['slo_fundus']
            if slo_fundus.shape[0] != self.resolution:
                slo_fundus = resize(slo_fundus, (self.resolution, self.resolution))
            slo_fundus = slo_fundus[np.newaxis, :, :]
            if self.depth>1:
                slo_fundus = np.repeat(slo_fundus, self.depth, axis=0)
            data_sample = slo_fundus.astype(np.float32)
        elif self.modality_type == 'color_fundus':
            slo_fundus = raw_data['color_fundus']
            if slo_fundus.shape[0] != self.resolution:
                slo_fundus = resize(slo_fundus, (self.resolution, self.resolution))
            slo_fundus = np.transpose(slo_fundus, (2, 0, 1))
#             if self.depth>1:
#                 slo_fundus = np.repeat(slo_fundus, self.depth, axis=0)
            data_sample = slo_fundus.astype(np.float32)
            if self.transform:
#                 if self.aug_method == 'm1':
#                     print(2222222222222)
                data_sample = torch.Tensor(data_sample)
                data_sample = self.transform(data_sample).float()
#                 elif self.aug_method == 'm2':
#                     data_sample = torch.Tensor(data_sample)
#                     data_sample = self.transform(data_sample).float()
        elif self.modality_type == 'oct_bscans':

            oct_img = raw_data['oct_bscans']
            # data_sample = oct_img
            # print(len(oct_img.shape))
            if oct_img.shape[1] != self.resolution:
                oct_img_array = []
                for img in oct_img:
                    oct_img_array.append(resize(img, (self.resolution, self.resolution)))
                data_sample = np.stack(oct_img_array, axis=0)
            else:
                data_sample = oct_img
            data_sample = np.swapaxes(data_sample, 0, 1)
            # data_sample = data_sample[None, :, :, :]
            data_sample = data_sample.astype(np.float32)
            # data_sample = np.stack([oct_img_array/255.]*(1), axis=0).astype(float)
            # data_sample = np.stack([oct_img_array]*(1), axis=0).astype(float)
            if self.transform:
                data_sample = self.transform(data_sample).float()

        elif self.modality_type == 'oct_bscans_3d':
            oct_img = raw_data['oct_bscans']
            data_sample = oct_img
            # print(len(oct_img.shape))
            # if oct_img.shape[1] != self.resolution:
            #     oct_img_array = []
            #     for img in oct_img:
            #         oct_img_array.append(resize(img, (self.resolution, self.resolution)))
            #     data_sample = np.stack(oct_img_array, axis=0)
            # else:
            #     data_sample = oct_img
            # data_sample = np.swapaxes(data_sample, 0, 1)
            data_sample = data_sample[None, :, :, :]
            data_sample = data_sample.astype(np.float32)
            if self.transform:
                data_sample = self.transform(data_sample).float()

        elif self.modality_type == 'ilm':
            ilm_sample = raw_data[self.modality_type]
            ilm_sample = ilm_sample - np.min(ilm_sample)
            # ilm_sample = (ilm_sample - ilm_sample.min()) / (2000 - ilm_sample.min())
            if ilm_sample.shape[0] != self.resolution:
                ilm_sample = resize(ilm_sample, (self.resolution, self.resolution))
            # ilm_sample = (ilm_sample - np.min(ilm_sample)) / 2000
            # ilm_sample = np.log(1 + ilm_sample - np.min(ilm_sample))
            # ilm_sample = np.exp(ilm_sample - np.min(ilm_sample))-1
            ilm_sample = ilm_sample[np.newaxis, :, :]
            if self.depth>1:
                ilm_sample = np.repeat(ilm_sample, self.depth, axis=0)
            data_sample = ilm_sample.astype(np.float32)
        elif self.modality_type == 'rnflt+ilm':
            rnflt_sample = raw_data['rnflt']
            # rnflt_sample = np.clip(rnflt_sample, -2, 350)
            # rnflt_sample = rnflt_sample - 2
            if rnflt_sample.shape[0] != self.resolution:
                rnflt_sample = resize(rnflt_sample, (self.resolution, self.resolution))
            rnflt_sample = rnflt_sample[np.newaxis, :, :]
            if self.depth>1:
                rnflt_sample = np.repeat(rnflt_sample, self.depth, axis=0)
            # rnflt_sample = np.repeat(rnflt_sample, 3, axis=0)
            
            ilm_sample = raw_data['ilm']
            ilm_sample = ilm_sample - np.min(ilm_sample)
            if ilm_sample.shape[0] != self.resolution:
                ilm_sample = resize(ilm_sample, (self.resolution, self.resolution))
            ilm_sample = ilm_sample[np.newaxis, :, :]
            if self.depth>1:
                ilm_sample = np.repeat(ilm_sample, self.depth, axis=0)
            # ilm_sample = np.repeat(ilm_sample, 3, axis=0)

            data_sample = np.concatenate((rnflt_sample, ilm_sample), axis=0)
            data_sample = data_sample.astype(np.float32)
        elif self.modality_type == 'clockhours':
            data_sample = raw_data[self.modality_type].astype(np.float32)
        
        if self.dataset == 'grape':
            if self.progression_type == 'MD':
                y = torch.tensor(float(raw_data['progression_md'].item()))
            elif self.progression_type == 'PLR2':
                y = torch.tensor(float(raw_data['progression_plr2'].item()))
            elif self.progression_type == 'PLR3':
                y = torch.tensor(float(raw_data['progression_plr3'].item()))
        else:
            if self.task == 'md':
                y = torch.tensor(float(raw_data['md'].item()))
            elif self.task == 'tds':
                # y = raw_data['tds'].astype(np.float32)
                y = torch.tensor(float(raw_data['glaucoma'].item()))
            elif self.task == 'cls':
                # y = torch.tensor(float(raw_data['progression'][self.progression_index])).float()
                y = torch.tensor(float(raw_data['glaucoma'].item()))

        attr = []
        
        if self.dataset == 'grape':
#             attr.append(torch.tensor(random.sample([0, 1, 2], 1)[0]).int())
            attr.append(torch.tensor(int(raw_data['male'].item())).int())
#             attr.append(torch.tensor(random.sample([0, 1], 1)[0]).int())
        else:
            attr.append(torch.tensor(int(raw_data['race'].item())).int())
            attr.append(torch.tensor(int(raw_data['male'].item())).int())
            attr.append(torch.tensor(int(raw_data['hispanic'].item())).int())
        # attr.append(torch.tensor(raw_data['maritalstatus'].item()).int())
        # attr.append(torch.tensor(raw_data['language'].item()).int())
        

        return data_sample, y, attr # , datadir

def load_data_(
    *, data_dir, batch_size, image_size, class_cond=False, deterministic=False, isHAVO=0, subset='train'
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not isHAVO:
        if not data_dir:
            raise ValueError("unspecified data directory")
        all_files = _list_image_files_recursively(data_dir)
        classes = None
        if class_cond:
            # Assume classes are the first part of the filename,
            # before an underscore.
            class_names = [bf.basename(path).split("_")[0] for path in all_files]
            sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
            classes = [sorted_classes[x] for x in class_names]
        dataset = ImageDataset(
            image_size,
            all_files,
            classes=classes,
            shard=MPI.COMM_WORLD.Get_rank(),
            num_shards=MPI.COMM_WORLD.Get_size(),
        )
    elif isHAVO == 1:
        dataset = HAVO(data_dir, subset=subset, resolution=image_size)
    elif isHAVO == 2:
        dataset = HAVO_RNFLT(data_dir, subset=subset, resolution=image_size)

    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader

def load_data(
    *, data_dir, batch_size, image_size, class_cond=False, deterministic=False
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    all_files = _list_image_files_recursively(data_dir)
    classes = None
    if class_cond:
        # Assume classes are the first part of the filename,
        # before an underscore.
        class_names = [bf.basename(path).split("_")[0] for path in all_files]
        sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
        classes = [sorted_classes[x] for x in class_names]
    dataset = ImageDataset(
        image_size,
        all_files,
        classes=classes,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(self, resolution, image_paths, classes=None, shard=0, num_shards=1):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()

        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
        while min(*pil_image.size) >= 2 * self.resolution:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        scale = self.resolution / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )

        arr = np.array(pil_image.convert("RGB"))
        crop_y = (arr.shape[0] - self.resolution) // 2
        crop_x = (arr.shape[1] - self.resolution) // 2
        arr = arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]
        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)

        return np.transpose(arr, [2, 0, 1]), out_dict
