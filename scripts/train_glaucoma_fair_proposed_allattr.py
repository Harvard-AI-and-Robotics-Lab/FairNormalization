# https://github.com/developer0hye/PyTorch-ImageNet
import os
import argparse
import random
import time
import json
import numpy as np
import copy

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import *
from torch.optim import *
import torch.nn.functional as F

from sklearn.metrics import *
from sklearn.model_selection import KFold

import sys
sys.path.append('.')

from src.modules import *
from src.data_handler import *
from src import logger
from src.class_balanced_loss import *
from typing import NamedTuple

from fairlearn.metrics import *


class Imbalanced_Info(NamedTuple):
    beta: float = 0.9999
    gamma: float = 2.0
    samples_per_attr: list[int] = None
    loss_type: str = "sigmoid"
    no_of_classes: int = 2
    no_of_attr: int = 3

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')

parser.add_argument('--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')

parser.add_argument('--wd', '--weight-decay', default=6e-5, type=float,
                    metavar='W', help='weight decay (default: 6e-5)',
                    dest='weight_decay')

parser.add_argument('--seed', default=9694, type=int,
                    help='seed for initializing training. ')

parser.add_argument('--start-epoch', default=0, type=int)

parser.add_argument('--pretrained-weights', default='', type=str)

# parser.add_argument('--model-architecture', default='whitenet', type=str)

parser.add_argument('--result_dir', default='./results', type=str)
parser.add_argument('--dataset', default='others', type=str)
parser.add_argument('--progression_type', default='MD', type=str)
parser.add_argument('--data_dir', default='./results', type=str)
parser.add_argument('--model_type', default='./results', type=str)
parser.add_argument('--task', default='./results', type=str)
parser.add_argument('--image_size', default=224, type=int)
parser.add_argument('--loss_type', default='bce', type=str)
parser.add_argument('--progression_outcome', default='', type=str)
parser.add_argument('--modality_types', default='rnflt', type=str, help='rnflt|bscans')
parser.add_argument('--fuse_coef', default=1.0, type=float)
parser.add_argument('--imbalance_beta', default=-1, type=float, help='default: 0.9999, if beta<0, then roll back to conventional loss')
parser.add_argument('--split_seed', default=-1, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--perf_file', default='', type=str)
parser.add_argument('--time_window', default=-1, type=int)
parser.add_argument('--normalization_type', default='fin', type=str, help='fin|bn|lbn')
parser.add_argument('--fin_mu', default=0.001, type=float)
parser.add_argument('--fin_sigma', default=0.1, type=float)
parser.add_argument('--fin_momentum', default=0.1, type=float)
parser.add_argument('--fin_sigma_coef', nargs='+', type=float)
parser.add_argument('--attribute_type', default='race', type=str, help='race|gender')
parser.add_argument('--subset_name', default='test', type=str)
parser.add_argument("--need_balance", type=str2bool, nargs='?',
                        const=True, default=False,
                        help="Oversampling or not")
parser.add_argument('--dataset_proportion', default=1., type=float)
parser.add_argument('--split_file', default='', type=str)

                    
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    np.random.seed(seed)
    random.seed(seed)

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

@torch.no_grad()
def traverse_samples(model, train_dataset_loader, imbalanced_info=None):
    global device

    # model.fc2.register_forward_hook(get_activation('ag_norm'))

    preds = []
    attrs = []
    feats = []
    feats_by_attr = [ [] for _ in range(imbalanced_info.no_of_attr) ]
    for i, (input, target, attr) in enumerate(train_dataset_loader):
        input = input.to(device)
        target = target.to(device)
        attr = attr.to(device)

        # pred = model(input).squeeze(1)
        logit, feat = forward_model_with_fin(model, input, attr)
        feat = feat.detach().clone()

        for idx in range(feat.shape[0]):
            feats_by_attr[attr[idx]].append((feat[idx,:].unsqueeze(0)))

    mus = []
    sigmas = []
    for idx in range(len(feats_by_attr)):
        feats_by_attr[idx] = torch.cat(feats_by_attr[idx], dim=0)
        mus.append(torch.mean(feats_by_attr[idx], dim=0))
        sigmas.append(torch.std(feats_by_attr[idx], dim=0))

    return mus, sigmas


def train(model, criterion, optimizer, scaler, train_dataset_loader, epoch, total_iteration, imbalanced_info=None, time_window=-1, attribute_index=0):
    global device

    model.train()
    
    loss_batch = []
    top1_accuracy_batch = []
    top5_accuracy_batch = []
    
    preds = []
    gts = []
    attrs = []
    datadirs = []

    # beta = 0.9999
    # gamma = 2.0
    # samples_per_cls = [2,3,1,2,2]
    # loss_type = "sigmoid"

    preds_by_attr = [ [] for _ in range(imbalanced_info.no_of_attr) ]
    gts_by_attr = [ [] for _ in range(imbalanced_info.no_of_attr) ]
    t1 = time.time()
    for i, (input, target, attr) in enumerate(train_dataset_loader):
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            input = input.to(device)
            target = target.to(device)
            # attr = attr.to(device)
            selected_attr = attr[attribute_index].to(device)

            # pred = model(input).squeeze(1)
            pred, feat = forward_model_with_fin_(model, input, selected_attr)
            pred = pred.squeeze(1)

            if imbalanced_info.beta <= 0.:
                loss = criterion(pred, target)
            else:
                loss_weights = compute_rescaled_weight(imbalanced_info.samples_per_attr[attribute_index], len(imbalanced_info.samples_per_attr[attribute_index]), imbalanced_info.beta)
                loss = F.binary_cross_entropy_with_logits(pred, target, weight=loss_weights.type_as(pred))
            # loss = CB_loss_(target, pred, 
            #         imbalanced_info.samples_per_attr[attr], imbalanced_info.no_of_attr, imbalanced_info.no_of_classes, 
            #         imbalanced_info.loss_type, imbalanced_info.beta, imbalanced_info.gamma)
            
            pred_prob = torch.sigmoid(pred.detach())
            # pred_prob = F.softmax(pred.detach(), dim=1)
            preds.append(pred_prob.detach().cpu().numpy())
            gts.append(target.detach().cpu().numpy())
            attr = torch.vstack(attr)
            attrs.append(attr.detach().cpu().numpy())
            # datadirs = datadirs + datadir

            # for j, x in enumerate(attr.detach().cpu().numpy()):
            #     preds_by_attr[x].append(pred_prob[j])
            #     gts_by_attr[x].append(target[j].item())

        loss_batch.append(loss.item())
        
        top1_accuracy = accuracy(pred.detach().cpu().numpy(), target.detach().cpu().numpy(), topk=(1,))
        # top1_accuracy, top5_accuracy = accuracy(pred, target)
        
        top1_accuracy_batch.append(top1_accuracy)
        # top5_accuracy_batch.append(top5_accuracy)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if time_window > 0 and (i % time_window == 0):
            # mus, sigmas = traverse_samples(model, train_dataset_loader, imbalanced_info=imb_info)
            # model[1].update_mus_sigmas(mus, sigmas)
            logger.log(f'step {i} - {model[1]}')

    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0)
    attrs = np.concatenate(attrs, axis=1).astype(int)
    cur_auc = auc_score(preds, gts)
    # acc = np.mean(top1_accuracy_batch)
    acc = accuracy(preds, gts, topk=(1,))

    # pred_labels = np.argmax(preds, axis=1)
    # pred_labels = (preds >= 0.5).astype(float)
    # dpd = demographic_parity_difference(gts,
    #                             pred_labels,
    #                             sensitive_features=attrs)
    # dpr = demographic_parity_ratio(gts,
    #                             pred_labels,
    #                             sensitive_features=attrs)
    # eod = equalized_odds_difference(gts,
    #                             pred_labels,
    #                             sensitive_features=attrs)
    # eor = equalized_odds_ratio(gts,
    #                             pred_labels,
    #                             sensitive_features=attrs)

    # datadirs = np.array(datadirs)

    torch.cuda.synchronize()
    t2 = time.time()

    print(f"train ====> epcoh {epoch} loss: {np.mean(loss_batch):.4f} auc: {cur_auc:.4f} time: {t2 - t1:.4f}")

    # preds_by_attr_tmp = []
    # gts_by_attr_tmp = []
    # aucs_by_attr = []
    # for one_attr in np.unique(attrs).astype(int):
    #     preds_by_attr_tmp.append(preds[attrs == one_attr])
    #     gts_by_attr_tmp.append(gts[attrs == one_attr])
    #     aucs_by_attr.append(auc_score(preds[attrs == one_attr], gts[attrs == one_attr]))
    #     print(f'{one_attr}-attr auc: {aucs_by_attr[-1]:.4f}')

    t1 = time.time()

    # print("epoch: ", epoch,
    #          " loss: ", np.mean(loss_batch),
    #          " top1 acc: ", np.mean(top1_accuracy_batch),
    #          " top5 acc: ", np.mean(top5_accuracy_batch),
    #          " time per 50 iter(sec): ", t2 - t1)

    #     if i % 50 == 0:

    #         torch.cuda.synchronize()
    #         t2 = time.time()
            
    #         print("epoch: ", epoch,
    #          "iteration: ", i, "/", total_iteration,
    #          " loss: ", np.mean(loss_batch),
    #          " top1 acc: ", np.mean(top1_accuracy_batch),
    #          " top5 acc: ", np.mean(top5_accuracy_batch),
    #          " time per 50 iter(sec): ", t2 - t1)

    #         t1 = time.time()
    # [preds_by_attr_tmp, gts_by_attr_tmp, aucs_by_attr], [acc, dpd, dpr, eod, eor]
    return np.mean(loss_batch), acc, cur_auc, preds, gts, attrs
    

def validation(model, criterion, optimizer, validation_dataset_loader, epoch, result_dir=None, imbalanced_info=None, attribute_index=0):
    global device

    model.eval()
    
    loss_batch = []
    top1_accuracy_batch = []
    top5_accuracy_batch = []

    preds = []
    gts = []
    attrs = []
    datadirs = []

    preds_by_attr = [ [] for _ in range(imbalanced_info.no_of_attr) ]
    gts_by_attr = [ [] for _ in range(imbalanced_info.no_of_attr) ]

    with torch.no_grad():
        for i, (input, target, attr) in enumerate(validation_dataset_loader):
            input = input.to(device)
            target = target.to(device)
            # attr = attr.to(device)
            selected_attr = attr[attribute_index].to(device)
            
            # pred = model(input).squeeze(1)
            pred, feat = forward_model_with_fin_(model, input, selected_attr)
            pred = pred.squeeze(1)

            if imbalanced_info.beta <= 0.:
                loss = criterion(pred, target)
            else:
                loss_weights = compute_rescaled_weight(imbalanced_info.samples_per_attr[attribute_index], len(imbalanced_info.samples_per_attr[attribute_index]), imbalanced_info.beta)
                loss = F.binary_cross_entropy_with_logits(pred, target, weight=loss_weights.type_as(pred))
            # loss = CB_loss_(target, pred, 
            #         imbalanced_info.samples_per_attr, imbalanced_info.no_of_attr, imbalanced_info.no_of_classes, 
            #         imbalanced_info.loss_type, imbalanced_info.beta, imbalanced_info.gamma)

            pred_prob = torch.sigmoid(pred.detach())
            # pred_prob = F.softmax(pred.detach(), dim=1)
            preds.append(pred_prob.detach().cpu().numpy())
            gts.append(target.detach().cpu().numpy())
            attr = torch.vstack(attr)
            attrs.append(attr.detach().cpu().numpy())
            # datadirs = datadirs + datadir

            # for j, x in enumerate(attr.detach().cpu().numpy()):
            #     preds_by_attr[x].append(pred_prob[j])
            #     gts_by_attr[x].append(target[j].item())
            

            loss_batch.append(loss.item())

            # pred_prob = torch.sigmoid(pred.detach())
            # preds.append(pred_prob.detach().cpu().numpy())
            # gts.append(target.detach().cpu().numpy())
            
            top1_accuracy = accuracy(pred.cpu().numpy(), target.cpu().numpy(), topk=(1,)) 
        
            top1_accuracy_batch.append(top1_accuracy)
            # top5_accuracy_batch.append(top5_accuracy)
        
    loss = np.mean(loss_batch)
    # top5_accuracy = np.mean(top5_accuracy_batch)

    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0)
    attrs = np.concatenate(attrs, axis=1).astype(int)
    cur_auc = auc_score(preds, gts)
    # acc = np.mean(top1_accuracy_batch)
    acc = accuracy(preds, gts, topk=(1,))

    # pred_labels = np.argmax(preds, axis=1)
    # pred_labels = (preds >= 0.5).astype(float)
    # dpd = demographic_parity_difference(gts,
    #                             pred_labels,
    #                             sensitive_features=attrs)
    # dpr = demographic_parity_ratio(gts,
    #                             pred_labels,
    #                             sensitive_features=attrs)
    # eod = equalized_odds_difference(gts,
    #                             pred_labels,
    #                             sensitive_features=attrs)
    # eor = equalized_odds_ratio(gts,
    #                             pred_labels,
    #                             sensitive_features=attrs)

    # datadirs = np.array(datadirs)

    print(f"test <==== epcoh {epoch} loss: {np.mean(loss_batch):.4f} auc: {cur_auc:.4f}")

    # preds_by_attr_tmp = []
    # gts_by_attr_tmp = []
    # aucs_by_attr = []
    # for one_attr in np.unique(attrs).astype(int):
    #     preds_by_attr_tmp.append(preds[attrs == one_attr])
    #     gts_by_attr_tmp.append(gts[attrs == one_attr])
    #     aucs_by_attr.append(auc_score(preds[attrs == one_attr], gts[attrs == one_attr]))
    #     print(f'{one_attr}-attr auc: {aucs_by_attr[-1]:.4f}')

    # print("val-", "epoch: ", epoch, " loss: ", loss, " top1 acc: ", top1_accuracy, " top5 acc: ", top5_accuracy)
    # [preds_by_attr_tmp, gts_by_attr_tmp, aucs_by_attr], [acc, dpd, dpr, eod, eor]    
    return loss, acc, cur_auc, preds, gts, attrs


if __name__ == '__main__':
    args = parser.parse_args()

    if args.seed < 0:
        args.seed = int(np.random.randint(10000, size=1)[0])
    set_random_seed(args.seed)

    if args.split_seed < 0:
        args.split_seed = int(np.random.randint(10000, size=1)[0])

    logger.log(f'===> random seed: {args.seed}')

    # if not os.path.exists(args.result_dir):
    #     os.makedirs(args.result_dir)
    logger.configure(dir=args.result_dir, log_suffix='train')

    with open(os.path.join(args.result_dir, f'args_train.txt'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # n_splits = 3
    # kfold = KFold(n_splits=n_splits, shuffle=True, random_state=args.split_seed)

    # train_dataset = dataset.ImageClassificationDataset(dataset_path='D:/datasets/ILSVRC2012_ImageNet/ILSVRC2012_img_train', phase="train")
    # validation_dataset = dataset.ImageClassificationDataset(dataset_path='D:/datasets/ILSVRC2012_ImageNet/ILSVRC2012_img_val', phase="validation")
    # havo_dataset = HAVO_Longitudinal_Race(args.data_dir, progression_type=args.progression_outcome, modality_type=args.modality_types, task=args.task, resolution=args.image_size)
    # val_dataset = HAVO_Longitudinal_New(args.data_dir, subset=args.data_val_subset, split_file=args.split_file, progression_type=args.progression_outcome, modality_type=args.modality_types, task=args.task, resolution=args.image_size, stretch=args.stretch_ratio_vf, depth=args.input_depth)
    
    # pids, dict_pid_fid, _ = get_all_pids_filter(args.data_dir, list(havo_dataset.race_mapping.keys()))
    # pids = np.array(pids)

    # all_preds, all_gts = [None]*n_splits, [None]*n_splits
    # all_attrs = [None]*n_splits
    # all_datadirs = [None]*n_splits
    # for fold, (train_pids, test_pids) in enumerate(kfold.split(pids)):
    #     print(f'============= start fold {fold} =============')
    #     train_ids = []
    #     for x in train_pids:
    #         train_ids = train_ids + dict_pid_fid[pids[x]]
    #     test_ids = []
    #     for x in test_pids:
    #         test_ids = test_ids + dict_pid_fid[pids[x]]

    if args.model_type == 'vit' or args.model_type == 'swin':
        args.image_size = 224

#     attribute_mapping = {'race':0, 'gender':1, 'hispanic':2}
    attribute_mapping = {'gender':0}


    trn_havo_dataset = Harvard_Glaucoma_Fairness_withSplit(args.data_dir, subset='train', split_file=args.split_file, modality_type=args.modality_types, task=args.task, resolution=args.image_size, attribute_type=args.attribute_type, needBalance=args.need_balance, dataset_proportion=args.dataset_proportion, dataset=args.dataset, progression_type=args.progression_type)
    tst_havo_dataset = Harvard_Glaucoma_Fairness_withSplit(args.data_dir, subset='test', split_file=args.split_file, modality_type=args.modality_types, task=args.task, resolution=args.image_size, attribute_type=args.attribute_type, dataset=args.dataset, progression_type=args.progression_type)

    # print(f'trn patients {len(train_pids)} with {len(train_ids)} samples, val patients {len(test_pids)} with {len(test_ids)} samples')
    logger.log(f'trn patients {len(trn_havo_dataset)} with {len(trn_havo_dataset)} samples, val patients {len(tst_havo_dataset)} with {len(tst_havo_dataset)} samples')
    # train_ids = fids[train_pids]
    # test_ids = fids[test_pids]
    # train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    # test_subsampler = torch.utils.data.SequentialSampler(test_ids)

    train_dataset_loader = torch.utils.data.DataLoader(
        trn_havo_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)
    
    validation_dataset_loader = torch.utils.data.DataLoader(
        tst_havo_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    samples_per_attr = get_num_by_group_(train_dataset_loader)
    # print(f'group information:')
    logger.log(f'group information:')
    # print(samples_per_attr)
    logger.log(samples_per_attr)
    imb_info = Imbalanced_Info(beta=args.imbalance_beta, no_of_attr=len(samples_per_attr), samples_per_attr=samples_per_attr)
    # , samples_per_attr=np.array(samples_per_attr), no_of_attr=len(samples_per_attr), no_of_classes=2

    best_global_perf_file = os.path.join(os.path.dirname(args.result_dir), f'best_{args.perf_file}')
    lastep_global_perf_file = os.path.join(os.path.dirname(args.result_dir), f'last_{args.perf_file}')

    acc_head_str = ''
    auc_head_str = ''
    dpd_head_str = ''
    eod_head_str = ''
    esacc_head_str = ''
    esauc_head_str = ''
    group_disparity_head_str = ''

    if args.perf_file != '':
        if not os.path.exists(best_global_perf_file):
            for i in range(len(samples_per_attr)):
                # acc_head_str += ', '.join([f'acc_attr{i}' for x in range(len(samples_per_attr[i]))]) + ', '
                auc_head_str += ', '.join([f'auc_attr{i}_group{x}' for x in range(len(samples_per_attr[i]))]) + ', '
            dpd_head_str += ', '.join([f'dpd_attr{i}' for i in range(len(samples_per_attr))]) + ', '
            eod_head_str += ', '.join([f'eod_attr{i}' for i in range(len(samples_per_attr))]) + ', '
            esacc_head_str += ', '.join([f'esacc_attr{i}' for i in range(len(samples_per_attr))]) + ', '
            esauc_head_str += ', '.join([f'esauc_attr{i}' for i in range(len(samples_per_attr))]) + ', '

            group_disparity_head_str += ', '.join([f'std_group_disparity_attr{i}, max_group_disparity_attr{i}' for i in range(len(samples_per_attr))]) + ', '
            
            with open(best_global_perf_file, 'w') as f:
                f.write(f'epoch, acc, {esacc_head_str} auc, {esauc_head_str} {auc_head_str} {dpd_head_str} {eod_head_str} {group_disparity_head_str} path\n')
        # if not os.path.exists(lastep_global_perf_file):
        #     acc_head_str = ', '.join([f'acc_class{x}' for x in range(len(samples_per_attr))])
        #     auc_head_str = ', '.join([f'auc_class{x}' for x in range(len(samples_per_attr))])
        #     with open(lastep_global_perf_file, 'w') as f:
        #         f.write(f'epoch, es_acc, acc, {acc_head_str}, es_auc, auc, {auc_head_str}, dpd, dpr, eod, eor, path\n')

    # model_dict = {'whitenet': whitenet.WhiteNet(),
    #               'tiny': tiny.YOLOv3TinyBackbone()}
    # model = model_dict[args.model_architecture]
    # model = model.to(device)

    if args.task == 'md':
        out_dim = 1
        # predictor_head = General_Logistic(trn_dataset.min_vf_val/trn_dataset.max_vf_val*args.stretch_ratio_vf, args.stretch_ratio_vf)
        criterion = nn.MSELoss()
        predictor_head = nn.Identity() # nn.Tanhshrink()
    elif args.task == 'cls': 
        # out_dim = 1 if args.modality_types == 'rnflt' else 200
        out_dim = 1
        # predictor_head = nn.Sigmoid()
        criterion = nn.BCEWithLogitsLoss()
        # criterion = nn.CrossEntropyLoss()
        predictor_head = nn.Sigmoid()
    elif args.task == 'tds': 
        out_dim = 52
        criterion = nn.MSELoss()
        # predictor_head = General_Logistic(trn_dataset.min_vf_val/trn_dataset.max_vf_val*args.stretch_ratio_vf, args.stretch_ratio_vf)
        predictor_head = nn.Identity()

    if args.model_type == 'vit':
        in_feat_to_final = 768
    elif args.model_type == 'efficientnet':
        in_feat_to_final = 1280
    elif args.model_type == 'resnet':
        in_feat_to_final = 2048
    elif args.model_type == 'vgg':
        in_feat_to_final = 4096

    if args.modality_types == 'oct_bscans_3d':
        in_feat_to_final = 512

    if args.normalization_type == 'fin':
        # ag_norm = Fair_Identity_Normalizer_(len(samples_per_attr[attribute_mapping[args.attribute_type]]), \
        #     dim=in_feat_to_final, mu=args.fin_mu, sigma=args.fin_sigma, momentum=args.fin_momentum) #  [0]*imb_info.no_of_attr, [1]*imb_info.no_of_attr
        # fin_1 = Fair_Identity_Normalizer_3D(len(samples_per_attr[attribute_mapping[args.attribute_type]]), \
        #     dims=[320,7,7], mu=args.fin_mu, sigma=args.fin_sigma, momentum=args.fin_momentum) #  [0]*imb_info.no_of_attr, [1]*imb_info.no_of_attr
        fin_2 = Fair_Identity_Normalizer_1D_(len(samples_per_attr[attribute_mapping[args.attribute_type]]), \
            dim=in_feat_to_final, mu=args.fin_mu, sigma=args.fin_sigma, momentum=args.fin_momentum) # , sigma_coefs=torch.tensor(args.fin_sigma_coef),  [0]*imb_info.no_of_attr, [1]*imb_info.no_of_attr, sigma_coefs=torch.tensor(args.fin_sigma_coef),
    elif args.normalization_type == 'lbn':
        ag_norm = Learnable_BatchNorm1d(dim=in_feat_to_final)
    elif args.normalization_type == 'bn':
        ag_norm = nn.BatchNorm1d(in_feat_to_final)

    if args.modality_types == 'ilm' or args.modality_types == 'rnflt' or args.modality_types == 'slo_fundus':
        in_dim = 1
        model = create_model(model_type=args.model_type, in_dim=in_dim, out_dim=out_dim, include_final=False)
    elif args.modality_types == 'oct_bscans':
        in_dim = 200
        model = create_model(model_type=args.model_type, in_dim=in_dim, out_dim=out_dim, include_final=False)
    elif args.modality_types == 'color_fundus':
        in_dim = 3
        model = create_model(model_type=args.model_type, in_dim=in_dim, out_dim=out_dim, include_final=False)
    elif args.modality_types == 'oct_bscans_3d':
        in_dim = 1
        model = ConvNet_3D(include_final=False)
        final_layer = nn.Linear(512, out_dim)
    elif args.modality_types == 'rnflt+ilm':
        in_dim = 2
        model = OphBackbone(model_type=args.model_type, in_dim=in_dim, coef=args.fuse_coef)
    if args.model_type == 'vit':
        final_layer = nn.Linear(in_features=768, out_features=out_dim, bias=True)
    elif args.model_type == 'efficientnet':
        final_layer = nn.Linear(in_features=1280, out_features=out_dim, bias=False)
    elif args.model_type == 'resnet':
        final_layer = nn.Linear(in_features=2048, out_features=out_dim, bias=True)
    elif args.model_type == 'vgg':
        final_layer = nn.Linear(in_features=4096, out_features=out_dim, bias=True)

    if args.modality_types == 'oct_bscans_3d':
        final_layer = nn.Linear(in_features=512, out_features=out_dim)

    # for name, module in model.named_modules():
    #     print(name)
    # for i, layer in enumerate(model.children()):
    #     print(layer)

    model = nn.Sequential(model, fin_2, final_layer)

    # part2 = nn.Sequential(copy.deepcopy(model.features[8]), copy.deepcopy(model.avgpool),\
    #                 nn.Flatten(), copy.deepcopy(model.classifier))
    # part1 = copy.deepcopy(model.features)
    # part1[8] = nn.Identity()
    # model = nn.Sequential(part1, fin_1, part2, fin_2, final_layer)

    model = model.to(device)

    scaler = torch.cuda.amp.GradScaler()

    # criterion = nn.CrossEntropyLoss()

    # optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                             momentum=args.momentum,
    #                             nesterov=True,
    #                             weight_decay=args.weight_decay)
    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.0, 0.1), weight_decay=args.weight_decay)
    
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    
    # print("len of train_dataset: ", len(trn_havo_dataset))
    # print("len of validation_dataset: ", len(tst_havo_dataset))
    
    start_epoch = 0
    best_top1_accuracy = 0.

    if args.pretrained_weights != "":
        checkpoint = torch.load(args.pretrained_weights)

        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # top1_accuracy = checkpoint['top1_accuracy']
        # best_top1_accuracy = checkpoint['best_top1_accuracy']
    
    # print("#parameters of model: ", utils.count_total_prameters(model))
    
    total_iteration = len(trn_havo_dataset)//args.batch_size

    best_auc_groups = None
    best_acc_groups = None
    best_pred_gt_by_attr = None
    best_auc = sys.float_info.min
    best_acc = sys.float_info.min
    best_es_acc = sys.float_info.min
    best_es_auc = sys.float_info.min
    best_ep = 0
    best_between_group_disparity = None

    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc, train_auc, trn_preds, trn_gts, trn_attrs = train(model, criterion, optimizer, scaler, train_dataset_loader, epoch, total_iteration, imbalanced_info=imb_info, time_window=args.time_window, attribute_index=attribute_mapping[args.attribute_type])
        test_loss, test_acc, test_auc, tst_preds, tst_gts, tst_attrs = validation(model, criterion, optimizer, validation_dataset_loader, epoch, imbalanced_info=imb_info, attribute_index=attribute_mapping[args.attribute_type])
        scheduler.step()

        # trn_pred_gt_by_attrs, trn_other_metrics
        # tst_pred_gt_by_attrs, tst_other_metrics
        # val_es_acc, val_es_auc, val_aucs_by_attrs, val_dpds, val_eods = evalute_perf_by_attr(tst_preds, tst_gts, tst_attrs)
        val_es_acc, val_es_auc, val_aucs_by_attrs, val_dpds, val_eods, between_group_disparity = evalute_comprehensive_perf(tst_preds, tst_gts, tst_attrs)

        # state = {
        # 'epoch': epoch,# zero indexing
        # 'fold': fold,
        # 'model_state_dict': model.state_dict(),
        # 'optimizer_state_dict' : optimizer.state_dict(),
        # 'scaler_state_dict' : scaler.state_dict(),
        # 'scheduler_state_dict' : scheduler.state_dict(),
        # 'train_auc': train_auc,
        # 'test_auc': test_auc
        # }
        # torch.save(state, os.path.join(args.result_dir, f"model_fold{fold}_ep{epoch:03d}.pth"))

        # trn_acc_groups = []
        # trn_auc_groups = []
        # for i_group in range(len(trn_pred_gt_by_attrs[0])):
        #     trn_acc_groups.append(accuracy(trn_pred_gt_by_attrs[0][i_group], trn_pred_gt_by_attrs[1][i_group], topk=(1,))) 
        #     trn_auc_groups.append(auc_score(trn_pred_gt_by_attrs[0][i_group], trn_pred_gt_by_attrs[1][i_group]))
        
        # acc_groups = []
        # auc_groups = []
        # for i_group in range(len(tst_pred_gt_by_attrs[0])):
        #     acc_groups.append(accuracy(tst_pred_gt_by_attrs[0][i_group], tst_pred_gt_by_attrs[1][i_group], topk=(1,))) 
        #     auc_groups.append(auc_score(tst_pred_gt_by_attrs[0][i_group], tst_pred_gt_by_attrs[1][i_group]))

        # es_acc = equity_scaled_accuracy(tst_preds, tst_gts, tst_attrs)
        # es_auc = equity_scaled_AUC(tst_preds, tst_gts, tst_attrs)

        if best_auc <= test_auc:
            best_auc = test_auc
            best_acc = test_acc
            best_ep = epoch
            # all_preds[fold] = tst_preds
            # all_gts[fold] = tst_gts
            # all_attrs[fold] = tst_attrs
            # all_datadirs[fold] = tst_datadirs
            # best_pred_gt_by_attr = tst_pred_gt_by_attrs
            # best_tst_other_metrics = tst_other_metrics
            # best_acc_groups = acc_groups
            best_auc_groups = val_aucs_by_attrs
            best_dpd_groups = val_dpds
            best_eod_groups = val_eods
            best_es_acc = val_es_acc
            best_es_auc = val_es_auc
            best_between_group_disparity = between_group_disparity

            state = {
            'epoch': epoch,# zero indexing
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict(),
            'scaler_state_dict' : scaler.state_dict(),
            'scheduler_state_dict' : scheduler.state_dict(),
            'train_auc': train_auc,
            'test_auc': test_auc
            }
            torch.save(state, os.path.join(args.result_dir, f"model_ep{epoch:03d}.pth"))

        # print(f'---- best AUC {best_auc:.4f} at epoch {best_ep}')
        logger.log(f'---- best AUC {best_auc:.4f} at epoch {best_ep}')
        logger.log(f'---- best AUC by groups and attributes at epoch {best_ep}')
        logger.log(best_auc_groups)
        # for i_attr in range(len(best_pred_gt_by_attr[-1])):
        #     print(f'---- best AUC at {i_attr}-attr {best_pred_gt_by_attr[-1][i_attr]:.4f} at epoch {best_ep}')
        #     logger.log(f'---- best AUC at {i_attr}-attr {best_pred_gt_by_attr[-1][i_attr]:.4f} at epoch {best_ep}')
    
        if args.result_dir is not None:
            np.savez(os.path.join(args.result_dir, f'pred_gt_ep{epoch:03d}.npz'), 
                        val_pred=tst_preds, val_gt=tst_gts, val_attr=tst_attrs)


        logger.logkv('epoch', epoch)
        logger.logkv('trn_loss', round(train_loss,4))
        logger.logkv('trn_acc', round(train_acc,4))
        logger.logkv('trn_auc', round(train_auc,4))
        # logger.logkv('trn_acc', round(trn_other_metrics[0],4))
        # logger.logkv('trn_dpd', round(trn_other_metrics[1],4))
        # logger.logkv('trn_dpr', round(trn_other_metrics[2],4))
        # logger.logkv('trn_eod', round(trn_other_metrics[3],4))
        # logger.logkv('trn_eor', round(trn_other_metrics[4],4))
        # for i_group in range(len(trn_acc_groups)):
        #     logger.logkv(f'trn_acc_class{i_group}', round(trn_acc_groups[i_group],4))
        # for i_group in range(len(trn_auc_groups)):
        #     logger.logkv(f'trn_auc_class{i_group}', round(trn_auc_groups[i_group],4))

        logger.logkv('val_loss', round(test_loss,4))
        logger.logkv('val_acc', round(test_acc,4))
        logger.logkv('val_auc', round(test_auc,4))
        # logger.logkv('val_es_acc', round(es_acc,4))
        # logger.logkv('val_es_auc', round(es_auc,4))

        # logger.logkv('val_dpd', round(tst_other_metrics[1],4))
        # logger.logkv('val_dpr', round(tst_other_metrics[2],4))
        # logger.logkv('val_eod', round(tst_other_metrics[3],4))
        # logger.logkv('val_eor', round(tst_other_metrics[4],4))
        # for i_group in range(len(acc_groups)):
        #     logger.logkv(f'val_acc_class{i_group}', round(acc_groups[i_group],4))
        # for i_group in range(len(auc_groups)):
        #     logger.logkv(f'val_auc_class{i_group}', round(auc_groups[i_group],4))

        for ii in range(len(val_es_acc)):
            logger.logkv(f'val_es_acc_attr{ii}', round(val_es_acc[ii],4))
        for ii in range(len(val_es_auc)):
            logger.logkv(f'val_es_auc_attr{ii}', round(val_es_auc[ii],4))
        for ii in range(len(val_aucs_by_attrs)):
            for iii in range(len(val_aucs_by_attrs[ii])):
                logger.logkv(f'val_auc_attr{ii}_group{iii}', round(val_aucs_by_attrs[ii][iii],4))

        for ii in range(len(between_group_disparity)):
            logger.logkv(f'val_auc_attr{ii}_std_group_disparity', round(between_group_disparity[ii][0],4))
            logger.logkv(f'val_auc_attr{ii}_max_group_disparity', round(between_group_disparity[ii][1],4))

        for ii in range(len(val_dpds)):
            logger.logkv(f'val_dpd_attr{ii}', round(val_dpds[ii],4))
        for ii in range(len(val_eods)):
            logger.logkv(f'val_eod_attr{ii}', round(val_eods[ii],4))

        logger.dumpkvs()

        # if (epoch == args.epochs-1) and (args.perf_file != ''):
        #     if os.path.exists(lastep_global_perf_file):
        #         with open(lastep_global_perf_file, 'a') as f:
        #             acc_head_str = ', '.join([f'{x:.4f}' for x in acc_groups])
        #             auc_head_str = ', '.join([f'{x:.4f}' for x in auc_groups])
        #             path_str = f'{args.result_dir}'
        #             f.write(f'{best_ep}, {es_acc:.4f}, {best_acc:.4f}, {acc_head_str}, {es_auc:.4f}, {test_auc:.4f}, {auc_head_str}, {tst_other_metrics[1]:.4f}, {tst_other_metrics[2]:.4f}, {tst_other_metrics[3]:.4f}, {tst_other_metrics[4]:.4f}, {path_str}\n')

        # if best_top1_accuracy <= top1_accuracy:
        #     best_top1_accuracy = top1_accuracy
        #     torch.save(state, os.path.join("./", args.model_architecture+"_best.pth"))   

    # fold_preds = np.concatenate(all_preds[:fold+1], axis=0)
    # fold_gts = np.concatenate(all_gts[:fold+1], axis=0)
    # fold_attrs = np.concatenate(all_attrs[:fold+1], axis=0).astype(int)
    # # fold_datadirs = np.concatenate(all_datadirs[:fold+1], axis=0)

    # assert fold_preds.shape[0] == fold_gts.shape[0] and fold_gts.shape[0] == fold_attrs.shape[0] 
    # # and fold_attrs.shape[0] == fold_datadirs.shape[0]
    
    # if args.result_dir is not None:
    #     np.savez(os.path.join(args.result_dir, f'all_pred_gt_fold{fold}.npz'), 
    #                 val_pred=fold_preds, val_gt=fold_gts, val_attr=fold_attrs) # , val_datadir=fold_datadirs
    # cur_auc = auc_score(fold_preds, fold_gts)

    # print(f"==== fold {fold} - modality {args.modality_types} - progression {args.progression_outcome} auc: {cur_auc:.4f}")
    # logger.log(f'==== fold {fold} - modality {args.modality_types} - progression {args.progression_outcome} auc: {cur_auc:.4f}')

    # for one_attr in np.unique(fold_attrs).astype(int):
    #     cur_auc = auc_score(fold_preds[fold_attrs == one_attr], fold_gts[fold_attrs == one_attr])
    #     print(f"==== fold {fold} - modality {args.modality_types} - progression {args.progression_outcome} {one_attr}-th attr auc: {cur_auc:.4f}")
    #     logger.log(f'==== fold {fold} - modality {args.modality_types} - progression {args.progression_outcome} {one_attr}-th attr auc: {cur_auc:.4f}')

    if args.perf_file != '':
        if os.path.exists(best_global_perf_file):
            with open(best_global_perf_file, 'a') as f:
                # acc_head_str = ', '.join([f'{x:.4f}' for x in best_acc_groups])
                # auc_head_str = ', '.join([f'{x:.4f}' for x in best_auc_groups])

                esacc_head_str = ', '.join([f'{x:.4f}' for x in best_es_acc]) + ', '
                esauc_head_str = ', '.join([f'{x:.4f}' for x in best_es_auc]) + ', '

                auc_head_str = ''
                for i in range(len(best_auc_groups)):
                    auc_head_str += ', '.join([f'{x:.4f}' for x in best_auc_groups[i]]) + ', '

                group_disparity_str = ''
                for i in range(len(best_between_group_disparity)):
                    group_disparity_str += ', '.join([f'{x:.4f}' for x in best_between_group_disparity[i]]) + ', '
                
                dpd_head_str = ', '.join([f'{x:.4f}' for x in best_dpd_groups]) + ', '
                eod_head_str = ', '.join([f'{x:.4f}' for x in best_eod_groups]) + ', '

                path_str = f'{args.result_dir}_seed{args.seed}_auc{best_auc:.4f}'
                f.write(f'{best_ep}, {best_acc:.4f}, {esacc_head_str} {best_auc:.4f}, {esauc_head_str} {auc_head_str} {dpd_head_str} {eod_head_str} {group_disparity_str} {path_str}\n')
                # f.write('epoch, acc, auc, dpd, dpr, eod, eor, path\n')

    os.rename(args.result_dir, f'{args.result_dir}_seed{args.seed}_auc{best_auc:.4f}')
