import sys, os

import blobfile as bf
import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP

from torchvision.models import *
import torch.nn.functional as F

from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, ExponentialLR, StepLR

from sklearn.metrics import *
from fairlearn.metrics import *

# sys.path.append('.')
# from improved_diffusion import dist_util

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1, 5)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    if len(output.shape) == 1:
        acc = np.sum((output >= 0.5).astype(float) == target)/target.shape[0]
        return acc.item()
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.shape[0]

        _, pred = output.topk(maxk, dim=1)
        target = target.view(batch_size, 1).repeat(1, maxk)
        
        correct = (pred == target)
  
        topk_accuracy = []
        for k in topk:
            accuracy = correct[:, :k].float().sum().item() # [0, batch_size]
            accuracy /= batch_size # [0, 1.]
            topk_accuracy.append(accuracy)
        
        return topk_accuracy[0]

def compute_auc(pred_prob, y, num_classes=2):
    if torch.is_tensor(pred_prob):
        pred_prob = pred_prob.detach().cpu().numpy()
    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()

    if num_classes == 2:
        fpr, tpr, thresholds = roc_curve(y, pred_prob)
        auc_val = auc(fpr, tpr)
    elif num_classes > 2:
        y_onehot = num_to_onehot(y, num_classes)
        auc_val = roc_auc_score(y_onehot, pred_prob, average='macro', multi_class='ovr')

    return auc_val

def auc_score(pred_prob, y):
    if torch.is_tensor(pred_prob):
        pred_prob = pred_prob.detach().cpu().numpy()
    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()

    if np.unique(y).shape[0]>2:
        AUC = roc_auc_score(y, pred_prob, multi_class='ovr')
    else:
        fpr, tpr, thresholds = roc_curve(y, pred_prob)
        AUC = auc(fpr, tpr)
    
    return AUC

def num_to_onehot(nums, num_to_class):
    # nums = [1, 0, 3]
    nums = nums.astype(int)
    # n_values = np.max(nums) + 1
    n_values = num_to_class
    onehot_vec = np.eye(n_values)[nums]
    return onehot_vec

def prob_to_label(pred_prob):
    # Find the indices of the highest probabilities for each sample
    max_prob_indices = np.argmax(pred_prob, axis=1)

    # Create one-hot vectors for each sample
    one_hot_vectors = np.zeros_like(pred_prob)
    one_hot_vectors[np.arange(len(max_prob_indices)), max_prob_indices] = 1

    return one_hot_vectors

def numeric_to_one_hot(y, num_classes=None):
    y = np.asarray(y, dtype=np.int32)

    if num_classes is None:
        num_classes = np.max(y) + 1
    
    one_hot_array = np.zeros((len(y), num_classes))
    one_hot_array[np.arange(len(y)), y] = 1
    
    return one_hot_array

def multiclass_demographic_parity(pred_prob, y, attrs):

    pred_one_hot = prob_to_label(pred_prob)

    gt_one_hot = numeric_to_one_hot(y)

    scores = []
    for i in range(pred_one_hot.shape[1]):
        tmp_score = demographic_parity_difference(pred_one_hot[:,i],
                                gt_one_hot[:,i],
                                sensitive_features=attrs)

        scores.append(tmp_score)

    avg_score = np.mean(scores)
        
    return avg_score

def multiclass_equalized_odds(pred_prob, y, attrs):

    pred_one_hot = prob_to_label(pred_prob)

    gt_one_hot = numeric_to_one_hot(y)

    scores = []
    for i in range(pred_one_hot.shape[1]):
        tmp_score = equalized_odds_difference(pred_one_hot[:,i],
                            gt_one_hot[:,i],
                            sensitive_features=attrs)

        scores.append(tmp_score)

    avg_score = np.mean(scores)
        
    return avg_score

def multiclass_demographic_parity_(pred_prob, y, attrs):
    # https://github.com/wbawakate/fairtorch/blob/master/examples/demo.ipynb
    # expected value of y_pred = 1 for each sensitive group
    # E[h(X)|A=a]
    # correspond to demographic parity

    if torch.is_tensor(pred_prob):
        pred_prob = pred_prob.detach().cpu().numpy()
    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()

    attrs_set = np.unique(attrs)
    y_pred = np.argmax(pred_prob, axis=1)

    mc_dpd = 0
    for i in range(pred_prob.shape[1]):
        tmp_preds = (y_pred==i).astype(int)
        tmp_not_preds = 1 - tmp_preds

        # pred_by_attrs = []
        # for j in attrs_set:
        #     idx = attrs==j
        #     pred_by_attrs.append(tmp_preds[idx].mean().item())
        # mc_dpd += np.max(pred_by_attrs) - np.min(pred_by_attrs)

        dp_by_attrs = []
        for j in attrs_set:
            idx = attrs==j
            tmp = np.abs(tmp_preds.mean().item() - tmp_preds[idx].mean().item()) + np.abs(tmp_not_preds.mean().item() - tmp_not_preds[idx].mean().item())
            dp_by_attrs.append(tmp)
            print(tmp)
        mc_dpd += np.mean(dp_by_attrs).item()

    mc_dpd = mc_dpd / pred_prob.shape[1]
        
    return mc_dpd

def auc_score_multiclass(pred_prob, y, num_of_class=3, eps=0.01):
    if torch.is_tensor(pred_prob):
        pred_prob = pred_prob.detach().cpu().numpy()
    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()

    # specificity: 80% 85% 90% 95%
    # FPR: 20% 15% 10% 5%
    # if np.unique(y).shape[0]>2:
    #     AUC = roc_auc_score(y, pred_prob, multi_class='ovr')
    # else:
    sensitivity_at_diff_specificity = [-1]*4
    y_onehot = num_to_onehot(y, num_of_class)
    fpr, tpr, thresholds = roc_curve(y_onehot.ravel(), pred_prob.ravel())
    for i in range(len(fpr)):
        cur_fpr = fpr[i]
        cur_tpr = tpr[i]
        if np.abs(cur_fpr-0.2) <= eps:
            sensitivity_at_diff_specificity[0] = cur_tpr
        if np.abs(cur_fpr-0.15) <= eps:
            sensitivity_at_diff_specificity[1] = cur_tpr
        if np.abs(cur_fpr-0.1) <= eps:
            sensitivity_at_diff_specificity[2] = cur_tpr
        if np.abs(cur_fpr-0.05) <= eps:
            sensitivity_at_diff_specificity[3] = cur_tpr
    AUC = auc(fpr, tpr)
    
    return AUC, sensitivity_at_diff_specificity

def equity_scaled_accuracy(output, target, attrs, alpha=1.):
    es_acc = 0
    overall_acc = np.sum((output >= 0.5).astype(float) == target)/target.shape[0]
    tmp = 0
    identity_wise_perf = []
    identity_wise_num = []
    for one_attr in np.unique(attrs).astype(int):
        pred_group = output[attrs == one_attr]
        gt_group = target[attrs == one_attr]
        acc = np.sum((pred_group >= 0.5).astype(float) == gt_group)/gt_group.shape[0]

        identity_wise_perf.append(acc)
        identity_wise_num.append(gt_group.shape[0])
        # tmp += gt_group.shape[0]/target.shape[0] * np.abs(overall_acc-acc)

        # acc_std = np.std((pred_group >= 0.5).astype(float) == gt_group)
        # es_acc += gt_group.shape[0]/target.shape[0] * (acc / (alpha*np.log(1+acc_std) + 1))
        # es_acc += gt_group.shape[0]/target.shape[0] * (acc / (alpha*acc_std + 1))

    # mu = np.mean(identity_wise_perf)
    # for i in range(len(identity_wise_perf)):
    #     # tmp += identity_wise_num[i]/target.shape[0] * (identity_wise_perf[i]-mu)**2
    #     tmp += (identity_wise_perf[i]-mu)**2
    # tmp = np.sqrt(tmp/(len(identity_wise_perf)-1))
    # es_acc = (overall_acc / (alpha*tmp + 1))

    # es_acc = (overall_acc / (alpha*np.exp(tmp) + 1))

    for i in range(len(identity_wise_perf)):
        tmp += np.abs(identity_wise_perf[i]-overall_acc)
    es_acc = (overall_acc / (alpha*tmp + 1))
    
    return es_acc

def equity_scaled_AUC(output, target, attrs, alpha=1., num_classes=2):
    es_auc = 0
    tmp = 0
    identity_wise_perf = []
    identity_wise_num = []
    # fpr, tpr, thresholds = roc_curve(target, output)
    # overall_auc = auc(fpr, tpr)
    if num_classes == 2:
        fpr, tpr, thresholds = roc_curve(target, output)
        overall_auc = auc(fpr, tpr)
    elif num_classes > 2:
        y_onehot = num_to_onehot(target, num_classes)
        overall_auc = roc_auc_score(y_onehot, output, average='macro', multi_class='ovr')

    for one_attr in np.unique(attrs).astype(int):
        pred_group = output[attrs == one_attr]
        gt_group = target[attrs == one_attr]


        # fpr, tpr, thresholds = roc_curve(gt_group, pred_group)
        # group_auc = auc(fpr, tpr)
        if num_classes == 2:
            fpr, tpr, thresholds = roc_curve(gt_group, pred_group)
            group_auc = auc(fpr, tpr)
        elif num_classes > 2:
            y_onehot = num_to_onehot(gt_group, num_classes)
            group_auc = roc_auc_score(y_onehot, pred_group, average='macro', multi_class='ovr')

        # tmp += gt_group.shape[0]/target.shape[0] * np.abs(overall_auc-group_auc)
        
        identity_wise_perf.append(group_auc)
        identity_wise_num.append(gt_group.shape[0])

    # mu = np.mean(identity_wise_perf)
    # for i in range(len(identity_wise_perf)):
    #     # tmp += identity_wise_num[i]/target.shape[0] * (identity_wise_perf[i]-mu)**2
    #     tmp += (identity_wise_perf[i]-mu)**2
    # tmp = np.sqrt(tmp/(len(identity_wise_perf)-1))
    # es_auc = (overall_auc / (alpha*tmp + 1))

    for i in range(len(identity_wise_perf)):
        tmp += np.abs(identity_wise_perf[i]-overall_auc)
    es_auc = (overall_auc / (alpha*tmp + 1))

    # es_auc = (overall_auc / (alpha*np.log(1+tmp) + 1))
    # es_auc = (overall_auc / (alpha*np.exp(tmp) + 1))

    return es_auc

def evalute_perf_by_attr(preds, gts, attrs=None, num_classes=2):

    esaccs_by_attrs = []
    esaucs_by_attrs = []
    aucs_by_attrs = []
    dpds = []
    dprs = []
    eods = []
    eors = []
    for i in range(attrs.shape[0]):
        attr = attrs[i,:]

        es_acc = equity_scaled_accuracy(preds, gts, attr)
        esaccs_by_attrs.append(es_acc)
        es_auc = equity_scaled_AUC(preds, gts, attr, num_classes=num_classes)
        esaucs_by_attrs.append(es_auc)

        aucs_by_group = []
        elements = np.unique(attr).astype(int)
        for e in elements:
            aucs_by_group.append( compute_auc(preds[attr == e], gts[attr == e], num_classes=num_classes) )
        aucs_by_attrs.append(aucs_by_group)
        pred_labels = (preds >= 0.5).astype(float)
        if num_classes == 2:
            dpd = demographic_parity_difference(gts,
                                        pred_labels,
                                        sensitive_features=attr)
            dpr = demographic_parity_ratio(gts,
                                        pred_labels,
                                        sensitive_features=attr)
            eod = equalized_odds_difference(gts,
                                        pred_labels,
                                        sensitive_features=attr)
            eor = equalized_odds_ratio(gts,
                                        pred_labels,
                                        sensitive_features=attr)
        elif num_classes > 2:
            dpd = multiclass_demographic_parity(preds, gts, attr)
            dpr = 0
            eod = multiclass_equalized_odds(preds, gts, attr)
            eor = 0

        dpds.append(dpd)
        eods.append(eod)

    return esaccs_by_attrs, esaucs_by_attrs, aucs_by_attrs, dpds, eods


def evalute_comprehensive_perf(preds, gts, attrs=None, num_classes=2):

    esaccs_by_attrs = []
    esaucs_by_attrs = []
    aucs_by_attrs = []
    dpds = []
    dprs = []
    eods = []
    eors = []
    between_group_disparity = []

    overall_auc = compute_auc(preds, gts, num_classes=num_classes)

    for i in range(attrs.shape[0]):
        attr = attrs[i,:]

        es_acc = equity_scaled_accuracy(preds, gts, attr)
        esaccs_by_attrs.append(es_acc)
        es_auc = equity_scaled_AUC(preds, gts, attr, num_classes=num_classes)
        esaucs_by_attrs.append(es_auc)

        aucs_by_group = []
        elements = np.unique(attr).astype(int)
        for e in elements:
            aucs_by_group.append( compute_auc(preds[attr == e], gts[attr == e], num_classes=num_classes) )
        aucs_by_attrs.append(aucs_by_group)
        std_disparity, max_disparity = compute_between_group_disparity(aucs_by_group, overall_auc)
        between_group_disparity.append([std_disparity, max_disparity])

        pred_labels = (preds >= 0.5).astype(float)
        if num_classes == 2:
            dpd = demographic_parity_difference(gts,
                                        pred_labels,
                                        sensitive_features=attr)
            dpr = demographic_parity_ratio(gts,
                                        pred_labels,
                                        sensitive_features=attr)
            eod = equalized_odds_difference(gts,
                                        pred_labels,
                                        sensitive_features=attr)
            eor = equalized_odds_ratio(gts,
                                        pred_labels,
                                        sensitive_features=attr)
        elif num_classes > 2:
            dpd = multiclass_demographic_parity(preds, gts, attr)
            dpr = 0
            eod = multiclass_equalized_odds(preds, gts, attr)
            eor = 0

        dpds.append(dpd)
        eods.append(eod)

    return esaccs_by_attrs, esaucs_by_attrs, aucs_by_attrs, dpds, eods, between_group_disparity

def evalute_comprehensive_perf_(preds, gts, attrs=None, num_classes=2):

    esaccs_by_attrs = []
    esaucs_by_attrs = []
    aucs_by_attrs = []
    dpds = []
    dprs = []
    eods = []
    eors = []
    between_group_disparity = []

    overall_auc = compute_auc(preds, gts, num_classes=num_classes)

    for i in range(attrs.shape[0]):
        attr = attrs[i,:]

        es_acc = equity_scaled_accuracy(preds, gts, attr)
        esaccs_by_attrs.append(es_acc)
        es_auc = equity_scaled_AUC(preds, gts, attr, num_classes=num_classes)
        esaucs_by_attrs.append(es_auc)

        aucs_by_group = []
        elements = np.unique(attr).astype(int)
        for e in elements:
            aucs_by_group.append( compute_auc(preds[attr == e], gts[attr == e], num_classes=num_classes) )
        aucs_by_attrs.append(aucs_by_group)
        std_disparity, max_disparity = compute_between_group_disparity_half(aucs_by_group, overall_auc)
        between_group_disparity.append([std_disparity, max_disparity])

        pred_labels = (preds >= 0.5).astype(float)
        if num_classes == 2:
            dpd = demographic_parity_difference(gts,
                                        pred_labels,
                                        sensitive_features=attr)
            dpr = demographic_parity_ratio(gts,
                                        pred_labels,
                                        sensitive_features=attr)
            eod = equalized_odds_difference(gts,
                                        pred_labels,
                                        sensitive_features=attr)
            eor = equalized_odds_ratio(gts,
                                        pred_labels,
                                        sensitive_features=attr)
        elif num_classes > 2:
            dpd = multiclass_demographic_parity(preds, gts, attr)
            dpr = 0
            eod = multiclass_equalized_odds(preds, gts, attr)
            eor = 0

        dpds.append(dpd)
        eods.append(eod)

    return esaccs_by_attrs, esaucs_by_attrs, aucs_by_attrs, dpds, eods, between_group_disparity

def scale_losses(loss_tensor, attr=None, level='individual', fair_scaling_beta=1.):
    # level: 'individual' | 'group'
    if level == 'individual':
        loss = ( (torch.exp(loss_tensor/fair_scaling_beta) / torch.exp(loss_tensor/fair_scaling_beta).sum())*loss_tensor.shape[0]*loss_tensor ).mean()
    elif level == 'group':
        tmp_weights = torch.zeros(int(torch.max(attr).item())+1).type(loss_tensor.type())
        for x in attr:
            tmp_weights[x.long()] = torch.mean(loss_tensor[attr==x]).item()
        tmp_weights = torch.softmax(tmp_weights/fair_scaling_beta, dim=0)
        tmp_weights_inplace = tmp_weights[attr.long()]*loss_tensor.shape[0]
        loss = (tmp_weights_inplace * loss_tensor).mean()
    return loss

def compute_between_group_disparity(auc_list, overall_auc):
    return np.std(auc_list) / overall_auc, (np.max(auc_list)-np.min(auc_list)) / overall_auc

def compute_between_group_disparity_half(auc_list, overall_auc):
    return np.std(auc_list) / np.abs(overall_auc-0.5), (np.max(auc_list)-np.min(auc_list)) / np.abs(overall_auc-0.5)

def get_num_by_group(train_dataset_loader, n_group=3):
    samples_per_cls = [0]*n_group
    all_attrs = []
    for i, (input, target, attr) in enumerate(train_dataset_loader):
        attr_array = attr.detach().cpu().numpy().tolist()
        all_attrs = all_attrs + attr_array
        # for j in range(n_group):
        #     tmp = np.count_nonzero(attr_array == j)
        #     samples_per_cls[j] = samples_per_cls[j] + tmp
    all_attrs,samples_per_attr = np.unique(all_attrs, return_counts=True)

    return all_attrs, samples_per_attr

def get_num_by_group_(train_dataset_loader, n_group=3):
    samples_per_cls = [0]*n_group
    all_attrs = [ [] for _ in range(n_group)]
    for i, (input, target, attr) in enumerate(train_dataset_loader):
        for i in range(len(attr)):
            one_attr = attr[i]
            attr_array = one_attr.detach().cpu().numpy().tolist()
            all_attrs[i] = all_attrs[i] + attr_array
        # for j in range(n_group):
        #     tmp = np.count_nonzero(attr_array == j)
        #     samples_per_cls[j] = samples_per_cls[j] + tmp
    ret = []
    for i in range(len(all_attrs)):
        all_attrs_per_identity = all_attrs[i]
        _, samples_per_attr = np.unique(all_attrs_per_identity, return_counts=True)
        ret.append(samples_per_attr)

    return ret

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

class Rescaled_Softsign(nn.Module):
    def __init__(self, rescale=1.):
        super().__init__()
        self.rescale = rescale
        self.acti_func = nn.Softsign()

    def forward(self, x):
        y = self.acti_func(x)
        y = y*self.rescale
        return y

class Rescaled_Sigmoid(nn.Module):
    def __init__(self, rescale=1.):
        super().__init__()
        self.rescale = rescale
        self.acti_func = nn.Sigmoid()

    def forward(self, x):
        y = self.acti_func(x)-0.5
        y = y*self.rescale
        return y

def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)

def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}

def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def reward_function(delta_err, reward_type='exp_delta', alpha=1.0):
    cur_reward = 0
    if reward_type == 'delta':
        cur_reward = alpha*max(delta_err, 0)
    elif reward_type == 'exp_delta':
        # cur_reward = max(np.exp(err_before - err_after)-1, 0)
        cur_reward = max(np.exp(alpha*delta_err)-1, 0)
        # cur_reward = np.exp(alpha*delta_err)
        # cur_reward = np.exp(alpha*max(delta_err, 0)) - 1
    elif reward_type == 'log_delta':
        cur_reward = max(np.log(alpha*delta_err + 1), 0)
        # cur_reward = np.log(alpha*max(delta_err + 1, 0))
    elif reward_type == 'sqr_delta':
        cur_reward = max(alpha*delta_err, 0)**2
    return cur_reward

def reward_function_(y_hat, y_gt, reward_type='mae', eps=1e-6):
    cur_reward = 0
    if reward_type == 'mae':
        cur_reward = -torch.log(torch.abs(y_hat-y_gt)+eps).mean().item()
    elif reward_type == 'ce':
        # cur_reward = max(np.exp(err_before - err_after)-1, 0)
        # cur_reward = max(np.exp(alpha*delta_err)-1, 0)
        cur_reward = F.binary_cross_entropy(y_hat, y_gt).item()
        # cur_reward = np.exp(alpha*max(delta_err, 0)) - 1
    return cur_reward

class General_Logistic(nn.Module):
    def __init__(self, min_val=-38.0, max_val=26.0): # , th=-1.
        super().__init__()
        # self.min_val = min_val
        # self.max_val = max_val
        self.min_nml_val = min_val # -1.46153846154
        self.max_nml_val = max_val # 1

        # A+\frac{\left(K-A\right)}{\left(C+Q\exp\left(-Bx\right)\right)^{\frac{1}{v}}} on Desmos
        self.A = self.min_nml_val
        self.K = self.max_nml_val
        self.C = 1
        self.Q = 1
        self.B = 1

        self.nu = np.log2(self.C+self.Q*np.exp(1))/np.log2((self.K-self.A)/(-self.A))
        
    def forward(self, x):
        out = self.A + (self.K-self.A)/(self.C+self.Q*torch.exp( -self.B * x ))**(1/self.nu)
        return out

class Attribute_Grouped_Normalizer(nn.Module):
    def __init__(self, num_attr=0, dim=0, mus=None, sigmas=None, momentum=0.9):
        super().__init__()
        self.num_attr = num_attr
        self.dim=0
        self.mus = mus
        self.sigmas = sigmas
        self.eps = 1e-6
        self.momentum = momentum
        # if mus is not None:
        #     self.num_attr = len(mus)
        #     self.mus = mus
        # if sigmas is not None:
        #     self.sigmas = sigmas

    def forward(self, x, attr):
        if self.mus is None:
            self.mus = []
            for i in range(self.num_attr):
                self.mus.append(torch.zeros(x.shape[1]).type(x.type()))
        if self.sigmas is None:
            self.sigmas = []
            for i in range(self.num_attr):
                self.sigmas.append(torch.ones(x.shape[1]).type(x.type()))
        for idx in range(x.shape[0]):
            x[idx,:] = (x[idx,:] - self.mus[attr[idx]])/(self.sigmas[attr[idx]] + self.eps)
        
        return x

    # @torch.no_grad()
    def update_mus_sigmas(self, mus, sigmas):
        if self.momentum >= 0 and self.momentum < 1:
            for i in range(self.num_attr):
                self.mus[i] = self.momentum*self.mus[i] + (1-self.momentum)*mus[i]
                self.sigmas[i] = sigmas[i]

    def __repr__(self):
        if self.mus is not None and self.sigmas is not None:
            out_str = ', '.join([f'G{i}: ({torch.mean(m).item():f}, {torch.mean(s).item():f})' for i, (m, s) in enumerate(zip(self.mus, self.sigmas))])
        else:
            out_str = 'Attribute-Grouped Normalizer is not initialized yet.'
        return out_str

def forward_model_with_fin(model, data, attr):
    feat = model[0](data)
    if type(model[1]).__name__ != 'Fair_Identity_Normalizer' and type(model[1]).__name__ != 'Fair_Identity_Normalizer_':
        nml_feat = model[1](feat)
    else:
        nml_feat = model[1](feat, attr)
    logit = model[2](nml_feat)
    return logit, feat

def forward_model_with_fin_(model, data, attr):
    feat = data
    logit_feat = None
    for i, layer in enumerate(model.children()):
        if not type(layer).__name__.startswith('Fair_Identity_Normalizer'):
            feat = layer(feat)
        else:
            feat = layer(feat, attr)
        if i == len(model)-1:
            logit_feat = feat.detach().clone()
    logit = feat

    return logit, logit_feat

class Fair_Scaler(nn.Module):
    def __init__(self, beta=.9, bias=0, metric_scores=None):
        self.beta = beta
        self.bias = bias

    def forward(self, attr, metric_scores):
        metric_scores_ = metric_scores.clone().type_as(attr)
        weights = (1-self.beta) / (1-self.beta**(metric_scores_-self.bias))
        # weights = weights / torch.sum(weights) * metric_scores.shape[0]
        instance_weights = weights[attr]

        return instance_weights

class Fair_Identity_Normalizer_3D(nn.Module):
    def __init__(self, num_attr=0, dims=None, mu=0.001, sigma=0.1, momentum=0, test=False):
        super().__init__()
        self.num_attr = num_attr
        self.dims = dims

        # self.mus = nn.Parameter(torch.ones(self.num_attr, self.dim)*mu)
        # self.mus = nn.Parameter((torch.randn(self.num_attr, self.dim)-0.5)*2*mu) # mu=0.01 -> auc: 0.865
        # self.mus = nn.Parameter((torch.rand(self.num_attr, self.dim)-0.5)*2*mu)

        self.mus = nn.Parameter(torch.randn(self.num_attr, self.dims[0], self.dims[1], self.dims[2])*mu)
        self.sigmas = nn.Parameter(torch.randn(self.num_attr, self.dims[0], self.dims[1], self.dims[2])*sigma)
        # self.sigma_coefs = nn.Parameter(torch.randn(self.num_attr, self.dim)*sigma)
        if test:
            self.sigmas = nn.Parameter(torch.ones(self.num_attr, self.dims[0], self.dims[1], self.dims[2])*sigma)
        self.eps = 1e-6
        self.momentum = momentum


    def forward(self, x, attr):
        x_clone = x.clone()
        for idx in range(x.shape[0]):
            x[idx,:,:,:] = (x[idx,:,:,:] - self.mus[attr[idx],:,:,:])/( torch.log(1+torch.exp(self.sigmas[attr[idx],:,:,:])) + self.eps)
        x = (1-self.momentum)*x + self.momentum*x_clone

        return x

    # @torch.no_grad()
    # def update_mus_sigmas(self, mus, sigmas):
    #     if self.momentum >= 0 and self.momentum < 1:
    #         for i in range(self.num_attr):
    #             self.mus[i] = self.momentum*self.mus[i] + (1-self.momentum)*mus[i]
    #             self.sigmas[i] = sigmas[i]

    def __repr__(self):
        if self.mus is not None and self.sigmas is not None:
            sigma = torch.log(1+torch.exp(self.sigmas))
            sigma = torch.mean(sigma, dim=[1,2,3])
            mu = torch.mean(self.mus, dim=[1,2,3])
            out_str = ', '.join([f'G{i}: ({mu[i].item():f}, {sigma[i].item():f})' for i in range(mu.shape[0])])
        else:
            out_str = 'Attribute-Grouped Normalizer is not initialized yet.'
        return out_str

class Fair_Identity_Normalizer_Single(nn.Module):
    def __init__(self, num_attr=0, dim=0, mu=0.001, sigma=1.0, momentum=0, test=False):
        super().__init__()
        self.num_attr = num_attr
        self.dim = dim

        # self.mus = nn.Parameter(torch.ones(self.num_attr, self.dim)*mu)
        # self.mus = nn.Parameter((torch.randn(self.num_attr, self.dim)-0.5)*2*mu) # mu=0.01 -> auc: 0.865
        # self.mus = nn.Parameter((torch.rand(self.num_attr, self.dim)-0.5)*2*mu)

        self.mus = nn.Parameter(torch.randn(self.num_attr)*mu)
        self.sigmas = nn.Parameter(torch.ones(self.num_attr)*sigma)
        # self.sigma_coefs = nn.Parameter(torch.randn(self.num_attr, self.dim)*sigma)
        if test:
            self.sigmas = nn.Parameter(torch.ones(self.num_attr)*sigma)
        self.eps = 1e-6
        self.momentum = momentum


    def forward(self, x, attr):
        x_clone = x.clone()
        for idx in range(x.shape[0]):
            x[idx,:] = (x[idx,:] - self.mus[attr[idx]])/( torch.log(1+torch.exp(self.sigmas[attr[idx]])) + self.eps)
        x = (1-self.momentum)*x + self.momentum*x_clone

        return x

    # @torch.no_grad()
    # def update_mus_sigmas(self, mus, sigmas):
    #     if self.momentum >= 0 and self.momentum < 1:
    #         for i in range(self.num_attr):
    #             self.mus[i] = self.momentum*self.mus[i] + (1-self.momentum)*mus[i]
    #             self.sigmas[i] = sigmas[i]

    def __repr__(self):
        if self.mus is not None and self.sigmas is not None:
            sigma = torch.log(1+torch.exp(self.sigmas))
            # sigma = torch.mean(sigma, dim=1)
            # mu = torch.mean(self.mus, dim=1)
            mu = self.mus
            out_str = ', '.join([f'G{i}: ({mu[i].item():f}, {sigma[i].item():f})' for i in range(mu.shape[0])])
        else:
            out_str = 'Attribute-Grouped Normalizer is not initialized yet.'
        return out_str

class Fair_Identity_Normalizer(nn.Module):
    def __init__(self, num_attr=0, dim=0, mu=0.001, sigma=0.1, momentum=0, test=False):
        super().__init__()
        self.num_attr = num_attr
        self.dim = dim

        # self.mus = nn.Parameter(torch.ones(self.num_attr, self.dim)*mu)
        # self.mus = nn.Parameter((torch.randn(self.num_attr, self.dim)-0.5)*2*mu) # mu=0.01 -> auc: 0.865
        # self.mus = nn.Parameter((torch.rand(self.num_attr, self.dim)-0.5)*2*mu)

        self.mus = nn.Parameter(torch.randn(self.num_attr, self.dim)*mu)
        self.sigmas = nn.Parameter(torch.randn(self.num_attr, self.dim)*sigma)
        # self.sigma_coefs = nn.Parameter(torch.randn(self.num_attr, self.dim)*sigma)
        if test:
            self.sigmas = nn.Parameter(torch.ones(self.num_attr, self.dim)*sigma)
        self.eps = 1e-6
        self.momentum = momentum


    def forward(self, x, attr):
        x_clone = x.clone()
        for idx in range(x.shape[0]):
            x[idx,:] = (x[idx,:] - self.mus[attr[idx], :])/( torch.log(1+torch.exp(self.sigmas[attr[idx], :])) + self.eps)
        x = (1-self.momentum)*x + self.momentum*x_clone

        return x

    # @torch.no_grad()
    # def update_mus_sigmas(self, mus, sigmas):
    #     if self.momentum >= 0 and self.momentum < 1:
    #         for i in range(self.num_attr):
    #             self.mus[i] = self.momentum*self.mus[i] + (1-self.momentum)*mus[i]
    #             self.sigmas[i] = sigmas[i]

    def __repr__(self):
        if self.mus is not None and self.sigmas is not None:
            sigma = torch.log(1+torch.exp(self.sigmas))
            sigma = torch.mean(sigma, dim=1)
            mu = torch.mean(self.mus, dim=1)
            out_str = ', '.join([f'G{i}: ({mu[i].item():f}, {sigma[i].item():f})' for i in range(mu.shape[0])])
        else:
            out_str = 'Attribute-Grouped Normalizer is not initialized yet.'
        return out_str

class Fair_Identity_Normalizer_1D(nn.Module):
    def __init__(self, num_attr=0, dim=0, mu=0.001, sigma=0.1, sigma_coefs=None, momentum=0, test=False):
        super().__init__()
        self.num_attr = num_attr
        self.dim = dim

        # self.mus = nn.Parameter(torch.ones(self.num_attr, self.dim)*mu)
        # self.mus = nn.Parameter((torch.randn(self.num_attr, self.dim)-0.5)*2*mu) # mu=0.01 -> auc: 0.865
        # self.mus = nn.Parameter((torch.rand(self.num_attr, self.dim)-0.5)*2*mu)

        self.mus = nn.Parameter(torch.randn(self.num_attr, self.dim)*mu)
        self.sigmas = nn.Parameter(torch.randn(self.num_attr, self.dim)*sigma)
        # self.sigma_coefs = nn.Parameter(sigma_coefs, requires_grad=False)
        if test:
            self.sigmas = nn.Parameter(torch.ones(self.num_attr, self.dim)*sigma)
        self.eps = 1e-6
        self.momentum = momentum


    def forward(self, x, attr):
        x_clone = x.clone()
        for idx in range(x.shape[0]):
            # x[idx,:] = self.sigma_coefs[attr[idx]]*(x[idx,:] - self.mus[attr[idx], :])/( torch.log(1 + torch.exp(self.sigmas[attr[idx], :])) + self.eps)
            x[idx,:] = (x[idx,:] - self.mus[attr[idx], :])/( torch.log(1 + torch.exp(self.sigmas[attr[idx], :])) + self.eps)
        x = (1-self.momentum)*x + self.momentum*x_clone
        # self.sigma_coefs[attr[idx]]*

        return x

    # @torch.no_grad()
    # def update_mus_sigmas(self, mus, sigmas):
    #     if self.momentum >= 0 and self.momentum < 1:
    #         for i in range(self.num_attr):
    #             self.mus[i] = self.momentum*self.mus[i] + (1-self.momentum)*mus[i]
    #             self.sigmas[i] = sigmas[i]

    def __repr__(self):
        if self.mus is not None and self.sigmas is not None:
            sigma = torch.log(1+torch.exp(self.sigmas))
            sigma = torch.mean(sigma, dim=1)
            mu = torch.mean(self.mus, dim=1)
            out_str = ', '.join([f'G{i}: ({mu[i].item():f}, {sigma[i].item():f})' for i in range(mu.shape[0])])
        else:
            out_str = 'Attribute-Grouped Normalizer is not initialized yet.'
        return out_str

class Fair_Identity_Normalizer_1D_(nn.Module):
    def __init__(self, num_attr=0, dim=0, mu=0.001, sigma=0.1, sigma_coefs=[1.,1.,1.], momentum=0, test=False):
        super().__init__()
        self.num_attr = num_attr
        self.dim = dim

        # self.mus = nn.Parameter(torch.ones(self.num_attr, self.dim)*mu)
        # self.mus = nn.Parameter((torch.randn(self.num_attr, self.dim)-0.5)*2*mu) # mu=0.01 -> auc: 0.865
        # self.mus = nn.Parameter((torch.rand(self.num_attr, self.dim)-0.5)*2*mu)

        self.mus = nn.Parameter(torch.randn(self.num_attr, self.dim)*mu) # , requires_grad=False
        self.sigmas = nn.Parameter(torch.randn(self.num_attr, self.dim)*sigma)
        # self.sigma_coefs = sigma_coefs
        # self.sigma_coefs = nn.Parameter(sigma_coefs, requires_grad=False)
        if test:
            self.sigmas = nn.Parameter(torch.ones(self.num_attr, self.dim)*sigma)
        self.eps = 1e-6
        self.momentum = momentum


    def forward(self, x, attr):
        x_clone = x.clone()
        for idx in range(x.shape[0]):
            # x[idx,:] = self.sigma_coefs[attr[idx]]*(x[idx,:] - self.mus[attr[idx], :])/( torch.log(1 + torch.exp(self.sigmas[attr[idx], :])) + self.eps)
            x[idx,:] = (x[idx,:] - self.mus[attr[idx], :])/( torch.log(1 + torch.exp(self.sigmas[attr[idx], :])) + self.eps)
            # x[idx,:] = (x[idx,:] - self.mus[attr[idx], :])/( self.sigma_coefs[attr[idx]] * torch.log(1 + torch.exp(self.sigmas[attr[idx], :])) + self.eps)
            # x[idx,:] = x[idx,:] / ( torch.log(1 + torch.exp(self.sigmas[attr[idx], :])) + self.eps)
            
        x = (1-self.momentum)*x + self.momentum*x_clone
        # self.sigma_coefs[attr[idx]]*

        return x

    # @torch.no_grad()
    # def update_mus_sigmas(self, mus, sigmas):
    #     if self.momentum >= 0 and self.momentum < 1:
    #         for i in range(self.num_attr):
    #             self.mus[i] = self.momentum*self.mus[i] + (1-self.momentum)*mus[i]
    #             self.sigmas[i] = sigmas[i]

    def __repr__(self):
        if self.mus is not None and self.sigmas is not None:
            sigma = torch.log(1+torch.exp(self.sigmas))
            sigma = torch.mean(sigma, dim=1)
            mu = torch.mean(self.mus, dim=1)
            out_str = ', '.join([f'G{i}: ({mu[i].item():f}, {sigma[i].item():f})' for i in range(mu.shape[0])])
        else:
            out_str = 'Attribute-Grouped Normalizer is not initialized yet.'
        return out_str

class Learnable_BatchNorm1d(nn.Module):
    def __init__(self, dim=0, mu=0, sigma=0.1, momentum=0.9):
        super().__init__()
        self.dim = dim
        self.mus = nn.Parameter(torch.ones(1, self.dim)*mu)
        self.sigmas = nn.Parameter(torch.ones(1, self.dim)*sigma)
        self.eps = 1e-6
        self.momentum = momentum

    def forward(self, x):
        for idx in range(x.shape[0]):
            x = (x - self.mus)/( torch.log(1+torch.exp(self.sigmas)) + self.eps)
        
        return x

    # @torch.no_grad()
    # def update_mus_sigmas(self, mus, sigmas):
    #     if self.momentum >= 0 and self.momentum < 1:
    #         for i in range(self.num_attr):
    #             self.mus[i] = self.momentum*self.mus[i] + (1-self.momentum)*mus[i]
    #             self.sigmas[i] = sigmas[i]

    def __repr__(self):
        if self.mus is not None and self.sigmas is not None:
            sigma = torch.log(1+torch.exp(self.sigmas))
            sigma = torch.mean(sigma)
            mu = torch.mean(self.mus)
            out_str = f'Learnable BatchNorm: ({mu:f}, {sigma:f})'
        else:
            out_str = 'Attribute-Grouped Normalizer is not initialized yet.'
        return out_str

class MD_Mapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_features = 52
        self.out_features = 1
        
        weight = [0.010346387,0.010663622,0.010032727,0.007129987,0.014017274,0.018062957,0.018842243,0.016647837,0.015124109,0.011389459,0.014017678,0.022160035,0.02378898,0.02383174,0.02191793,0.019983033,0.0159671,0.0115242,0.007463015,0.018917531,0.023298082,0.02881147,0.027520778,0.025385285,0.023138773,0.016495131,0.008567998,0.017318338,0.028689633,0.02881154,0.028483851,0.025037148,0.023584995,0.016130119,0.015494349,0.024661184,0.028129123,0.028682529,0.026372951,0.024033034,-0.001105303,0.016997128,0.01889403,0.023627078,0.024890497,0.023402898,0.0218989,0.017713769,0.015848428,0.018916324,0.018597527,0.019021584]
        bias = 0.000592563
        self.weight = torch.nn.Parameter(torch.tensor(weight), requires_grad=False)
        self.bias = torch.nn.Parameter(torch.tensor(bias), requires_grad=False)

    def forward(self, input):
        assert input.shape[1] == self.in_features
        output = input @ self.weight.t() + self.bias
        return output

class LogisticRegression(torch.nn.Module):
     def __init__(self, input_dim, output_dim):
         super(LogisticRegression, self).__init__()
         self.linear = torch.nn.Linear(input_dim, output_dim, bias=True)
     def forward(self, x):
         outputs = torch.sigmoid(self.linear(x))
         return outputs

class ConvNet_3D(nn.Module):
    def __init__(self, width=200, height=200, depth=200, out_dim=1, include_final=True):
        super().__init__()

        self.include_final = include_final
        
        self.conv1 = nn.Conv3d(1, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=2)  
        self.bn1 = nn.BatchNorm3d(64)
        
        self.conv2 = nn.Conv3d(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=2)
        self.bn2 = nn.BatchNorm3d(64)
        
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool3d(kernel_size=2)
        self.bn3 = nn.BatchNorm3d(128)
        
        self.conv4 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool3d(kernel_size=2)
        self.bn4 = nn.BatchNorm3d(256)
        
        self.gap = nn.AdaptiveAvgPool3d(1)
        
        self.fc1 = nn.Linear(256, 512)
        self.drop = nn.Dropout(0.3)
        
        if self.include_final:
            self.fc2 = nn.Linear(512, out_dim)
        else:
            self.fc2 = nn.Identity()
        
        # self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = F.relu(self.bn1(self.pool1(self.conv1(x))))
        x = F.relu(self.bn2(self.pool2(self.conv2(x))))
        x = F.relu(self.bn3(self.pool3(self.conv3(x))))
        x = F.relu(self.bn4(self.pool4(self.conv4(x))))
        
        x = torch.squeeze( self.gap(x), (2, 3, 4) )
        
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        
        x = self.fc2(x)
        # x = self.sigmoid(x)
        
        return x

def create_model(model_type='efficientnet', in_dim=1, out_dim=1, use_pretrained=True, include_final=True):
    vf_predictor = None
    if model_type == 'vit':
        vf_predictor = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        vf_predictor.conv_proj = nn.Conv2d(in_dim, 768, kernel_size=(16, 16), stride=(16, 16))
        if include_final:
            vf_predictor.heads[0] = nn.Linear(in_features=768, out_features=out_dim, bias=True)
        else:
            vf_predictor.heads[0] = nn.Identity()
        # vf_predictor = vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_V1)
        # vf_predictor.conv_proj = nn.Conv2d(1, 1024, kernel_size=(16, 16), stride=(16, 16))
        # vf_predictor.heads.head = nn.Linear(in_features=1024, out_features=out_dim, bias=True)
    elif model_type == 'efficientnet':
        load_weights = None
        if use_pretrained:
            load_weights = EfficientNet_B1_Weights.IMAGENET1K_V2
        vf_predictor = efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V2)
        if in_dim != 3:
            vf_predictor.features[0][0] = nn.Conv2d(in_dim, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        if include_final:
            vf_predictor.classifier[1] = nn.Linear(in_features=1280, out_features=out_dim, bias=False)
        else:
            vf_predictor.classifier[1] = nn.Identity()
        # vf_predictor = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1)
        # vf_predictor.classifier[1] = nn.Linear(in_features=1280, out_features=256, bias=True)
        # vf_predictor = nn.Sequential(vf_predictor, \
        #                             nn.SiLU(inplace=True), \
        #                             nn.Linear(in_features=256, out_features=128, bias=True), \
        #                             nn.SiLU(inplace=True), \
        #                             nn.Linear(in_features=128, out_features=out_dim, bias=True))
    elif model_type == 'efficientnet_v2':
        vf_predictor = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        # vf_predictor.features[0][0] = nn.Conv2d(in_dim, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        if include_final:
            vf_predictor.classifier[1] = nn.Linear(in_features=1280, out_features=out_dim, bias=True)
        else:
            vf_predictor.classifier[1] = nn.Identity()
        # vf_predictor.avgpool = nn.Flatten()
        # vf_predictor.classifier = nn.Identity()
    elif model_type == 'resnet':
        vf_predictor = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        if in_dim != 3:
            vf_predictor.conv1 = nn.Conv2d(in_dim, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        if include_final:
            vf_predictor.fc = nn.Linear(in_features=2048, out_features=out_dim, bias=True)
        else:
            vf_predictor.fc = nn.Identity()
    elif model_type == 'swin':
        vf_predictor = swin_b(weights=Swin_B_Weights.IMAGENET1K_V1)
        # vf_predictor.features[0][0] = nn.Conv2d(in_dim, 128, kernel_size=(4, 4), stride=(4, 4))
        vf_predictor.head = nn.Linear(in_features=1024, out_features=out_dim, bias=True)
    elif model_type == 'vgg':
        vf_predictor = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        if in_dim != 3:
            vf_predictor.features[0] = nn.Conv2d(in_dim, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        if include_final:
            vf_predictor.classifier[6] = nn.Linear(in_features=4096, out_features=out_dim, bias=True)
        else:
            vf_predictor.classifier[6] = nn.Identity()
    elif model_type == 'resnext':
        # vf_predictor = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
        # vf_predictor.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # vf_predictor.fc = nn.Linear(in_features=2048, out_features=out_dim, bias=True)
        # vf_predictor = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V2)
        # vf_predictor.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # vf_predictor.fc = nn.Linear(in_features=2048, out_features=out_dim, bias=True)
        # vf_predictor = resnext101_32x8d(weights=ResNeXt101_32X8D_Weights.IMAGENET1K_V2)
        # vf_predictor.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # vf_predictor.fc = nn.Linear(in_features=2048, out_features=out_dim, bias=True)
        vf_predictor = resnext101_64x4d(weights=ResNeXt101_64X4D_Weights.IMAGENET1K_V1)
        # vf_predictor.conv1 = nn.Conv2d(in_dim, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        vf_predictor.fc = nn.Linear(in_features=2048, out_features=out_dim, bias=True)
    elif model_type == 'wideresnet':
        vf_predictor = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        # vf_predictor.conv1 = nn.Conv2d(in_dim, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        vf_predictor.fc = nn.Linear(in_features=2048, out_features=out_dim, bias=True)
    elif model_type == 'convnext':
        vf_predictor = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        # vf_predictor.features[0][0] = nn.Conv2d(in_dim, 96, kernel_size=(4, 4), stride=(4, 4))
        vf_predictor.classifier[2] = nn.Linear(in_features=768, out_features=out_dim, bias=True)
    elif model_type == 'yanet':
        vf_predictor = nn.Sequential(nn.Conv2d(in_dim, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False), 
                                    nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                    nn.SiLU(inplace=True),
                                    nn.MaxPool2d((2,2)),
                                    nn.Conv2d(32, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                                    nn.SiLU(inplace=True),
                                    nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
                                    nn.AdaptiveAvgPool2d(output_size=1),
                                    nn.Flatten(),
                                    nn.Dropout(p=0.2,inplace=True),
                                    nn.Linear(in_features=512, out_features=out_dim, bias=True))
                                    # nn.LayerNorm([32, 200, 200]),
    return vf_predictor

class Model_With_Time(nn.Module):
    def __init__(self, encoder=None, bias=True):
        super(Model_With_Time, self).__init__()
        self.encoder = encoder
        self.bias = bias
        self.classifier = nn.Linear(in_features=2, out_features=1, bias=self.bias)

    def forward(self, x, t):
        x_feat = self.encoder(x)
        x_feat = torch.cat((x_feat, t), dim=1)
        x_out = self.classifier(x_feat)
        return x_out

class OphBackbone_concat(nn.Module):
    def __init__(self, model_type='efficientnet', in_dim=1):
        super(OphBackbone, self).__init__()
        self.model_type = model_type
        self.in_dim = in_dim
        self.unit_feat_dim = 1280

        encoders = []
        for i in range(self.in_dim):
            # cur_encoder = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
            # cur_encoder.features[0][0] = nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            # cur_encoder.classifier[1] = nn.Identity()

            cur_encoder = efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V2)
            cur_encoder.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            cur_encoder.classifier[1] = nn.Linear(in_features=1280, out_features=1, bias=False)

            encoders.append(cur_encoder)
        self.encoders = nn.ModuleList(encoders)
        # self.encoder.classifier[1] = nn.Linear(in_features=1280, out_features=out_dim, bias=True)

        self.linear = nn.Linear(in_features=self.unit_feat_dim*self.in_dim, out_features=1, bias=True)

    def forward(self, x):
        x_out = []
        for i, l in enumerate(self.encoders):
            x_out.append(self.encoders[i](x[:,i:i+1,:,:]))
        x_out = torch.cat(x_out, dim=1)
        y = self.linear(x_out)
        return y

class OphBackbone(nn.Module):
    def __init__(self, model_type='efficientnet', in_dim=1, coef=1.):
        super(OphBackbone, self).__init__()
        self.model_type = model_type
        self.in_dim = in_dim
        self.unit_feat_dim = 1280
        self.coefs = [1., coef]

        encoders = []
        for i in range(self.in_dim):
            # cur_encoder = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
            # cur_encoder.features[0][0] = nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            # cur_encoder.classifier[1] = nn.Linear(in_features=1280, out_features=1, bias=True)

            cur_encoder = efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V2)
            cur_encoder.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            cur_encoder.classifier[1] = nn.Linear(in_features=1280, out_features=1, bias=False)
            
            encoders.append(cur_encoder)
        self.encoders = nn.ModuleList(encoders)

    def forward(self, x):
        x_out = None
        for i, l in enumerate(self.encoders):
            if x_out is None:
                x_out = self.coefs[i] * self.encoders[i](x[:,i:i+1,:,:])
            else:
                x_out += self.coefs[i] * self.encoders[i](x[:,i:i+1,:,:])
        y = x_out
        return y

class OphBackbone_(nn.Module):
    def __init__(self, model_type='efficientnet', in_dim=1):
        super(OphBackbone_, self).__init__()
        self.model_type = model_type
        self.in_dim = in_dim
        self.unit_feat_dim = 1280

        encoders = []
        for i in range(self.in_dim):
            cur_encoder = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
            cur_encoder.features[0][0] = nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            cur_encoder.classifier[1] = nn.Linear(in_features=1280, out_features=1, bias=True)
            encoders.append(cur_encoder)
        self.encoders = nn.ModuleList(encoders)
        # self.encoder.classifier[1] = nn.Linear(in_features=1280, out_features=out_dim, bias=True)

        # self.linear = nn.Linear(in_features=self.in_dim, out_features=1, bias=True)

    def forward(self, x):
        x_0 = self.encoders[0](x[:,0:0+1,:,:])
        x_1 = self.encoders[1](x[:,1:1+1,:,:])
        y = x_0 + x_1/(1+torch.abs(x_0))
        return y

class OphBackbone_Multiply(nn.Module):
    def __init__(self, model_type='efficientnet', in_dim=1):
        super(OphBackbone_, self).__init__()
        self.model_type = model_type
        self.in_dim = in_dim
        self.unit_feat_dim = 1280

        encoders = []
        for i in range(self.in_dim):
            cur_encoder = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
            cur_encoder.features[0][0] = nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            cur_encoder.classifier[1] = nn.Linear(in_features=1280, out_features=1, bias=True)
            encoders.append(cur_encoder)
        self.encoders = nn.ModuleList(encoders)
        # self.encoder.classifier[1] = nn.Linear(in_features=1280, out_features=out_dim, bias=True)

        # self.linear = nn.Linear(in_features=self.in_dim, out_features=1, bias=True)

    def forward(self, x):
        x_out = None
        for i, l in enumerate(self.encoders):
            # x_out.append(self.encoders[i](x[:,i:i+1,:,:]))
            if x_out is None:
                x_out = self.encoders[i](x[:,i:i+1,:,:])
            else:
                x_out *= self.encoders[i](x[:,i:i+1,:,:])
        y = x_out
        # x_out = torch.cat(x_out, dim=1)
        # y = self.linear(x_out)
        return y

def create_model_(model_type='efficientnet', in_dim=1, out_dim=1):
    vf_predictor = None
    if model_type == 'vit':
        vf_predictor = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        vf_predictor.conv_proj = nn.Conv2d(in_dim, 768, kernel_size=(16, 16), stride=(16, 16))
        vf_predictor.heads[0] = nn.Linear(in_features=768, out_features=out_dim, bias=True)
    elif model_type == 'efficientnet':
        vf_predictor = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        vf_predictor.features[0][0] = nn.Conv2d(in_dim, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        vf_predictor.classifier[1] = nn.Linear(in_features=1280, out_features=out_dim, bias=True)
    elif model_type == 'resnet':
        vf_predictor = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        vf_predictor.conv1 = nn.Conv2d(in_dim, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        vf_predictor.fc = nn.Linear(in_features=2048, out_features=out_dim, bias=True)
    elif model_type == 'swin':
        vf_predictor = swin_b(weights=Swin_B_Weights.IMAGENET1K_V1)
        vf_predictor.features[0][0] = nn.Conv2d(in_dim, 128, kernel_size=(4, 4), stride=(4, 4))
        vf_predictor.head = nn.Linear(in_features=1024, out_features=out_dim, bias=True)
    elif model_type == 'vgg':
        vf_predictor = vgg11(weights=VGG11_Weights.IMAGENET1K_V1)
        vf_predictor.features[0] = nn.Conv2d(in_dim, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        vf_predictor.classifier[6] = nn.Linear(in_features=4096, out_features=out_dim, bias=True)
    elif model_type == 'resnext':
        vf_predictor = resnext101_64x4d(weights=ResNeXt101_64X4D_Weights.IMAGENET1K_V1)
        vf_predictor.conv1 = nn.Conv2d(in_dim, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        vf_predictor.fc = nn.Linear(in_features=2048, out_features=out_dim, bias=True)
    elif model_type == 'wideresnet':
        vf_predictor = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
        vf_predictor.conv1 = nn.Conv2d(in_dim, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        vf_predictor.fc = nn.Linear(in_features=2048, out_features=out_dim, bias=True)
    elif model_type == 'convnext':
        vf_predictor = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        vf_predictor.features[0][0] = nn.Conv2d(in_dim, 96, kernel_size=(4, 4), stride=(4, 4))
        vf_predictor.classifier[2] = nn.Linear(in_features=768, out_features=out_dim, bias=True)
    return vf_predictor

# def create_model_(model_type='efficientnet', in_dim=1, out_dim=1):
#     vf_predictor = None
#     tds_module = None
#     out_feat = -1
#     if model_type == 'vit':
#         vf_predictor = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
#         vf_predictor.conv_proj = nn.Conv2d(1, 768, kernel_size=(16, 16), stride=(16, 16))
#         if in_dim == 2:
#             vf_predictor.heads[0] = nn.Identity()
#         else:
#             vf_predictor.heads[0] = nn.Linear(in_features=768, out_features=out_dim, bias=True)
#         # vf_predictor = vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_V1)
#         # vf_predictor.conv_proj = nn.Conv2d(1, 1024, kernel_size=(16, 16), stride=(16, 16))
#         # vf_predictor.heads.head = nn.Linear(in_features=1024, out_features=out_dim, bias=True)
#     elif model_type == 'efficientnet':
#         vf_predictor = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
#         vf_predictor.features[0][0] = nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#         if in_dim == 2:
#             vf_predictor.classifier[1] = nn.Identity()
#         else:
#             vf_predictor.classifier[1] = nn.Linear(in_features=1280, out_features=out_dim, bias=True)
#         # vf_predictor = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1)
#         # vf_predictor.classifier[1] = nn.Linear(in_features=1280, out_features=256, bias=True)
#         # vf_predictor = nn.Sequential(vf_predictor, \
#         #                             nn.SiLU(inplace=True), \
#         #                             nn.Linear(in_features=256, out_features=128, bias=True), \
#         #                             nn.SiLU(inplace=True), \
#         #                             nn.Linear(in_features=128, out_features=out_dim, bias=True))
#     elif model_type == 'resnet':
#         vf_predictor = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
#         vf_predictor.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#         if in_dim == 2:
#             vf_predictor.fc = nn.Identity()
#         else:
#             vf_predictor.fc = nn.Linear(in_features=2048, out_features=out_dim, bias=True)
#     elif model_type == 'swin':
#         vf_predictor = swin_b(weights=Swin_B_Weights.IMAGENET1K_V1)
#         vf_predictor.features[0][0] = nn.Conv2d(1, 128, kernel_size=(4, 4), stride=(4, 4))
#         if in_dim == 2:
#             vf_predictor.head = nn.Identity()
#         else:
#             vf_predictor.head = nn.Linear(in_features=1024, out_features=out_dim, bias=True)
#     elif model_type == 'vgg':
#         vf_predictor = vgg11(weights=VGG11_Weights.IMAGENET1K_V1)
#         vf_predictor.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
#         if in_dim == 2:
#             vf_predictor.classifier[6] = nn.Identity()
#         else:
#             vf_predictor.classifier[6] = nn.Linear(in_features=4096, out_features=out_dim, bias=True)
#     elif model_type == 'resnext':
#         # vf_predictor = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
#         # vf_predictor.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#         # vf_predictor.fc = nn.Linear(in_features=2048, out_features=out_dim, bias=True)
#         # vf_predictor = resnext50_32x4d(weights=ResNeXt50_32X4D_Weights.IMAGENET1K_V2)
#         # vf_predictor.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#         # vf_predictor.fc = nn.Linear(in_features=2048, out_features=out_dim, bias=True)
#         # vf_predictor = resnext101_32x8d(weights=ResNeXt101_32X8D_Weights.IMAGENET1K_V2)
#         # vf_predictor.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#         # vf_predictor.fc = nn.Linear(in_features=2048, out_features=out_dim, bias=True)
#         vf_predictor = resnext101_64x4d(weights=ResNeXt101_64X4D_Weights.IMAGENET1K_V1)
#         vf_predictor.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#         if in_dim == 2:
#             vf_predictor.fc = nn.Identity()
#         else:
#             vf_predictor.fc = nn.Linear(in_features=2048, out_features=out_dim, bias=True)
#     elif model_type == 'wideresnet':
#         vf_predictor = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
#         vf_predictor.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#         if in_dim == 2:
#             vf_predictor.fc = nn.Identity()
#         else:
#             vf_predictor.fc = nn.Linear(in_features=2048, out_features=out_dim, bias=True)
#     elif model_type == 'convnext':
#         vf_predictor = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
#         vf_predictor.features[0][0] = nn.Conv2d(1, 96, kernel_size=(4, 4), stride=(4, 4))
#         if in_dim == 2:
#             vf_predictor.classifier[2] = nn.Identity()
#         else:
#             vf_predictor.classifier[2] = nn.Linear(in_features=768, out_features=out_dim, bias=True)

#     # if in_dim == 2:
#     #     if model_type == 'efficientnet':
#     #         vf_predictor = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
#     #         vf_predictor.features[0][0] = nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
#     #         vf_predictor.classifier[1] = nn.Linear(in_features=1280, out_features=out_dim, bias=True)
#     #         tds_module.Sequential(nn.Linear(in_features=52, out_features=128, bias=True),
#     #                     nn.ReLU(inplace=True),
#     #                     nn.Linear(in_features=128, out_features=512, bias=False),
#     #                     nn.ReLU(inplace=True),
#     #                     nn.Linear(in_features=512, out_features=1280, bias=True))


#     return vf_predictor

class GlauClassifier(nn.Module):
    def __init__(self, ):
        super(GlauClassifier, self).__init__(model_type='efficientnet', in_dim=1, out_dim=1)
        self.model_type = model_type
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rnflt_encoder = create_model_(model_type=model_type, in_dim=in_dim, out_dim=out_dim)
        out_feat = -1
        if model_type == 'efficientnet':
            out_feat = 1280
        self.tds_encoder = nn.Sequential(nn.Linear(in_features=52, out_features=128, bias=True),
                                nn.ReLU(inplace=True),
                                nn.Linear(in_features=128, out_features=512, bias=False),
                                nn.ReLU(inplace=True),
                                nn.Linear(in_features=512, out_features=out_feat, bias=False))
        self.classifier = nn.Linear(in_features=out_feat*2, out_features=out_dim, bias=True)

    def forward(self, x, tds):
        rnflt_feat = self.rnflt_encoder(x)
        tds_feat = self.tds_encoder(tds)
        in_feat = torch.cat((rnflt_feat,tds_feat))
        y_hat = self.classifier(in_feat)
        return y_hat

class Model_Wrapper():
    """docstring for Model_Wrapper"""
    def __init__(self, model,
                result_dir='.',
                resume_checkpoint=None,
                ema_rate=0.9,
                identifier='predictor',
                logger=None):
        super().__init__()
        self.model = model
        self.model_params = list(model.parameters())
        # self.model_params = list(filter(lambda p: p.requires_grad, model.parameters()))
        self.ddp_model = DDP(
                model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        self.identifier = identifier

        self.resume_checkpoint = resume_checkpoint
        self.resume_epoch = 0
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )

        self.result_dir = result_dir
        self.logger = logger 

    def _log(self, text):
        if self.logger is not None:
            self.logger.log(text)
        else:
            print(text)

    def _load_checkpoint(self, opt=None):
        self._load_and_sync_parameters()
        if self.resume_checkpoint:
            if opt is not None:
                opt = self._load_optimizer_state(opt)
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.model_params) for _ in range(len(self.ema_rate))
            ]
        return opt

    def _load_and_sync_parameters(self):
        resume_checkpoint = self.resume_checkpoint

        if resume_checkpoint:
            self.resume_epoch = parse_resume_step_from_filename_(resume_checkpoint)
            if dist.get_rank() == 0:
                self._log(f"{self.identifier} - loading vf predictor from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.model_params)

        # main_checkpoint = resume_checkpoint
        ema_checkpoint = find_ema_checkpoint_(self.resume_checkpoint, self.resume_epoch, rate, self.identifier)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                self._log(f"{self.identifier} - loading vf EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self._state_dict_to_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _state_dict_to_params(self, state_dict):
        params = [state_dict[name] for name, _ in self.model.named_parameters()]
        # if self.use_fp16:
        #     return make_master_params(params)
        # else:
        return params

    def save(self, epoch, opt):
        def save_checkpoint(rate, epoch, params):
            state_dict = self._master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                self._log(f"saving {self.identifier} {rate}...")
                if not rate:
                    filename = f"{self.identifier}_{epoch:06d}.pt"
                else:
                    filename = f"ema_{self.identifier}_{rate}_{epoch:06d}.pt"
                with bf.BlobFile(bf.join(self.result_dir, filename), "wb") as f:
                    torch.save(state_dict, f)

        save_checkpoint(0, epoch, self.model_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, epoch, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(self.result_dir, f"opt_{self.identifier}_{epoch:06d}.pt"),
                "wb",
            ) as f:
                torch.save(opt.state_dict(), f)

        dist.barrier()

    def udpate_ema(self):
        def update_ema(target_params, source_params, rate=0.99):
            """
            Update target parameters to be closer to those of source parameters using
            an exponential moving average.

            :param target_params: the target parameter sequence.
            :param source_params: the source parameter sequence.
            :param rate: the EMA rate (closer to 1 means slower).
            """
            for targ, src in zip(target_params, source_params):
                targ.detach().mul_(rate).add_(src, alpha=1 - rate)

        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.model_params, rate=rate)

    def _master_params_to_state_dict(self, master_params):
        # if self.use_fp16:
        #     master_params = unflatten_master_params(
        #         self.model.parameters(), master_params
        #     )
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        params = [state_dict[name] for name, _ in self.model.named_parameters()]
        # if self.use_fp16:
        #     return make_master_params(params)
        # else:
        return params


    def _load_optimizer_state(self, opt):
        main_checkpoint = self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt_{self.identifier}_{self.resume_epoch:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            self._log(f"{self.identifier} - loading vf optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            opt.load_state_dict(state_dict)
        return opt


def find_ema_checkpoint_vf(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_vf_predictor_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None
        
def parse_resume_step_from_filename_vf(filename):
    """
    Parse filenames of the form path/to/vf_predictor_NNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("vf_predictor_")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0

def find_ema_checkpoint_(main_checkpoint, step, rate, identifier):
    if main_checkpoint is None:
        return None
    filename = f"ema_{identifier}_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None

def parse_resume_step_from_filename_(filename, identifier='predictor'):
    """
    Parse filenames of the form path/to/vf_predictor_NNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split(f"{identifier}_")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0

def get_blob_logdir():
    return os.environ.get("DIFFUSION_BLOB_LOGDIR", logger.get_dir())

def classify_glaucoma(mds, num_class=2, th=-1):
    # glau = np.where(mds<-3.0, np.ones_like(mds), np.zeros_like(mds))
    if num_class==3:
        borderline = np.where((mds>=-3.0) & (mds<-1.0), np.ones_like(mds), np.zeros_like(mds))
        non_glau = np.where(mds>=-1.0, np.ones_like(mds), np.zeros_like(mds))
        y_pred = borderline + non_glau * 2
    elif num_class==2:
        y_pred = np.where(mds>=th, np.ones_like(mds), np.zeros_like(mds))
    return y_pred

def classify(prob):
    y = (prob>=0.5).astype(float)
    return y

def to_one_hot_vector(arr):
    arr = arr.astype(int)
    shape = (arr.shape[0], arr.max()+1)
    one_hot = np.zeros(shape)
    rows = np.arange(arr.shape[0])
    one_hot[rows, arr] = 1
    return one_hot

def compute_weight(weight, step, rampup_step=4000):
    return weight if rampup_step == 0 or step > rampup_step else weight * step / rampup_step

#=====> lr schedulers

# https://github.com/godofpdog/ViT_PyTorch/blob/af086058764e55a48043db7f6f7c32b685db9427/vit_pytorch/solver.py
def get_scheduler(optimizer, args):
    if args.scheduler is not None:
        if args.scheduler == 'cosine':
            return CosineAnnealingLR(optimizer, args.t_max, args.eta_min)
        elif args.scheduler == 'step':
            return StepLR(optimizer, args.step_size, args.gamma)
        elif args.scheduler == 'exp':
            return ExponentialLR(optimizer, args.gamma)
        else:
            raise ValueError('Invalid scheduler.')
    else:
        return ConstantScheduler(optimizer)

# https://github.com/jeonsworld/ViT-pytorch/blob/main/utils/scheduler.py

class ConstantLRSchedule(LambdaLR):
    """ Constant learning rate schedule.
    """
    def __init__(self, optimizer, last_epoch=-1):
        super(ConstantLRSchedule, self).__init__(optimizer, lambda _: 1.0, last_epoch=last_epoch)

class WarmupConstantSchedule(LambdaLR):
    """ Linear warmup and then constant.
        Linearly increases learning rate schedule from 0 to 1 over `warmup_steps` training steps.
        Keeps learning rate schedule equal to 1. after warmup_steps.
    """
    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(WarmupConstantSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        return 1.


class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))


class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """
    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))
#<===== lr schedulers
