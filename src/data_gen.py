from scipy.io import loadmat
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.model_selection import train_test_split
import random
import tifffile as tiff
import math
import os


def load_data(data_sign, data_path_prefix):
    if data_sign == "Indian":
        data = sio.loadmat('%s/Indian_pines_corrected.mat' % data_path_prefix)['indian_pines_corrected']
        labels = sio.loadmat('%s/Indian_pines_gt.mat' % data_path_prefix)['indian_pines_gt']
    elif data_sign == "Pavia":
        data = sio.loadmat('%s/PaviaU.mat' % data_path_prefix)['paviaU']
        labels = sio.loadmat('%s/PaviaU_gt.mat' % data_path_prefix)['paviaU_gt'] 
    elif data_sign == "Houston":
        data = sio.loadmat('%s/Houston.mat' % data_path_prefix)['img']
        labels = sio.loadmat('%s/Houston_gt.mat' % data_path_prefix)['Houston_gt']
    elif data_sign == 'Salinas':
        data = sio.loadmat('%s/Salinas_corrected.mat' % data_path_prefix)['salinas_corrected']
        labels = sio.loadmat('%s/Salinas_gt.mat' % data_path_prefix)['salinas_gt']
    elif data_sign == 'WH' or data_sign=='Honghu':
        data = sio.loadmat('%s/WHU_Hi_HongHu.mat' % data_path_prefix)['WHU_Hi_HongHu']
        labels = sio.loadmat('%s/WHU_Hi_HongHu_gt.mat' % data_path_prefix)['WHU_Hi_HongHu_gt']
    return data, labels

def gen(data_sign, train_num_per_class, data_path_prefix, max_percent=0.5):
    data, labels = load_data(data_sign, data_path_prefix)
    h, w, c = data.shape
    class_num = labels.max()
    class2data = {}
    for i in range(h):
        for j in range(w):
            if labels[i,j] > 0:
                if labels[i, j] in class2data:
                    class2data[labels[i,j]].append([i, j])
                else:
                    class2data[labels[i,j]] = [[i,j]]

    TR = np.zeros_like(labels)
    TE = np.zeros_like(labels)
    for cl in range(class_num):
        class_index = cl + 1
        ll = class2data[class_index]
        all_index = list(range(len(ll)))
        real_train_num = train_num_per_class
        if len(all_index) <= train_num_per_class:
            real_train_num = int(len(all_index) * max_percent) 
        select_train_index = set(random.sample(all_index, real_train_num))
        for index in select_train_index:
            item = ll[index]
            TR[item[0], item[1]] = class_index
    TE = labels - TR
    target = {}
    target['TE'] = TE
    target['TR'] = TR
    target['input'] = data
    return target

def gen_pct(data_sign, data_path_prefix, real_pct=0.01):
    data, labels = load_data(data_sign, data_path_prefix)
    h, w, c = data.shape
    class_num = labels.max()
    index_data = []
    for i in range(h):
        for j in range(w):
            if labels[i,j] > 0:
                index_data.append([i, j])
    sample_num = len(index_data)
    train_num = int(sample_num * real_pct)
    test_num = sample_num - train_num

    TR = np.zeros_like(labels)
    TE = np.zeros_like(labels)

    index_ll = list(range(len(index_data)))
    train_ll = set(random.sample(index_ll, train_num))
    for ii in train_ll:
        item = index_data[ii]
        TR[item[0], item[1]] = labels[item[0], item[1]]
    TE = labels - TR
    target = {}
    target['TE'] = TE
    target['TR'] = TR
    target['input'] = data
    atr =  TR[TR>0].shape[0]
    ate =  TE[TE>0].shape[0]
    aa = labels[labels>0].shape[0]
    print('train=%s, test=%s, all=%s' % (atr, ate, aa))
    return target


def gen_per_class_pct(data_sign, data_path_prefix, real_pct):
    data, labels = load_data(data_sign, data_path_prefix)
    h, w, c = data.shape
    class_num = labels.max()
    class2data = {}
    for i in range(h):
        for j in range(w):
            if labels[i,j] > 0:
                if labels[i, j] in class2data:
                    class2data[labels[i,j]].append([i, j])
                else:
                    class2data[labels[i,j]] = [[i,j]]

    TR = np.zeros_like(labels)
    TE = np.zeros_like(labels)
    for cl in range(class_num):
        class_index = cl + 1
        ll = class2data[class_index]
        all_index = list(range(len(ll)))
        # real_train_num = round(len(ll) * real_pct)
        real_train_num = math.ceil(len(ll) * real_pct)
        select_train_index = set(random.sample(all_index, real_train_num))
        for index in select_train_index:
            item = ll[index]
            TR[item[0], item[1]] = class_index
    TE = labels - TR
    target = {}
    target['TE'] = TE
    target['TR'] = TR
    target['input'] = data
    ntr = TR[TR>0].shape[0]
    nte = TE[TE>0].shape[0]
    print('train=%s, test=%s' % (ntr, nte))
    return target

def run_temp():
    signs = ['Indian', 'Pavia']
    data_path_prefix = '../data'
    if not os.path.exists(data_path_prefix):
        os.makedirs(data_path_prefix)
    train_num_per_class_list = [30]
    # train_num_per_class_list = [60,70,80]
    times = 1
    for data_sign in signs:
        for train_num_per_class in train_num_per_class_list:
            for t in range(times):
                dir_ = '../data/%s' % data_sign
                os.makedirs(dir_, exist_ok=True)
                save_path = dir_ + '/%s_%s_%s_split.mat' %(data_sign, train_num_per_class, t)
                target = gen(data_sign, train_num_per_class, data_path_prefix)
                sio.savemat(save_path, target)
                print('save %s done.' % save_path)

def run():
    signs = ['Indian', 'Pavia', 'WH']
    data_path_prefix = '../../data'
    train_num_per_class_list = [10]
    # train_num_per_class_list = [60,70,80]
    for data_sign in signs:
        for train_num_per_class in train_num_per_class_list:
            save_path = '../../data/%s/%s_%s_split.mat' %(data_sign, data_sign, train_num_per_class)
            target = gen(data_sign, train_num_per_class, data_path_prefix)
            sio.savemat(save_path, target)
            print('save %s done.' % save_path)



def run_pct():
    signs = ['Indian','Pavia','WH']
    data_path_prefix = '../../data'
    train_pct = [0.1]
    for data_sign in signs:
        for pct in train_pct:
            save_path = '../../data/%s/%s_%s_split.mat' %(data_sign, data_sign, pct)
            target = gen_pct(data_sign, data_path_prefix, real_pct=pct)
            sio.savemat(save_path, target)
            print('save %s done.' % save_path)

def run_per_class_pct():
    signs = ['WH']
    data_path_prefix = '../../data'
    train_pct = [0.001]
    for data_sign in signs:
        for pct in train_pct:
            save_path = '../../data/%s/%s_%s_pc_split.mat' %(data_sign, data_sign, pct)
            target = gen_per_class_pct(data_sign, data_path_prefix, real_pct=pct)
            sio.savemat(save_path, target)
            print('save %s done.' % save_path)

if __name__ == "__main__":
    print(os.getcwd())
    run_temp()

