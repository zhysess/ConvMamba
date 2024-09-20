import os, sys, time, json
import time
from utils import recorder
from data_provider.data_provider import HSIDataLoader 
from trainer import get_trainer
from utils import config_path_prefix
import torch
import numpy as np
import scipy.io as sio
import plot
DEFAULT_RES_SAVE_PATH_PREFIX = "./exp_result/"


# random seed setting
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def pred2img(pred, param):
    def plot_one(name, cls_data, raw_data, save_path, dpi=100):
        cls_data = cls_data.astype(np.int8)
        h,w,c = raw_data.shape
        indian_color = plot.data_to_colormap2(cls_data).reshape((h,w,3))
        plot.classification_map(name, indian_color, raw_data, dpi, save_path)
    
    pred = pred + 1
    data_path = '%s/%s/%s_split.mat' % (param['data']['data_path_prefix'], \
                                        param['data']['data_sign'], \
                                        param['data']['data_file'])
    mat = sio.loadmat(data_path)
    data = mat['input']
    TR = mat['TR']
    TE = mat['TE']
    label = TR + TE 

    pred = pred.reshape(label.shape)
    pred[label==0] = 0
    plot_one('ConvMamba', pred, data, \
             f"{DEFAULT_RES_SAVE_PATH_PREFIX}/{param['data']['data_sign']}/pred", dpi=300)
def train_by_param(param, path):
    setup_seed(0)
    recorder.reset()
    if not os.path.exists(path):
        os.makedirs(path)
    dataloader = HSIDataLoader(param)
    train_loader,unlabel_loader, test_loader, all_loader = dataloader.generate_torch_dataset() 
    trainer = get_trainer(param)
    trainer.train(train_loader, unlabel_loader,test_loader)

    y_pred_test, y_test, eval_time = trainer.final_eval(all_loader)
    pred2img(y_pred_test, param)
    print("eval time is %s" % eval_time)
    print(f"{param['data']['data_sign']}, OA: {recorder.record_data['eval']['oa']}, AA: {recorder.record_data['eval']['aa']}, \
          kappa: {recorder.record_data['eval']['kappa']}")
    recorder.record_time(eval_time)
    recorder.record_param(param)
    recorder.to_file(path)


include_path = [
    'convmamba.json',
]


def run_all():
    save_path_prefix = DEFAULT_RES_SAVE_PATH_PREFIX
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)
    for name in include_path:
        path_param = '%s/%s' % (config_path_prefix, name)
        with open(path_param, 'r') as fin:
            param = json.loads(fin.read())
        dataset = ["Indian"]
        for ds in dataset:
            if ds == "Honghu":
                prefix = 'data_HH'
                datasets = [0]
            if ds == "Indian":
                prefix = 'data_IP'
                datasets = [0]
            if ds == "Pavia":
                prefix = 'data_PU'
                datasets = [0]
            param['data'] = param[prefix]
            for i in datasets: 
                param['data']['data_file'] = f"{param['data']['data_sign']}_30_{i}"
                print("start to train " + f"{param['data']['data_sign']}_{i}")
                path = f"{save_path_prefix}/{param['data']['data_sign']}/"
                train_by_param(param, path)
                print('model eval done')


if __name__ == "__main__":
    run_all()


