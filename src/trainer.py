import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models.pipeline import Classifier
import utils
from utils import recorder
from evaluation import HSIEvaluation
from torch.utils.tensorboard import SummaryWriter
from utils import device
import copy
import time
import os 


class BaseTrainer(object):
    def __init__(self, params) -> None:
        self.params = params
        self.net_params = params['net']
        self.train_params = params['train']
        self.device = device 
        self.evalator = HSIEvaluation(param=params)

        self.net = None
        self.criterion = None
        self.optimizer = None
        self.clip = 15
        self.unlabel_loader=None
        self.real_init()
        save_weight = './weight'
        os.makedirs(save_weight, exist_ok=True)
        self.best_model_path = os.path.join(save_weight, f"{self.params['data']['data_sign']}_best_model.pth")  # 保存最好的模型的路径

    def real_init(self):
        pass

    def get_loss(self, outputs, target):
        return self.criterion(outputs, target)
       
    def train(self, train_loader, unlabel_loader=None, test_loader=None):
        epochs = self.params['train'].get('epochs', 100)
        total_loss = 0
        epoch_avg_loss = utils.AvgrageMeter()
        # 创建 SummaryWriter
        # writer = SummaryWriter("exp2")
        best_oa = 0.0
        for epoch in range(epochs):
            self.net.train()
            epoch_avg_loss.reset()
            for i, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.net(data)
                loss = self.get_loss(outputs, target)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clip)
                self.optimizer.step()
                # batch stat
                total_loss += loss.item()
                epoch_avg_loss.update(loss.item(), data.shape[0])
            recorder.append_index_value("epoch_loss", epoch + 1, epoch_avg_loss.get_avg())
            # print('[Epoch: %d]  [epoch_loss: %.5f]  [all_epoch_loss: %.5f] [current_batch_loss: %.5f] [batch_num: %s]' % (epoch + 1,epoch_avg_loss.get_avg(),total_loss / (epoch + 1),loss.item(), epoch_avg_loss.get_num()))
            
            # writer.add_scalar('train_epoch_loss', epoch_avg_loss.get_avg(), epoch+1)
            
            # 一定epoch下进行一次eval
            if test_loader and (epoch+1) % 5 == 0:
                y_pred_test, y_test, evl_loss = self.test(test_loader)

                temp_res = self.evalator.eval(y_test, y_pred_test)
                recorder.append_index_value("train_oa", epoch+1, temp_res['oa'])
                recorder.append_index_value("train_aa", epoch+1, temp_res['aa'])
                recorder.append_index_value("train_kappa", epoch+1, temp_res['kappa'])

                if temp_res['oa'] > best_oa:
                    best_oa = temp_res['oa']
                    recorder.record_eval(copy.deepcopy(temp_res))
                    # 保存最好的模型权重
                    torch.save(self.net.state_dict(), self.best_model_path)
                # writer.add_scalars('evl', {
                #     "loss": evl_loss,
                #     "train_oa": temp_res['oa']/100,
                #     "train_aa": temp_res['aa']/100,
                #     "train_kappa": temp_res['kappa']/100
                # }, epoch+1)
                print('[--TEST--] [Epoch: %d] [oa: %.5f] [aa: %.5f] [kappa: %.5f] [num: %s]' % (epoch+1, temp_res['oa'], temp_res['aa'], temp_res['kappa'], str(y_test.shape)))
                    
        print('Finished Training')
        return True

    def final_eval(self, test_loader):
        self.net.load_state_dict(torch.load(self.best_model_path))
        start_eval_time = time.time()
        y_pred_test, y_test = self.all_test(test_loader)
        end_eval_time = time.time()
        eval_time = end_eval_time - start_eval_time
        # temp_res = self.evalator.eval(y_test, y_pred_test)
        return y_pred_test, y_test, eval_time

    def get_logits(self, output):
        if type(output) == tuple:
            return output[0]
        return output

    def test(self, test_loader):
        """
        provide test_loader, return test result(only net output)
        """
        count = 0

        self.net.eval()
        y_pred_test = 0
        y_test = 0
        epoch_avg_loss = utils.AvgrageMeter()
        epoch_avg_loss.reset()

        for inputs, labels in test_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            outputs = self.net(inputs) 
            loss = self.get_loss(outputs, labels)
            epoch_avg_loss.update(loss.item(), inputs.shape[0])
            
            outputs = self.get_logits(outputs)
            if len(outputs.shape) == 1:
                continue
            outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
            labels = labels.cpu()

            if count == 0:
                y_pred_test = outputs
                y_test = labels
                count = 1
            else:
                y_pred_test = np.concatenate((y_pred_test, outputs))
                y_test = np.concatenate((y_test, labels))
        return y_pred_test, y_test, epoch_avg_loss.get_avg()
    def all_test(self, all_loader):
        """
        provide test_loader, return test result(only net output)
        """
        count = 0
        self.net.eval()
        y_pred_test = 0
        y_test = 0
        for inputs, labels in all_loader:
            inputs = inputs.to(self.device)
            outputs = self.get_logits(self.net(inputs))
            if len(outputs.shape) == 1:
                continue
            outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
            if count == 0:
                y_pred_test = outputs
                y_test = labels
                count = 1
            else:
                y_pred_test = np.concatenate((y_pred_test, outputs))
                y_test = np.concatenate((y_test, labels))
        return y_pred_test, y_test




class MambaTrainer(BaseTrainer):
    def __init__(self, params):
        super(MambaTrainer, self).__init__(params)


    def real_init(self):
        # net
        self.net = Classifier(self.params).to(self.device)
        # loss
        self.criterion = nn.CrossEntropyLoss()
        # optimizer
        self.lr = self.train_params.get('lr', 0.001)
        self.weight_decay = self.train_params.get('weight_decay', 5e-3)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def get_loss(self, outputs, target):
        '''
            A_vecs: [batch, dim]
            B_vecs: [batch, dim]
            logits: [batch, class_num]
        '''
        logits = outputs
        
        loss_main = nn.CrossEntropyLoss()(logits, target) 

        return loss_main   


def get_trainer(params):
    trainer_type = params['net']['trainer']
    if trainer_type == "convmamba":
        return MambaTrainer(params)
    assert Exception("Trainer not implemented!")

