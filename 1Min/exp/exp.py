import os
import torch
import torch.nn as nn
from torch import optim
import numpy as np
from exp.exp_basis import Exp_Basic
from data_loader.data_loader import get_dataloader

class Exp(Exp_Basic):
    def __init__(self, args):
        super(Exp, self).__init__(args)
        self.all_test_preds = np.array([])
        

    def _build_model(self):
        self.model = self.model_dict[self.args.model].Model(self.args).float().to(self.device)
        return self.model

    def _acquire_device(self):
        if self.args.use_gpu:
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device
    


    def _get_data(self, flag):
        # 根据训练、验证、测试阶段切换年份跨度
        if flag == 'train':
            start_year = self.args.train_start
            end_year = self.args.train_end
            shuffle = True
        elif flag == 'val':
            start_year = self.args.val_start
            end_year = self.args.val_end
            shuffle = False
        else: # test
            start_year = self.args.test_start
            end_year = self.args.test_end
            shuffle = False

        # 封装 read_mode.ipynb 逻辑的配置字典
        data_config = {
            'basic_path': self.args.basic_path,
            'label_path': self.args.label_path,
            'selectF_infos': self.args.selectF_infos,
            'base_pathInfos': self.args.base_pathInfos, # 预先读好的df
            'start_year': start_year,
            'end_year': end_year,
            'label_name': self.args.label_name,
            'batch_size': self.args.batch_size
        }

        data_loader = get_baseline_loader(data_config, is_train=shuffle)
        return data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        # 量化收益率预测通常使用 MSE
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                total_loss.append(loss.item())
        
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self):
        train_loader = self._get_data(flag='train')
        vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, self.args.model_id)
        if not os.path.exists(path):
            os.makedirs(path)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        for epoch in range(self.args.train_epochs):
            train_loss = []
            self.model.train()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                model_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                loss.backward()
                model_optim.step()

            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader, criterion)

            print(f"Epoch: {epoch + 1} | Train Loss: {train_loss:.6f} Vali Loss: {vali_loss:.6f}")
            
            # 简单的模型保存逻辑
            torch.save(self.model.state_dict(), path + '/' + 'checkpoint.pth')

        return self.model

    def test(self):
        test_loader = self._get_data(flag='test')
        self.model.eval()
        
        preds = []
        trues = []
        
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                outputs = self.model(batch_x)
                
                preds.append(outputs.detach().cpu().numpy())
                trues.append(batch_y.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        
        # 计算量化常用指标：IC (信息系数)
        # 假设数据是按时间对齐的
        ic = np.corrcoef(preds.flatten(), trues.flatten())[0, 1]
        print(f"Test IC: {ic:.4f}")
        
        return preds, trues