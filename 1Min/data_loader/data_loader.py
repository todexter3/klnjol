import sys
import pandas as pd
import numpy as np
import json
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.append('/cpfs/fs3200/ppl/cta/project/flap01/common/python_modules/flap/optimized_frame/new_df_frame')
from gen_utils import merge_factor

class CTADataProcessor:
    def __init__(self, basic_path, input_format='feather'):
        self.basic_path = basic_path
        self.input_format = input_format

    #cjue
    def load_and_merge(self, label_path, selectF_infos, base_pathInfos, train_start, end_year, label_name='y20'):
        print(f"Loading labels: {label_name} for years {train_start}-{end_year}")
        label_file = f"{label_path}{label_name}.{self.input_format}"
        label_df = merge_factor(self.basic_path, [label_file], 
                                input_format=self.input_format, 
                                start_year=train_start, end_year=end_year, verbose=0)
        label_df = label_df.rename(columns={'date': 'time'})

        # 2. 读取因子配置并筛选路径
        # 假设 test_year 是 end_year + 1
        test_year_suffix = str(end_year + 1)[2:]
        selectF_fname = f"{selectF_infos}{label_name}/{test_year_suffix}.json"
        
        with open(selectF_fname, 'r') as file:
            # 这里的逻辑参考原文件：提取 factor_list 中的因子名
            selectF_list = [i.split('.')[2] for i in json.loads(file.read())['factor_list']]
        
        # 筛选对应的二进制路径
        base_selectF_paths = sorted(base_pathInfos[base_pathInfos['fname'].isin(selectF_list)]['bin_path'])

        # 3. 合并因子数据
        print(f"Merging {len(base_selectF_paths)} factors with 40 jobs...")
        factors_df = merge_factor(self.basic_path, base_selectF_paths, 
                                  n_jobs=40, input_format=self.input_format, 
                                  start_year=train_start, end_year=end_year)
        
        # 4. 内连接对齐
        final_df = pd.merge(factors_df, label_df[['time', 'id', label_name]], on=['time', 'id'], how='inner')
        return final_df

class Dataset(Dataset):
    def __init__(self, df, label_col):
        # 自动识别因子列（剔除 time, id 和 label）
        self.feature_cols = [c for c in df.columns if c not in ['time', 'id', label_col]]
        
        # 转换为 Numpy 并处理空值
        self.X = df[self.feature_cols].values.astype(np.float32)
        self.y = df[label_col].values.astype(np.float32)
        
        self.X = np.nan_to_num(self.X, nan=0.0)
        self.y = np.nan_to_num(self.y, nan=0.0)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(np.array([self.y[idx]]))

def get_dataloader(config_dict, is_train=True):
    """
    一键式获取 DataLoader
    config_dict 包含所有路径参数
    """
    processor = CTADataProcessor(config_dict['basic_path'])
    
    df = processor.load_and_merge(
        label_path=config_dict['label_path'],
        selectF_infos=config_dict['selectF_infos'],
        base_pathInfos=config_dict['base_pathInfos'],
        train_start=config_dict['start_year'],
        end_year=config_dict['end_year'],
        label_name=config_dict['label_name']
    )
    
    dataset = Dataset(df, config_dict['label_name'])
    loader = DataLoader(dataset, batch_size=config_dict['batch_size'], shuffle=is_train, num_workers=4)
    
    return loader