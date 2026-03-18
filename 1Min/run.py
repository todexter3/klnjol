import argparse
import os
import torch
import pandas as pd
from exp.exp import Exp

def main():

    # 实例化实验控制器
    exp = Exp(args)

    # 1. 训练与验证
    print('>>>>>>> Start Training >>>>>>>>>>>>>>>>>>>>>>>>>>')
    exp.train()

    # 2. 测试
    print('>>>>>>> Start Testing >>>>>>>>>>>>>>>>>>>>>>>>>>')
    preds, trues = exp.test()

    # 3. 保存结果 (可选)
    res_path = os.path.join(args.checkpoints, args.model_id, 'results.pth')
    torch.save({'preds': preds, 'trues': trues}, res_path)
    print(f'Results saved to {res_path}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MLP Baseline for CTA Factor Research')

    # 基本配置
    parser.add_argument('--model_id', type=str, default='cta_mlp_v1', help='模型运行标识')
    parser.add_argument('--model', type=str, default='MLP', help='模型运行标识')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='模型保存路径')

    # 数据路径 (对应 read_mode.ipynb 中的配置)
    parser.add_argument('--basic_path', type=str, default='/cpfs/dss/dev/cta/project/flap02/checkin/data/cta/1m/v1/', help='数据服务器根路径')
    parser.add_argument('--label_path', type=str, default='/cpfs/dss/dev/cta/project/flap01/p9/factor_set/binary/jchen/priceMain_set/1m/cta/autoRedig/saved/label_volAdj/',help='标签文件目录路径')
    parser.add_argument('--selectF_infos', type=str, default='/cpfs/dss/dev/cta/project/flap01/p9/factor_set/binary/jchen/priceMain_set/1m/cta/autoRedig/filter_json/json_res/first_step_volDealt_intraCrossDay_rnd1Test_corr08/autopriceMain_first_step_volDealt_intraCrossDay_thres_top_300_trunc_icts/', help='因子筛选 JSON 目录')
    parser.add_argument('--path_infos_csv', type=str, default='/cpfs/dss/dev/cta/project/flap01/p9/factor_set/binary/jchen/priceMain_set/1m/cta/autoRedig/autopriceMain_first_step_volDealt_intraCrossDay_pathmap.csv',help='包含因子 bin_path 的 CSV 文件路径')
    
    # 任务参数
    parser.add_argument('--label_name', type=str, default='y20', help='预测目标名称 (y4, y20, y80等)')
    parser.add_argument('--input_dim', type=int, help='因子特征维度 (由脚本自动获取或手动指定)')
    
    # 时间区间
    parser.add_argument('--train_start', type=int, default=2010, help='训练集开始年份')
    parser.add_argument('--train_end', type=int, default=2018, help='训练集结束年份')
    parser.add_argument('--val_start', type=int, default=2019, help='验证集开始年份')
    parser.add_argument('--val_end', type=int, default=2019, help='验证集结束年份')
    parser.add_argument('--test_start', type=int, default=2020, help='测试集开始年份')
    parser.add_argument('--test_end', type=int, default=2024, help='测试集结束年份')

    # 训练超参数
    parser.add_argument('--batch_size', type=int, default=2048, help='批大小')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--train_epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout 概率')

    # GPU 配置
    parser.add_argument('--use_gpu', type=bool, default=True, help='是否使用 GPU')
    parser.add_argument('--gpu', type=int, default=0, help='主 GPU 设备 ID')
    parser.add_argument('--use_multi_gpu', action='store_true', help='是否使用多显卡并行')
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='多显卡 ID 列表')

    args = parser.parse_args()

    # 处理多卡逻辑
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    # 加载因子路径信息表 (核心元数据)
    print("Loading path information...")
    args.base_pathInfos = pd.read_csv(args.path_infos_csv)

    main()