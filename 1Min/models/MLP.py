import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], dropout_prob=0.2):
        """
        简单的 3 层 MLP Baseline 模型
        Args:
            input_dim: 输入特征的维度（即因子的个数）
            hidden_dims: 隐藏层神经元数量列表，默认三层
            dropout_prob: Dropout 概率，用于防止过拟合
        """
        super(Model, self).__init__()
        
        # 第一层：Input -> Hidden 1
        self.layer1 = nn.Linear(input_dim, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        
        # 第二层：Hidden 1 -> Hidden 2
        self.layer2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])
        
        # 第三层：Hidden 2 -> Hidden 3
        self.layer3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.bn3 = nn.BatchNorm1d(hidden_dims[2])
        
        # 输出层：Hidden 3 -> Output (预测值 y)
        self.output_layer = nn.Linear(hidden_dims[2], 1)
        
        self.dropout = nn.Dropout(dropout_prob)
        
        # 初始化权重（使用 Kaiming 初始化，适合 ReLU 激活函数）
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        前向传播逻辑
        x shape: [batch_size, input_dim]
        """
        # Layer 1
        x = self.layer1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Layer 2
        x = self.layer2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Layer 3
        x = self.layer3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Output
        x = self.output_layer(x)
        return x

