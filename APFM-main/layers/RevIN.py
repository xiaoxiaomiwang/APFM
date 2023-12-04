import torch
import torch.nn as nn

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        num_features (int): 特征或通道的数量。
        eps (float): 用于数值稳定性的添加值。
        affine (bool): 如果为 True，则 RevIN 具有可学习的仿射参数，先将输入张量 x 乘以 self.affine_weight 进行缩放，然后加上 self.affine_bias 进行平移，对数据进行自适应的线性变换，以增强模型的表达能力。
        subtract_last (bool): 如果为 True，则从标准化中减去最后一个值。
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

        self.fc = nn.Linear(num_features, num_features)
        self.fc_denorm = nn.Linear(num_features, num_features)

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))
        ##########全连接替换###########
        self.linear_layer1 = nn.Linear(self.num_features, self.num_features, bias=True)
        self.linear_layer2 = nn.Linear(self.num_features, self.num_features, bias=True)

    def _get_statistics(self, x): #计算均值和标准差
        dim2reduce = tuple(range(1, x.ndim-1)) #为了计算均值和标准差，我们需要指定要减少的维度。range(1, x.ndim-1) 用于生成一个范围对象，表示从第二个维度到倒数第二个维度（排除第一个和最后一个维度）.将范围对象转换为元组，用于指定要在哪些维度上进行统计计算。
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1) #我们将最后一个时间步的值提取出来，并将其维度扩展为 (batch_size, 1, num_features)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach() #我们计算除最后一维以外的所有维度的均值，并将其维度保持不变，得到大小为 (batch_size, 1, num_features) 的均值张量
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach() #计算除最后一维以外的所有维度的方差，并添加一个小的值 eps 以增加数值稳定性。然后，我们取方差的平方根，得到标准差张量，并将其维度保持不变，得到大小为 (batch_size, 1, num_features) 的标准差张量

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last #从输入张量 x 中减去 self.last，即最后一个时间步的值
        else:
            x = x - self.mean #从输入张量 x 中减去 self.mean，即均值
        x = x / self.stdev #将结果除以 self.stdev，即标准差，以进行标准化
        if self.affine: #将标准化后的张量乘以 self.affine_weight，然后加上 self.affine_bias，以进行仿射变换
            x = x * self.affine_weight
            x = x + self.affine_bias
            # x = self.fc(x)
            # x = self.linear_layer1(x)

        return x

    def _denormalize(self, x):
        if self.affine: #先从输入张量 x 中减去 self.affine_bias，然后除以 self.affine_weight + self.eps*self.eps。这是对标准化的逆操作，以恢复原始的数值范围。
            x = self.fc_denorm(x)
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
            x = self.linear_layer2(x)

        x = x * self.stdev #将结果乘以标准差张量 self.stdev，以进行反标准化操作
        if self.subtract_last:
            x = x + self.last #将最后一个时间步的值 self.last 加回到反标准化的结果中。
        else:
            x = x + self.mean #将均值 self.mean 加回到反标准化的结果中
        return x
