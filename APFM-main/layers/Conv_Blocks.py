import torch
import torch.nn as nn
##########padding = dilation * (kernel_size - 1) / 2###########

# class Inception_Block_V1(nn.Module):
#     def __init__(self, in_channels, out_channels, init_weight=True):
#         super(Inception_Block_V1, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#
#         # 使用 ModuleList 动态生成卷积核
#         self.conv1_out = nn.ModuleList([
#             nn.Conv1d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i)
#             for i in range(self.num_kernels)
#         ])
#
#         if init_weight:
#             self._initialize_weights()
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#
#     def forward(self, x, period_list):
#         # 使用 period_list 中的每个值作为 num_kernels
#         self.num_kernels = len(period_list)
#         # 使用 ModuleList 中的每个卷积核进行前向传播
#         res_list = [conv(x) for conv in self.conv1_out]
#         # 将结果拼接起来
#         res = torch.stack(res_list, dim=-1).mean(-1)
#         return res


# class Inception_Block_V1(nn.Module):
#     def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
#         super(Inception_Block_V1, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.num_kernels = num_kernels
#         # kernels = []
#         # for i in range(self.num_kernels):
#         self.conv1_out = nn.Conv1d(in_channels, out_channels, kernel_size=2 * self.num_kernels + 1,  padding=self.num_kernels)
#         if init_weight:
#             self._initialize_weights()
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#
#     def forward(self, x):
#         res = self.conv1_out(x)
#         return res

class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res

class Inception_Block_V2(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels // 2):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=[1, 2 * i + 3], padding=[0, i + 1]))
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=[2 * i + 3, 1], padding=[i + 1, 0]))
        kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=1))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels + 1):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res
