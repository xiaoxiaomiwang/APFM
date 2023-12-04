import torch

#########################################
#PPA(x) Activation Function Forward
#########################################

class PPA_ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        zero = torch.zeros_like(input_)

        x1 = torch.where(input_ <= -4, zero, zero)
        x2 = torch.where((input_ > -4) & (input_ <= -3),
                         -0.000499 * input_ ** 3 - 0.005899 * input_ ** 2 - 0.023199 * input_ - 0.030399, zero)
        x3 = torch.where((input_ > -3) & (input_ < -2),
                         -0.008299 * input_ ** 3 - 0.0777 * input_ ** 2 - 0.2434 * input_ - 0.2554, zero)
        x4 = torch.where((input_ >= -2) & (input_ < -1),
                         0.0151 * input_ ** 3 + 0.0231 * input_ ** 2 - 0.120999 * input_ - 0.226599, zero)
        x5 = torch.where((input_ >= -1) & (input_ < 0),
                         0.184699 * input_ ** 3 + 0.588 * input_ ** 2 + 0.5 * input_ - 0.0009, zero)
        x6 = torch.where((input_ >= 0) & (input_ <= 1),
                         -0.1848 * input_ ** 3 + 0.588099 * input_ ** 2 + 0.5 * input_ - 0.0009, zero)
        x7 = torch.where((input_ > 1) & (input_ <= 2),
                         -0.0152 * input_ ** 3 + 0.0236 * input_ ** 2 + 1.120199 * input_ - 0.226199, zero)
        x8 = torch.where((input_ > 2) & (input_ <= 3),
                         0.0083 * input_ ** 3 - 0.0777 * input_ ** 2 + 1.2434 * input_ - 0.2554, zero)
        x9 = torch.where((input_ > 3) & (input_ < 4),
                         0.0005 * input_ ** 3 - 0.005899 * input_ ** 2 + 1.0232 * input_ - 0.0304, zero)
        x10 = torch.where(input_ >= 4, input_, zero)

        output = x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10
        return output

#########################################
# PPA(x) Activation Function Backward
#########################################

    @staticmethod
    def backward(ctx, grad_output):
        input_, = ctx.saved_tensors
        grad_input = grad_output.clone()
        zero = torch.zeros_like(input_)

        grad_x1 = torch.where(input_ <= -4, zero, zero)
        grad_x2 = torch.where((input_ > -4) & (input_ <= -3), 3 * (-0.000499) * input_ ** 2 - 0.005899 * 2 * input_ - 0.023199, zero)
        grad_x3 = torch.where((input_ > -3) & (input_ < -2), 3 * (-0.008299) * input_ ** 2 + 0.0777 * 2 * input_ - 0.2434, zero)
        grad_x4 = torch.where((input_ >= -2) & (input_ < -1), 3 * 0.0151 * input_ ** 2 + 0.0231 * 2 * input_ - 0.120999, zero)
        grad_x5 = torch.where((input_ >= -1) & (input_ < 0), 3 * 0.184699 * input_ ** 2 + 0.588099 * 2 * input_ + 0.5, zero)
        grad_x6 = torch.where((input_ >= 0) & (input_ <= 1), 3 * (-0.1848) * input_ ** 2 + 0.5881 * 2 * input_ + 0.5, zero)
        grad_x7 = torch.where((input_ > 1) & (input_ <= 2), 3 * (-0.0152) * input_ ** 2 + 0.0236 * 2 * input_ + 1.120199, zero)
        grad_x8 = torch.where((input_ > 2) & (input_ <= 3), 3 * 0.0083 * input_ ** 2 - 0.0777 * 2 * input_ + 1.2434, zero)
        grad_x9 = torch.where((input_ > 3) & (input_ < 4), 3 * 0.0005 * input_ ** 2 - 0.005899 * 2 * input_ + 1.0232, zero)
        grad_x10 = torch.where(input_ >= 4, 1, zero)

        grad_input = grad_input * (grad_x1 + grad_x2 + grad_x3 + grad_x4 + grad_x5 + grad_x6 + grad_x7 + grad_x8 + grad_x9 + grad_x10)
        return grad_input

