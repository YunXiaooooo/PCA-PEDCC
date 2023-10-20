import torch
import torchvision
import os
from torch import nn
from tool.pedccBasePretrain import generatePedcc_adamw, generatePedcc_cb


def load_barlowtwins(config):
    if os.path.exists(config.barlowtwins_path):
        model = torch.load(config.barlowtwins_path)  # 对应torch.save(model, mymodel.pth)， 保存有模型结构
    else:
        model = torch.hub.load('facebookresearch/barlowtwins:main', 'resnet50')
        torch.save(model, config.barlowtwins_path)

    model.fc = nn.Identity()
    return model


class MisakaNet(torch.nn.Module):
    def __init__(self, config):
        super(MisakaNet, self).__init__()
        self.barlowtwins = load_barlowtwins(config)

        self.bias = nn.Parameter(config.bias.float(), requires_grad=False)
        self.coeff = nn.Parameter(config.coeff.float(), requires_grad=False)
        # targetTmp = generatePedcc_cb(config.targetCenters.float(), 300)
        targetTmp = generatePedcc_adamw(config.targetCenters.float(), 1000)
        self.target = nn.Parameter(targetTmp.t().float(), requires_grad=False)

    def forward(self, x):
        xout = self.barlowtwins(x)
        xout = xout-self.bias
        xout = torch.mm(xout, self.coeff)

        xoutNorm = torch.linalg.norm(xout, dim=1, keepdim=True)
        xout = torch.div(xout, xoutNorm+0.0001)

        xout = torch.mm(xout, self.target)

        return xout


def model_state_load(model_path, model, optimizer):
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        valid_best_acc = checkpoint['valid_best_acc']
        print('load epoch {} success! '.format(start_epoch))
    else:
        start_epoch = 0
        valid_best_acc = 0
        print('No model is saved in \'{}\''.format(model_path))
    return start_epoch, valid_best_acc


def model_state_save(model, model_path, optimizer, epoch, valid_best_acc):
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'valid_best_acc':valid_best_acc}
    torch.save(state, model_path)
    print('model is saved when epoch=={}'.format(epoch))