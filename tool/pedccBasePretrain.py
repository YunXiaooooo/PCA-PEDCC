import torch
from torch import nn, optim
import numpy as np


class pedccModel(torch.nn.Module):
    def __init__(self, centers):
        super(pedccModel, self).__init__()

        targetMeans = centers

        targetMeans = targetMeans.cuda()
        targetMeans = targetMeans.to(torch.float)
        targetMeansNorm = torch.linalg.norm(targetMeans, dim=1, keepdim=True)

        self.M = torch.nn.parameter.Parameter(torch.div(targetMeans, targetMeansNorm), requires_grad=True)
        # M = torch.div(targetMeans, targetMeansNorm)
        # cosine = torch.mm(M, M.t())
        # theta = torch.acos(cosine)
        # theta = theta * 180 / 3.14156

    def forward(self, x):
        Mout = torch.mul(x, self.M)

        return Mout


def lossFun(u, target):
    u = u.cuda()
    target = target.cuda()
    classNum = u.shape[0]
    unorm = torch.linalg.norm(u, dim=1, keepdim=True)
    u = torch.div(u, unorm)

    d = 1-torch.mm(u, u.t())

    loss = pow(d-target, 2)
    diag = torch.diag(loss)
    diag = torch.diag_embed(diag)
    loss = loss-diag

    loss = torch.sum(loss)/(classNum*(classNum-1))

    return loss

def generatePedcc_adamw(centers, epochNum):
    with torch.no_grad():
        x = torch.ones(1, centers.shape[1]).cuda()

    classNum = centers.shape[0]
    target = torch.ones(classNum, classNum)*(classNum/(classNum-1))

    model = pedccModel(centers)
    model = model.cuda()
    optimizer = optim.AdamW(model.parameters(), lr=3e-2, weight_decay=0.001)

    for epoch in range(0, epochNum):
        model.train()
        Mout = model(x)
        loss = lossFun(Mout, target)

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            norm = torch.linalg.norm(Mout, dim=1, keepdim=True)
            M = torch.div(Mout, norm)
            cosine = torch.mm(M, M.t())
            theta = torch.acos(cosine)
            theta = theta * 180 / 3.14156
            theta = torch.nan_to_num(theta, nan=0.0)

            I = torch.eye(theta.shape[0]).cuda()
            theta = torch.mul(theta, 1-I)
            # m_theta = torch.sum(theta)/(theta.shape[0]*(theta.shape[0]-1))
            # print("m_theta", m_theta)
            # print("theta", theta)
            print("e=%d,loss=%.8f"%(epoch, loss.item()))

    print("theta", theta)

    return M.detach()






