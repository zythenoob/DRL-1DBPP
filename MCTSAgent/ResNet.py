import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam, SGD, RMSprop
from torch.optim.lr_scheduler import StepLR
from MCTSAgent.State import max_bin_opening
from MCTSAgent.main import search_device, update_device
from math import log, e, isnan
from PPOAgent.model import PointerNet, masked_softmax


def entropy(probs):
    probs = probs.cpu().detach().numpy()
    ent = 0
    for p in probs:
        for i in p:
            if i > 0:
                ent -= i * log(i, e)
    return ent / len(probs)


class EntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        # return -torch.mean(torch.sum(y * torch.log(torch.masked_fill(x, ~m, 1e-5)), dim=1))
        return -torch.mean(torch.sum(y * torch.log(x), dim=1))


class ResNet(nn.Module):
    def __init__(self, lr=0.001):
        super(ResNet, self).__init__()

        # self.in_channels = 64
        #
        # self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = nn.GroupNorm(32, 64)
        #
        # self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        # self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        # # self.layer3 = self._make_layer(BasicBlock, 256, 4, stride=2)
        #
        # self.pool = nn.AvgPool2d(3, stride=1)
        #
        # self.conv_p = nn.Conv2d(128, 32, kernel_size=1, stride=1)
        # self.bn_p = nn.GroupNorm(32, 32)
        # self.conv_v = nn.Conv2d(128, 32, kernel_size=1, stride=1)
        # self.bn_v = nn.GroupNorm(32, 32)
        #
        # # dense
        # self.dense_v1 = nn.Linear(768, 256)
        # self.dense_v2 = nn.Linear(256, 1)
        # self.dense_p = nn.Linear(768, max_bin_opening)

        # MLP
        self.layer1 = nn.Linear(160, 1024)
        self.layer2 = nn.Linear(1024, 1024)
        self.layer3 = nn.Linear(1024, 1024)
        self.layerv = nn.Linear(1024, 1)
        self.layer4 = nn.Linear(1024, max_bin_opening)

        # loss
        # self.loss_v = nn.MSELoss()
        self.loss_p = EntropyLoss()
        # optimizer
        # self.optimizer = RMSprop(self.parameters(), lr=lr, alpha=0.99, weight_decay=0.001)
        self.optimizer = Adam(self.parameters(), lr=lr)
        # self.optimizer = SGD(self.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
        self.scheduler = StepLR(self.optimizer, 500, gamma=0.99, last_epoch=-1)

        self.iter = 0

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, planes, stride))
            self.in_channels = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, m):
        # x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        # x = self.layer1(x)
        # x = self.layer2(x)
        # # # x = self.layer3(x)
        # x = self.pool(x)
        #
        # p = F.relu(self.bn_p(self.conv_p(x)), inplace=True)
        # p = torch.flatten(p, start_dim=1)
        # print(p.shape)
        # v = F.relu(self.bn_v(self.conv_v(x)), inplace=True)
        # v = torch.flatten(v, start_dim=1)
        #
        # value = torch.tanh(self.dense_v2(F.relu(self.dense_v1(v), inplace=True)))
        #
        # policy = torch.softmax(self.dense_p(p), dim=-1)
        # policy = masked_softmax(self.dense_p(p), mask=m.unsqueeze(0), dim=-1, log=False)

        x = torch.flatten(x, start_dim=1)
        x = torch.tanh(self.layer1(x))
        x = torch.tanh(self.layer2(x))
        x = torch.tanh(self.layer3(x))
        # value = torch.tanh(self.layerv(x))
        policy = torch.softmax(self.layer4(x), dim=-1)

        return policy

    def predict(self, state, m, device):
        mask = torch.tensor(m, device=device).bool()
        if len(np.array(state).shape) == 3:
            x = torch.tensor(state, device=device).float()
        else:
            x = torch.tensor(state, device=device).float().unsqueeze(0)

        return self(x, mask)

    def update(self, s, m, _v, _p, writer=None):
        p = self.predict(s, m, update_device)
        _v = torch.tensor(_v, device=update_device, dtype=torch.float).unsqueeze(-1)
        _p = torch.tensor(_p, device=update_device, dtype=torch.float)

        # value loss
        # v_loss = self.loss_v(v, _v)
        # policy loss
        loss = self.loss_p(p, _p)

        # loss = v_loss + p_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0, norm_type=2)
        self.optimizer.step()
        self.scheduler.step()

        # write summary
        if writer is not None:
            writer.add_scalar('Entropy/Policy Entropy', entropy(p), self.iter)
            writer.add_scalar('Loss/All', loss, self.iter)
            # writer.add_scalar('Loss/V', v_loss, self.iter)
            # writer.add_scalar('Loss/P', p_loss, self.iter)
            self.iter += 1


class BasicBlock(nn.Module):
    # 2 layer residual block
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.GroupNorm(32, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.GroupNorm(32, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            # Projection Shortcut
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride),
                nn.GroupNorm(32, self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out


class Bottleneck(nn.Module):
    # 3 layer residual block
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1)
        self.bn1 = nn.GroupNorm(32, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.GroupNorm(32, planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1)
        self.bn3 = nn.GroupNorm(32, self.expansion * planes)

        self.shortcut = nn.Sequential().cuda()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride),
                nn.GroupNorm(32, self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out
