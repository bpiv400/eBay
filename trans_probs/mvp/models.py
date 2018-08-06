import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import argparse
import pickle
import math
import sys
import os

# complex expecation


def get_model_class(exp_name):
    '''
    Description: Uses experiment name to grab the associated model
    from the models module and aliases it as net
    Input: string giving experiment name
    Output: class of model to be trained
    '''
    if 'cross' in exp_name:
        if 'simp' in exp_name:
            print('Model: Cross Simp')
            net = Cross_simp
        elif 'bet' in exp_name:
            print('Model: Cross between')
            net = Cross_simp
        else:
            net = Cross_comp
            print('Model cross comp')
    else:
        if 'simp' in exp_name:
            print('model exp simp')
            net = Exp_simp
        elif 'bet' in exp_name:
            print('model exp between')
            net = Exp_simp
        else:
            print('model exp comp')
            net = Exp_comp
    return net


class Exp_comp(nn.Module):
    def __init__(self, num_feat, num_units, num_classes, classes):
        super(Exp_comp, self).__init__()
        # print('feats %d' % num_feat)
        # print('units %d' % num_units)
        # print('num_classes %d' % num_classes)
        self.fc1 = nn.Linear(num_feat, num_units)
        self.fc2 = nn.Linear(num_units, num_units)
        self.fc3 = nn.Linear(num_units, num_classes)
        self.expect = classes
        self.expect.requires_grad_(False)
        self.expect = self.expect.view(-1, 1)

    def forward(self, x):
        x = self.fc1(x)
        # print(x.size())
        x = F.relu(x)
        x = self.fc2(x)
        # print(x.size())
        x = F.relu(x)
        x = self.fc3(x)
        # print(x.size())
        x = F.softmax(x, dim=1)
        # print(x)
        # print(x.size())
        # print(self.expect.size())
        x = torch.mm(x, self.expect)
        return x

# simple expectation model


class Exp_simp(nn.Module):
    def __init__(self, num_feat, num_units, num_classes, classes):
        super(Exp_simp, self).__init__()
        # print('feats %d' % num_feat)
        # print('units %d' % num_units)
        # print('num_classes %d' % num_classes)
        self.fc1 = nn.Linear(num_feat, num_units)
        self.fc2 = nn.Linear(num_units, num_classes)
        self.expect = classes
        self.expect.requires_grad_(False)
        self.expect = self.expect.view(-1, 1)

    def forward(self, x):
        x = self.fc1(x)
        # print(x.size())
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        # print(x)
        # print(x.size())
        # print(self.expect.size())
        x = torch.mm(x, self.expect)
        return x

# complex cross entropy model


class Cross_comp(nn.Module):
    def __init__(self, num_feat, num_units, num_classes):
        super(Cross_comp, self).__init__()
        # print('feats %d' % num_feat)
        # print('units %d' % num_units)
        # print('num_classes %d' % num_classes)
        self.fc1 = nn.Linear(num_feat, num_units)
        self.fc2 = nn.Linear(num_units, num_units)
        self.fc3 = nn.Linear(num_units, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        # print(x.size())
        x = F.relu(x)
        x = self.fc2(x)

        x = F.relu(x)
        x = self.fc3(x)
        # print(x)
        # print(x.size())
        # print(self.expect.size())
        return x


# simple cross entropy model
class Cross_simp(nn.Module):
    def __init__(self, num_feat, num_units, num_classes):
        super(Cross_simp, self).__init__()
        # print('feats %d' % num_feat)
        # print('units %d' % num_units)
        # print('num_classes %d' % num_classes)
        self.fc1 = nn.Linear(num_feat, num_units)
        self.fc2 = nn.Linear(num_units, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        # print(x.size())
        x = F.relu(x)
        x = self.fc2(x)
        # print(x)
        # print(x.size())
        # print(self.expect.size())
        return x
