import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
from vat import VATLoss
import data_utils
from torch.autograd import Variable
import utils
from torch.utils.data import DataLoader
import torchvision.models as models
import time


class Resnext_50(nn.Module):
    def __init__(self, num_class=10):
        super(Resnext_50, self).__init__()
        self.num_class = num_class
        self.feature = nn.Sequential(
            *list(models.resnext50_32x4d(pretrained=True).children())[:-1])
        self.fc = nn.Linear(2048 * 1 * 1, self.num_class)

    def forward(self, x):
        x = self.feature(x)
        batchsz = x.size(0)
        x = x.view(batchsz, 2048)
        x = self.fc(x)
        return x


def origin_test(model, test_dataloader):
    model.eval()
    acc_history = []
    total_correct = 0
    total_num = 0
    device = torch.device('cuda')
    for (img, label) in tqdm(test_dataloader):
        img = Variable(img)
        label = Variable(label)

        img = img.to(device)
        label = label.to(device)

        # [b, 10]
        logits = model(img)
        # [b]
        pred = logits.argmax(dim=1)
        # [b] vs [b] => scalar tensor
        correct = torch.eq(pred, label).float().sum().item()
        total_correct += correct
        total_num += img.size(0)
        # print(correct)
        acc = total_correct / total_num
        acc_history.append(acc)
    return torch.mean(torch.Tensor(acc_history)).item()
    print('test_acc: {:.4}'.format(torch.mean(torch.Tensor(acc_history))))
    print('epoch: {}|time: {:.4f}'.format(epoch, time.time() - start_))
    torch.save(model.state_dict(), 'checkpoints/model.pth')


def disturb_test(model, test_dataloader):
    acc_history = []
    total_correct = 0
    total_num = 0
    model.eval()
    for (data, target) in tqdm(test_dataloader):
        data = Variable(data)
        target = Variable(target)
        data, target = data.to(device), target.to(device)
        vat_loss = VATLoss(xi=10.0, eps=1.0, ip=1)

        # LDS should be calculated before the forward for cross entropy
        lds, d = vat_loss(model, data)
        data = torch.add(data,d)
        output = model(data)
        pred = output.argmax(dim=1)
        # [b] vs [b] => scalar tensor
        correct = torch.eq(pred, target).float().sum().item()
        total_correct += correct
        total_num += data.size(0)
        # print(correct)
        acc = total_correct / total_num
        acc_history.append(acc)
    return torch.mean(torch.Tensor(acc_history)).item()
        

if __name__ == '__main__':

    cifar = datasets.CIFAR10('/home/lywh/Documents/data', True, transform=transforms.Compose([
        transforms.Resize([32, 32]),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ]), download=False)
    cifar_trainloader = DataLoader(
        cifar, batch_size=16, shuffle=True, drop_last=True)

    cifar_test = datasets.CIFAR10('/home/lywh/Documents/data', False, transform=transforms.Compose([
        transforms.Resize([32, 32]),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.ToTensor()
    ]), download=False)
    cifar_testloader = DataLoader(
        cifar, batch_size=16, shuffle=True, drop_last=True)

    device = torch.device('cuda')
    model = Resnext_50().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # train(args, model, device, data_iterators, optimizer)
    # test(model, device, data_iterators)
    cross_entropy = nn.CrossEntropyLoss().to(device)

    best_disturb_acc = 1.0

    for epoch in range(1):

        loss_history = []
        model.train()
        start_ = time.time()
        d_values = []
        #for id,content in enumerate(cifar_trainloader):
        for id,(content,q) in tqdm(cifar_trainloader):
            print(id,type(content),len(content))

