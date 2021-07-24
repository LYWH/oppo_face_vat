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
        #print(x.shape)
        x = x.view(batchsz, 2048)
        x = self.fc(x)
        return x


class Resnet_18(nn.Module):
    def __init__(self, num_class=10):
        super(Resnet_18, self).__init__()
        self.num_class = num_class
        self.feature = nn.Sequential(
            *list(models.resnet18(pretrained=True).children())[:-1])
        self.fc = nn.Linear(512 * 1 * 1, self.num_class)

    def forward(self, x):
        x = self.feature(x)
        batchsz = x.size(0)
        #print(x.shape)
        x = x.view(batchsz, 512)
        x = self.fc(x)
        return x



def origin_test(model, test_dataloader):
    model.eval()
    acc_history = []
    total_correct = 0
    total_num = 0
    device = torch.device('cuda')
    for (img, label) in tqdm(test_dataloader,desc="正常测试中"):
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


def disturb_test(model, vat,test_dataloader):
    acc_history = []
    total_correct = 0
    total_num = 0
    model.eval()
    vat.eval()
    for (data, target) in tqdm(test_dataloader,desc="扰动测试中"):
        data = Variable(data)
        target = Variable(target)
        data, target = data.to(device), target.to(device)
        vat_loss = VATLoss(xi=10.0, eps=1.0, ip=1)

        # LDS should be calculated before the forward for cross entropy
        lds, d = vat(model, data)
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
        cifar, batch_size=64, shuffle=True, drop_last=True)

    cifar_test = datasets.CIFAR10('/home/lywh/Documents/data', False, transform=transforms.Compose([
        transforms.Resize([32, 32]),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        transforms.ToTensor()
    ]), download=False)
    cifar_testloader = DataLoader(
        cifar, batch_size=64, shuffle=True, drop_last=True)

    device = torch.device('cuda')
    model = Resnet_18().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    vat_loss = VATLoss(xi=10.0, eps=1.0, ip=1)

    # train(args, model, device, data_iterators, optimizer)
    # test(model, device, data_iterators)
    cross_entropy = nn.CrossEntropyLoss().to(device)

    ori_disturb_acc = -1
    disturb = 1

    for epoch in range(50):

        loss_history = []
        model.train()
        start_ = time.time()
        d_values = []
        pbar = tqdm(cifar_trainloader)
        vat_loss.train()
        for batch_idx, (data, target) in enumerate(pbar):
        #for batch_idx, data, target in tqdm(cifar_trainloader):
            data = Variable(data)
            target = Variable(target)
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            

            # LDS should be calculated before the forward for cross entropy
            lds, d = vat_loss(model, data)
            d_values.append((epoch, d.cpu().numpy()))
            output = model(data)
            loss = cross_entropy(output, target) + lds

            loss.backward()
            optimizer.step()
            loss_history.append(loss)
            pbar.set_description("loss:{}".format(sum(loss_history)/(len(loss_history)+0.0001)))

        test_acc = origin_test(model=model, test_dataloader=cifar_testloader)
        disturb_acc = disturb_test(model=model,vat=vat_loss,test_dataloader=cifar_testloader)
        print('test_acc: {:.6},disturb_acc:{:.6f}'.format(test_acc,disturb_acc))
        print('epoch: {}|time: {:.4f}'.format(epoch, time.time() - start_))
        if ori_disturb_acc < test_acc - disturb_acc:
            ori_disturb_acc = test_acc - disturb_acc
            print("扰动效果:{}".format(test_acc-disturb_acc))
            torch.save(model.state_dict(), 'checkpoints/model.pth')
        
