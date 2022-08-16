import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from torchvision.models import resnet34#,ResNet34_Weights
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import cfg
from utils import train_transform,test_transform,pad_crop,get_data,sp,focal_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score,roc_auc_score,recall_score,f1_score,precision_score
import argparse
from PIL import Image
import numpy as np

#weights=ResNet34_Weights..DEFAULT
class res34(nn.Module):
    def __init__(self):
        super(res34,self).__init__()
        #self.backbone = resnet34(weights = weights)
        self.backbone = nn.Sequential(*list(resnet34(pretrained = True).children())[:-1])
        if cfg.features>0:
            self.fl = nn.Sequential(
                      nn.Identity(),
                      nn.BatchNorm1d(cfg.features)
                      )
        self.fc = nn.Sequential(
                  nn.Linear(512+cfg.features, 256),
                  nn.BatchNorm1d(256),
                  nn.ReLU(),
                  nn.Dropout(0.2),
                  nn.Linear(256, 1),
                  )
    def forward(self,x,rad_ft):
        res_ft = self.backbone(x)
        if cfg.features>0:
            rad_ft = self.fl(rad_ft)
        ft = torch.cat([torch.flatten(res_ft,1),rad_ft],1)
        out = self.fc(ft)
        return out

class Data(Dataset):

    def __init__ (self, x, y, img_path, transform, padding=None):

        self.listImagePaths = []
        self.listImageFeatures = []
        self.listImageLabels = []
        self.transform = transform
        self.pad = padding

        for img_ft,img_label in zip(x,y):
            self.listImagePaths.append(os.path.join(img_path,img_ft[0]))
            self.listImageLabels.append(img_label)
            self.listImageFeatures.append([eval(r) for r in img_ft[1:1+cfg.features]])
    
    def __getitem__(self, index):

        imagePath = self.listImagePaths[index]
        imageData = Image.open(imagePath).convert('RGB')
        img = np.array(imageData)
        maskData = Image.open(imagePath[:-5]+'t.png').convert('L')
        mask = np.array(maskData)
        if self.pad is not None:
          img,mask = pad_crop(img,mask,self.pad)
        imageLabel= torch.FloatTensor([self.listImageLabels[index]])
        imageFeature = torch.FloatTensor(self.listImageFeatures[index])
        imageData = self.transform(image=img)['image']
        return  imageData,imageFeature,imageLabel

    def __len__(self):
        return len(self.listImageLabels)


def train_test(x_train,y_train,x_test,y_test):

    net = res34()
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    multisteplr = optim.lr_scheduler.MultiStepLR(optimizer,[20,25],gamma=0.5)

    train_data = Data(x_train,y_train,cfg[args.data].path,train_transform,padding = args.pad)
    train_loader = DataLoader(train_data, batch_size = args.bs, shuffle = True, num_workers = 10)
    if args.mode =='transfer':
        valid_data = Data(x_test,y_test,cfg['db'].path,test_transform,padding = args.pad)
    else:
        valid_data = Data(x_test,y_test,cfg[args.data].path,test_transform,padding = args.pad)
    valid_loader = DataLoader(valid_data, batch_size = args.bs, num_workers = 10)
    min_loss = 100 
    for epoch in range(args.epoch):
        #train
        net.train()
        for inputs,features, targets in train_loader:
            #inputs, targets = inputs.to(device), targets.to(device)
            inputs, features, targets = inputs.to(device), features.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs,features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        #test
        net.eval()
        test_loss = 0
        gt = torch.FloatTensor()
        out = torch.FloatTensor()
        gt,out = gt.to(device),out.to(device)
        with torch.no_grad():
            for inputs, features,targets in valid_loader:
                #inputs, targets = inputs.to(device), targets.to(device)
                inputs, features, targets = inputs.to(device), features.to(device), targets.to(device)
                outputs = net(inputs,features)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                outputs = torch.sigmoid(outputs)
                out = torch.cat((out,outputs),0)
                gt = torch.cat((gt,targets),0)

        predicted = out.round()
        gt,out,pred = gt.cpu().data.numpy(),out.cpu().data.numpy(),predicted.cpu().data.numpy()
        acc,auc,se,spe,pr,f1=accuracy_score(gt,pred),roc_auc_score(gt,out),recall_score(gt,pred),sp(gt,pred),precision_score(gt,pred,zero_division=0),f1_score(gt,pred)
        test_loss/=len(valid_loader)
        #multisteplr.step()
        # Save checkpoint.
        if test_loss < min_loss:
            state = {
                'net': net.state_dict(),
                'auc': auc,
                'loss':test_loss,
            }
            print('epoch {}, loss decreased to {}, acc:{}, auc:{}, se:{}, sp:{}, pr:{}, f1:{}'.format(epoch,test_loss,acc,auc,se,spe,pr,f1))
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/res34.t7')
            min_loss = test_loss
            val_acc,val_auc,val_se,val_sp,val_pr,val_f1 = acc,auc,se,spe,pr,f1
    return val_acc,val_auc,val_se,val_sp,val_pr,val_f1

parser = argparse.ArgumentParser(description='feature_select')
parser.add_argument('--mode', '-m', default='cross', type=str, choices = ['corss','transfer'], help='cross validation or test transfer performance')
parser.add_argument('--data', '-d', default='busi', type=str, choices = ['syu','busi','onl','db'], help='choose datasets')
parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
parser.add_argument('--epoch', default=30, type=int, help='epoch')
parser.add_argument('--bs', default=20, type=int, help='batch_size')
parser.add_argument('--ft', default=8, type=int, help='features')
parser.add_argument('--pad','-p', default=10, type=int, help='padding')
parser.add_argument('--loss', '-l', default='bce', type=str, choices = ['bce','focal'], help='loss function')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if args.loss =='focal':
    criterion = focal_loss
else:
    criterion = nn.BCEWithLogitsLoss()
if args.pad < 0: args.pad = None
cfg.features = args.ft

if args.mode =='cross':
    x,y = get_data(cfg[args.data].label,p=True)
    k = 5
    skf = StratifiedKFold(n_splits=k)
    test_acc,test_auc,test_se,test_sp,test_pr,test_f1 = [],[],[],[],[],[]

    for train_index,test_index in skf.split(x,y):
        x_train,x_test = x[train_index],x[test_index]
        y_train,y_test = y[train_index],y[test_index]
        val_acc,val_auc,val_se,val_sp,val_pr,val_f1 = train_test(x_train,y_train,x_test,y_test)
        test_acc.append(val_acc)
        test_auc.append(val_auc)
        test_se.append(val_se)
        test_sp.append(val_sp)
        test_pr.append(val_pr)
        test_f1.append(val_f1)
    print('Res34 padding{} cross results: acc:{}, auc:{}, se:{}, sp:{}, pr:{}, f1:{}'.format(args.pad,sum(test_acc)/k,sum(test_auc)/k,sum(test_se)/k,sum(test_sp)/k,sum(test_pr)/k,sum(test_f1)/k))
else:
    x_train,y_train = get_data(cfg['busi'].label,p=True)
    x_test,y_test = get_data(cfg['db'].label,p=True)
    val_acc,val_auc,val_se,val_sp,val_pr,val_f1 = train_test(x_train,y_train,x_test,y_test)
    print('Res34 padding{} transfer results on dataset b: acc:{}, auc:{}, se:{}, sp:{}, pr:{}, f1:{}'.format(args.pad,val_acc,val_auc,val_se,val_sp,val_pr,val_f1))

