import numpy as np
from PIL import Image
import SimpleITK as sitk
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as transforms
from radiomics import featureextractor,setVerbosity
from config import cfg
import torch
import csv
import cv2
import os

train_transform = A.Compose(
 [
  A.Resize(232,232),
  A.HorizontalFlip(p=0.5),
  A.ShiftScaleRotate(p=0.4, border_mode=0, shift_limit=0.04, scale_limit=0.03, rotate_limit = 10),
  A.RandomCrop(height=224,width=224),
  A.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
  ToTensorV2(),
 ]
)
test_transform = A.Compose(
 [
  A.Resize(224,224),
  #A.CenterCrop(height=224,width=224),
  A.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
  ToTensorV2(),
 ]
)

def get_bbox(mask):
    cnts, hierarchy = cv2.findContours(mask,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    area = [cv2.contourArea(cnt) for cnt in cnts]
    if area == []: return 0,0,mask.shape[1],mask.shape[0]
    max_idx = np.argmax(area)
    return cv2.boundingRect(cnts[max_idx])

def pad_crop(img,mask,pad):
    img = cv2.copyMakeBorder(img,pad,pad,pad,pad,cv2.BORDER_CONSTANT,value=0)
    x,y,w,h = get_bbox(mask)
    img = img[y:y+h+2*pad,x:x+w+2*pad]
    mask = cv2.copyMakeBorder(mask,pad,pad,pad,pad,cv2.BORDER_CONSTANT,value=0)
    mask = mask[y:y+h+2*pad,x:x+w+2*pad]
    return img,mask

def get_radiomics(img,mask,mode='min'):
    setVerbosity(50)
    image2d = sitk.GetImageFromArray(img)
    mask2d = sitk.GetImageFromArray(mask)
    if mode =='all':
        params = 'data/breast.yaml'
    else:
        params = 'data/fts.yaml'
    extractor = featureextractor.RadiomicsFeatureExtractor(params)
    result = extractor.execute(image2d, mask2d)
    return list(result.keys())[22:],list(result.values())[22:]

def get_data(label,p = False):
    x = []
    y = []
    with open(label,'r') as f:
        r = csv.DictReader(f)
        labels = r.fieldnames
        x_l = labels[2:]
        y_l = labels[1]
        for row in r:
            if p:
                x.append([row['Image']]+[eval(row[i]) for i in x_l])
            else:
                x.append([eval(row[i]) for i in x_l])
            y.append(eval(row[y_l]))
    x = np.array(x)
    y = np.array(y)
    return x,y

def sp(y_true,y_pred):
    y_pred = np.round(y_pred)
    eps = 1e-5
    return ((1-y_true)*(1-y_pred)).sum() / ((1-y_true).sum()+eps)

def focal_loss(y, t, gamma = 2, alpha = None):

    pt = torch.clamp(torch.sigmoid(y),1e-5,1-1e-5)
    loss1 = -t * (1-pt).detach()**gamma * torch.log(pt)
    loss0 = -(1-t) * pt.detach()**gamma * torch.log(1-pt)
    #loss1 = -t * (1-pt)**gamma * torch.log(pt)
    #loss0 = -(1-t) * pt**gamma * torch.log(1-pt)
    if alpha is not None:
        loss = alpha * loss1 + (1-alpha)*loss0
    else:
        loss = loss1 + loss0
    return loss.mean()


