import csv
from PIL import Image
import numpy as np
from config import cfg
from utils import get_radiomics
import os

def record_data(fl,pl,pad=0):
    for fs,p in zip(fl,pl):
      data=[]
      target = []
      names=''
      with open(fs,'r') as f:
        r = csv.DictReader(f)
        data = []
        target = []
        row = [l for l in r]
        for line in row:
            img_p = os.path.join(p,line['Image'])
            mask_p = os.path.join(p,line['Image'][:-5]+'t.png')
            img = np.array(Image.open(img_p).convert('L'))
            mask = np.array(Image.open(mask_p).convert('L'))
            #shape = img.shape
            #if pad is not None:
             # img,mask = pad_crop(img,mask,pad)
            #print(shape,img.shape)
            names,fts = get_radiomics(img,mask)
            data.append([ft+0 for ft in fts])
            target.append(eval(line['Label']))
      with open(fs,'w') as f:
        w = csv.writer(f)
        w.writerow(['Image','Label']+names)
        for i,line in enumerate(row):
            w.writerow(list(line.values())[:2]+data[i])
    return {'data':data,'target':target,'names':names}

record_data([cfg['busi'].label,cfg['db'].label],[cfg['busi'].path,cfg['db'].path])

