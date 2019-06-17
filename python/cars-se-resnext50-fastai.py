#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install console_progressbar')


# In[2]:


# This preprocessing portion of the code is provided by foamliu on his github repo
# https://github.com/foamliu/Car-Recognition/blob/master/pre-process.py

import tarfile
import scipy.io
import numpy as np
import os
import cv2 as cv
import shutil
import random
from console_progressbar import ProgressBar


# In[3]:


def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


# In[4]:


def save_train_data(fnames, labels, bboxes):
    src_folder ='../input/stanford-cars-dataset/cars_train/cars_train/'
    num_samples = len(fnames)

    train_split = 0.8
    num_train = int(round(num_samples * train_split))
    train_indexes = random.sample(range(num_samples), num_train)

    pb = ProgressBar(total=100, prefix='Save train data', suffix='', decimals=3, length=50, fill='=')

    for i in range(num_samples):
        fname = fnames[i]
        label = labels[i]
        (x1, y1, x2, y2) = bboxes[i]

        src_path = os.path.join(src_folder, fname)
        src_image = cv.imread(src_path)
        height, width = src_image.shape[:2]
        # margins of 16 pixels
        margin = 16
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(x2 + margin, width)
        y2 = min(y2 + margin, height)
        # print("{} -> {}".format(fname, label))
        pb.print_progress_bar((i + 1) * 100 / num_samples)

        if i in train_indexes:
            dst_folder = '/kaggle/working/data/train/'
        else:
            dst_folder = '/kaggle/working/data/valid/'

        dst_path = os.path.join(dst_folder, label)
        if not os.path.exists(dst_path):
            os.makedirs(dst_path)
        dst_path = os.path.join(dst_path, fname)

        crop_image = src_image[y1:y2, x1:x2]
        dst_img = cv.resize(src=crop_image, dsize=(img_height, img_width))
        cv.imwrite(dst_path, dst_img)


# In[5]:


def save_test_data(fnames, bboxes):
    src_folder = '../input/stanford-cars-dataset/cars_test/cars_test/'
    dst_folder = '/kaggle/working/data/test/'
    num_samples = len(fnames)

    pb = ProgressBar(total=100, prefix='Save test data', suffix='', decimals=3, length=50, fill='=')

    for i in range(num_samples):
        fname = fnames[i]
        (x1, y1, x2, y2) = bboxes[i]
        src_path = os.path.join(src_folder, fname)
        src_image = cv.imread(src_path)
        height, width = src_image.shape[:2]
        # margins of 16 pixels
        margin = 16
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(x2 + margin, width)
        y2 = min(y2 + margin, height)
        # print(fname)
        pb.print_progress_bar((i + 1) * 100 / num_samples)

        dst_path = os.path.join(dst_folder, fname)
        crop_image = src_image[y1:y2, x1:x2]
        dst_img = cv.resize(src=crop_image, dsize=(img_height, img_width))
        cv.imwrite(dst_path, dst_img)


# In[6]:


def process_train_data():
    print("Processing train data...")
    cars_annos = scipy.io.loadmat('../input/cars-devkit/cars_train_annos.mat')
    annotations = cars_annos['annotations']
    annotations = np.transpose(annotations)

    fnames = []
    class_ids = []
    bboxes = []
    labels = []

    for annotation in annotations:
        bbox_x1 = annotation[0][0][0][0]
        bbox_y1 = annotation[0][1][0][0]
        bbox_x2 = annotation[0][2][0][0]
        bbox_y2 = annotation[0][3][0][0]
        class_id = annotation[0][4][0][0]
        labels.append('%04d' % (class_id,))
        fname = annotation[0][5][0]
        bboxes.append((bbox_x1, bbox_y1, bbox_x2, bbox_y2))
        class_ids.append(class_id)
        fnames.append(fname)

    labels_count = np.unique(class_ids).shape[0]
    print(np.unique(class_ids))
    print('The number of different cars is %d' % labels_count)

    save_train_data(fnames, labels, bboxes)


# In[7]:


def process_test_data():
    print("Processing test data...")
    cars_annos = scipy.io.loadmat('../input/cars-devkit/cars_test_annos.mat')
    annotations = cars_annos['annotations']
    annotations = np.transpose(annotations)

    fnames = []
    bboxes = []

    for annotation in annotations:
        bbox_x1 = annotation[0][0][0][0]
        bbox_y1 = annotation[0][1][0][0]
        bbox_x2 = annotation[0][2][0][0]
        bbox_y2 = annotation[0][3][0][0]
        fname = annotation[0][4][0]
        bboxes.append((bbox_x1, bbox_y1, bbox_x2, bbox_y2))
        fnames.append(fname)

    save_test_data(fnames, bboxes)


# In[8]:


img_width, img_height = 224, 224

cars_meta = scipy.io.loadmat('../input/cars-devkit/cars_meta.mat')
class_names = cars_meta['class_names']  # shape=(1, 196)
class_names = np.transpose(class_names)
print('class_names.shape: ' + str(class_names.shape))
print('Sample class_name: [{}]'.format(class_names[8][0][0]))

ensure_folder('/kaggle/working/data/train')
ensure_folder('/kaggle/working/data/valid')
ensure_folder('/kaggle/working/data/test')

process_train_data()
process_test_data()


# In[9]:


import torchvision
from fastai.vision import *
from fastai.metrics import error_rate
from fastai import *
import cv2 as cv
import numpy as np
import pandas as pd
import scipy.io as sio


# In[10]:


tfms = get_transforms(do_flip=True, flip_vert=False, max_lighting=0.1, max_zoom=1.05,
                      max_warp=0.,
                      xtra_tfms=[rand_crop(), rand_zoom(1, 1.5),
                                 symmetric_warp(magnitude=(-0.2, 0.2))])

data = ImageDataBunch.from_folder('data/','train','valid',
                                  ds_tfms=tfms
                                  ,size=128,bs=64).normalize(imagenet_stats)


# In[11]:


data.show_batch(rows=3, figsize=(12,9))


# In[12]:


# class names and number of classes
# print(data.classes)
len(data.classes),data.c


# In[13]:


get_ipython().system('pip install pretrainedmodels')
import pretrainedmodels


# In[14]:


from torch import nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1., gamma=2.):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets, **kwargs):
        CE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-CE_loss)
        F_loss = self.alpha * ((1-pt)**self.gamma) * CE_loss
        return F_loss.mean()


# In[15]:


def se_resnext50_32x4d(pretrained=False):
    pretrained = 'imagenet' if pretrained else None
    model = pretrainedmodels.se_resnext50_32x4d(pretrained=pretrained)
    return nn.Sequential(*list(model.children()))


# In[16]:


learn = cnn_learner(data, se_resnext50_32x4d, pretrained=True, cut=-2,
                    split_on=lambda m: (m[0][3], m[1]), 
                    metrics=[accuracy])
learn.loss_fn = FocalLoss()


# In[17]:


learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[18]:


learn.fit_one_cycle(32, max_lr=slice(2e-2), wd=1e-5)


# In[19]:


interp = ClassificationInterpretation.from_learner(learn)

losses,idxs = interp.top_losses()

len(data.valid_ds)==len(losses)==len(idxs)


# In[20]:


learn.save('SE_ResNext50_1');
learn.unfreeze();
learn = learn.clip_grad();


# In[21]:


learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[22]:


learn.load('SE_ResNext50_1');
learn.unfreeze();
learn = learn.clip_grad();


# In[23]:


lr = [3e-3/100, 3e-3/20, 3e-3/10]
learn.fit_one_cycle(36, lr, wd=1e-7)


# In[24]:


learn.save('s2_SeResNext50_2');


# # Size 224

# In[25]:


SZ = 224
cutout_frac = 0.20
p_cutout = 0.75
cutout_sz = round(SZ*cutout_frac)
cutout_tfm = cutout(n_holes=(1,1), length=(cutout_sz, cutout_sz), p=p_cutout)

tfms = get_transforms(do_flip=True, max_rotate=15, flip_vert=False, max_lighting=0.1,
                      max_zoom=1.05, max_warp=0.,
                      xtra_tfms=[rand_crop(), rand_zoom(1, 1.5),
                                 symmetric_warp(magnitude=(-0.2, 0.2)), cutout_tfm])


# In[26]:


data = ImageDataBunch.from_folder('data/','train','valid',
                                  ds_tfms=tfms
                                  ,size=224,bs=32).normalize(imagenet_stats)

learn.data = data
data.train_ds[0][0].shape


# In[27]:


learn.load('s2_SeResNext50_2');
learn.freeze();
learn = learn.clip_grad();


# In[28]:


learn.loss_func = FocalLoss()


# In[29]:


learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[30]:


learn.fit_one_cycle(24, slice(3e-3), wd=5e-6)


# In[31]:


learn.save('SeResNxt50_FL_3');
learn.load('SeResNxt50_FL_3');


# In[32]:


learn.unfreeze();
learn = learn.clip_grad();


# In[33]:


lr = [1e-3/200, 1e-3/20, 1e-3/10]
learn.fit_one_cycle(32, lr)


# In[34]:


learn.save('SeResNxt50_FL_4');
learn.load('SeResNxt50_FL_4');


# # Size 299

# In[35]:


SZ = 299
cutout_frac = 0.20
p_cutout = 0.75
cutout_sz = round(SZ*cutout_frac)
cutout_tfm = cutout(n_holes=(1,1), length=(cutout_sz, cutout_sz), p=p_cutout)


# In[36]:


tfms = get_transforms(do_flip=True, max_rotate=15, flip_vert=False, max_lighting=0.1,
                      max_zoom=1.05, max_warp=0.,
                      xtra_tfms=[rand_crop(),
                                 symmetric_warp(magnitude=(-0.2, 0.2)), cutout_tfm])


# In[37]:


data = ImageDataBunch.from_folder('data/','train','valid',
                                  ds_tfms=tfms
                                  ,size=SZ,bs=24).normalize(imagenet_stats)

learn.data = data


# In[38]:


learn.load('SeResNxt50_FL_4');
learn.freeze();
learn = learn.clip_grad();
learn.mixup();


# In[39]:


learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[40]:


learn.fit_one_cycle(32, slice(1e-2))


# In[41]:


learn.save('SeResNext50_mixup_6');


# In[42]:


learn.load('SeResNext50_mixup_6');


# In[43]:


learn.unfreeze();
learn = learn.clip_grad();
# learn.mixup();


# In[44]:


learn.lr_find()
learn.recorder.plot(suggestion=True)


# In[45]:


lr = [2e-5, 2e-4, 2e-3]
learn.fit_one_cycle(64, lr)


# In[46]:


learn.export('/kaggle/working/fastai_resnet.pkl');


# # Predicting on the test set

# In[47]:


labels = sio.loadmat('../input/cars-devkit/cars_test_annos_withlabels.mat')


# In[48]:


x = []
for i in range(8041):
    x.append(np.transpose(np.array(labels['annotations']['fname']))[i][0][0])


# In[49]:


df=pd.DataFrame(data=np.transpose(np.array(labels['annotations']['class'],dtype=np.int)),
                  index=x)

df.to_csv('/kaggle/working/data/test_labels.csv')


# In[50]:


learn = load_learner('/kaggle/working/','fastai_resnet.pkl', test= 
                     ImageList.from_csv('/kaggle/working/data','test_labels.csv',folder='test'))
preds,y = learn.TTA(ds_type=DatasetType.Test)


# In[51]:


pd.DataFrame(preds.cpu().numpy()).to_csv('raw_test_preds.csv',index=False)


# In[52]:


a=preds;a.shape
b=np.array(labels['annotations']['class'],dtype=np.int)-1;b.shape 
b = torch.from_numpy(b)


# In[53]:


acc=accuracy(a,b);acc


# In[54]:


labelled_preds = torch.argmax(preds,1).cpu().numpy()
out = open('result.txt', 'a')
for val in labelled_preds:
    out.write('{}\n'.format(str(val+1)))
out.close()


# In[55]:


get_ipython().system('rm -rf data/')

