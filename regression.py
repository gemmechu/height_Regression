import cv2
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import matplotlib
# data storing library
import numpy as np
import pandas as pd
# torch libraries
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset, sampler
# architecture and data split library
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp
# augmenation library
from albumentations import (HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)
from albumentations.pytorch import ToTensor
# others
import os
import pdb
import time
import warnings
import random
from tqdm import tqdm_notebook as tqdm
import concurrent.futures
from pathlib import Path
import PIL
import collections
from torch.utils.tensorboard import SummaryWriter
import torchvision
from shutil import copyfile
# warning print supression
warnings.filterwarnings("ignore")

# *****************to reproduce same results fixing the seed and hash*******************
seed = 42
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# ************************************************************************
directory = '../../../../z/ghassena/GT30sec/heatmap'
im_dir = '../../../../z/ghassena/GT30sec/median'
df = []
# result = np.load('../Segmentation/sorted_50.npy',allow_pickle=True).item()
mydict = np.load('../Summary/variance_over_time_2.npy',allow_pickle=True).item()
for filename in os.listdir(directory):
    if filename.endswith('.npy'):
        filename = filename[:-3] + 'jpg'
        if mydict.get(filename) != None:
            if mydict[filename] < 35:
                if os.path.isfile(os.path.join(im_dir,filename)):
                    df.append(filename)
                else:
                    print(filename)
        else:
            print('None',filename)
        
print('my db: ',len(df))
print('len data',len(os.listdir(directory)))
pre = '../../../../z/ghassena/'
img_fol='GT30sec/median'
mask_fol='GT30sec/heatmap'


# df=[str(i)+'.png' for i in range(126)]
# # location of original and mask image
# img_fol='data/img'
# mask_fol='data/height'
mean, std=(0.485, 0.456, 0.406),(0.229, 0.224, 0.225)
# ************************************************************************


def get_transform(phase,mean,std):
    list_trans=[]
    if phase=='train':
        list_trans.extend([HorizontalFlip(p=0.5)])
    
    list_trans.extend([Resize(width=224,height=224),Normalize(mean=mean,std=std, p=1), ToTensor()])  #normalizing the data & then converting to tensors
    list_trans=Compose(list_trans)
    return list_trans


class MyDataset(Dataset):
    def __init__(self,df,img_fol,mask_fol,mean,std,phase):
        self.fname=df
        self.img_fol=img_fol
        self.mask_fol=mask_fol
        self.mean=mean
        self.std=std
        self.phase=phase
        self.trasnform=get_transform(phase,mean,std)
    def __getitem__(self, idx):
        name=self.fname[idx]
        img_name_path=os.path.join(self.img_fol,name)
        mask_name_path=img_name_path.split('.')[0].replace('median','heatmap')+'.npy'
        img=cv2.imread(str(pre + img_name_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask= np.load(str(pre + mask_name_path))
        
#         where_are_NaNs =   (np.isnan(mask))
#         mask[where_are_NaNs] = 0
        
        augmentation=self.trasnform(image=img, mask=mask)
        img_aug=augmentation['image']                           #[3,224,224] type:Tensor
        mask_aug=augmentation['mask']                           #[1,224,224] type:Tensor
        return img_aug, mask_aug
        

    def __len__(self):
        return len(self.fname)
    
def MyDataloader(df,img_fol,mask_fol,mean,std,phase,batch_size,num_workers):
    df_train,df_valid=train_test_split(df, test_size=0.2, random_state=69)
    df = df_train if phase=='train' else df_valid
    for_loader=MyDataset(df, img_fol, mask_fol, mean, std, phase)
#     print(for_loader.fname)
    dataloader=DataLoader(for_loader, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

    return dataloader, for_loader.fname

'''calculates dice scores when Scores class for it'''
def dice_score(pred, targ):
#     get boolean if there is nan value
    is_nan_targ =   (torch.isnan(targ))
    
#     exclude the nan value
    pred = pred[~is_nan_targ]
    targ = targ[~is_nan_targ]
    
    return (2. * (pred*targ).sum() / (pred+targ).sum())

class Scores:
    def __init__(self, phase, epoch):
        self.base_dice_scores = []

    def update(self, targets, outputs):
        probs = outputs
        dice= dice_score(probs, targets)
        self.base_dice_scores.append(dice)

    def get_metrics(self):
        dice = np.nanmean(self.base_dice_scores)         
        return dice

'''return dice score for epoch when called'''
def epoch_log(phase, epoch, epoch_loss, measure, start):
    '''logging the metrics at the end of an epoch'''
    dices= measure.get_metrics()    
    dice= dices                       
    print("Loss: %0.4f |dice: %0.4f" % (epoch_loss, dice))
    return dice

class My_Custom_loss(nn.Module):
    #get MSELoss by ignoring the prediction at nan pixel's 
    def __init__(self):
        super().__init__()
    def forward(self, output, target):
        is_nan =   (torch.isnan(target))
        
        
        loss = torch.square((output[~is_nan] - target[~is_nan]))

        
        loss = torch.mean(loss)
        
        return loss
class Trainer(object):
    def __init__(self,model):
        self.num_workers=4
        self.batch_size={'train': 16, 'val':16}

        self.lr=10e-3
        self.num_epochs=20
        self.phases=['train','val']
        self.best_loss=float('inf')
        self.device=torch.device("cuda:0")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        self.net=model.to(self.device)
        cudnn.benchmark= True

        self.criterion=My_Custom_loss()
        self.optimizer=optim.Adam(self.net.parameters(),lr=self.lr)
        self.scheduler=ReduceLROnPlateau(self.optimizer,mode='min',patience=3, verbose=True)
        self.dataloaders={phase: MyDataloader(df, img_fol,
                                               mask_fol, mean, std,
                                               phase=phase,batch_size=self.batch_size[phase],
                                               num_workers=self.num_workers)[0] for phase in self.phases}


    def forward(self, inp_images, tar_mask):
        inp_images=inp_images.to(self.device)
        tar_mask=tar_mask.to(self.device)
        pred_mask=self.net(inp_images)
        loss=self.criterion(pred_mask,tar_mask)
        
        return loss, pred_mask

    def iterate(self, epoch, phase):
        measure=Scores(phase, epoch)
        start=time.strftime("%H:%M:%S")
        print (f"Starting epoch: {epoch} | phase:{phase} | 🙊':{start}")
        
        batch_size=self.batch_size[phase]
        self.net.train(phase=="train")
        dataloader=self.dataloaders[phase]
        running_loss=0.0
        total_batches=len(dataloader)
        self.optimizer.zero_grad()
        
        for itr,batch in enumerate(dataloader):
#             print('batch',batch)
            images,mask_target=batch
#             print('mask', mask_target)
#             break
            loss, pred_mask=self.forward(images,mask_target)
  
            if phase=='train':
                loss.backward()

                self.optimizer.step()     
                
                self.optimizer.zero_grad()
                
            running_loss+= loss.item()
            pred_mask=pred_mask.detach().cpu()
            measure.update(mask_target,pred_mask)

        epoch_loss= running_loss /total_batches
        
        dice=epoch_log(phase, epoch, epoch_loss, measure, start)

        torch.cuda.empty_cache()

        return epoch_loss
    
    
    def start(self):
        
        for epoch in range (self.num_epochs):
            self.iterate(epoch,"train")
            state = {
                "epoch": epoch,
                "best_loss": self.best_loss,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            with torch.no_grad():
                val_loss=self.iterate(epoch,"val")
                self.scheduler.step(val_loss)
            if val_loss < self.best_loss:
                print("******** New optimal found, saving state ********")
                state["best_loss"] = self.best_loss = val_loss
                torch.save(state, "./model_office.pth")
                
def main():
    # imagenet mean/std will be used as the resnet backbone is trained on imagenet stats
    mean, std=(0.485, 0.456, 0.406),(0.229, 0.224, 0.225)
    print('start')
    model = smp.Unet("resnet34", encoder_weights="imagenet", classes=1, activation=None)
    model_trainer = Trainer(model)
    model_trainer.start()
from matplotlib.colors import LinearSegmentedColormap

def test():
    test_dataloader, names=MyDataloader(df,img_fol,mask_fol,mean,std,'val',1,4)
    ckpt_path='model_office.pth'
    
    device = torch.device("cuda:0")
    model = smp.Unet("resnet34", encoder_weights=None, classes=1, activation=None)
    model.to(device)
    model.eval()
    state = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state["state_dict"])
    if not (os.path.exists('output/pred')):
        os.makedirs('output/pred')
        os.makedirs('output/gt')
    # start prediction
    predictions = []
    colors = [(0, 0, 1), (0, 1, 1), (0, 1, 0.75), (0, 1, 0), (0.75, 1, 0),
              (1, 1, 0), (1, 0.8, 0), (1, 0.7, 0), (1, 0, 0)]
#     fig, (ax1,ax2)=plt.subplots(1,2)
#     fig.suptitle('predicted_mask//original_mask')
#     cm = LinearSegmentedColormap.from_list('sample', colors)
    cm = plt.get_cmap('viridis'); 
    cnt = 0
    
    for i, batch in enumerate((test_dataloader)):
        if cnt> 50:
            break
        images,mask_target = batch
        batch_preds = torch.sigmoid(model(images.to(device)))
        result = dice_score(batch_preds, mask_target.to(device))
        print(names[i], result)
        batch_preds = batch_preds.detach().cpu().numpy()

       
        is_nan_targ =   (torch.isnan(mask_target))
        mask_target[is_nan_targ] = 0
        
        plt.imshow(np.squeeze(batch_preds), cmap = cm)
        plt.colorbar(orientation='vertical')
        plt.savefig('output/pred/'+str(names[i]))
        plt.imshow(np.squeeze(mask_target),  cmap = cm)
        plt.savefig('output/gt/'+str(names[i]))
        plt.clf()
#         src  = os.path.join(directory,names[i])
#         dest = os.path.join('output/img',names[i])
#         copyfile(src,dest)
#         matplotlib.image.imsave('output/pred/'+str(names[i]),np.squeeze(batch_preds),vmin =0.5, vmax = 0.7,cmap =cm )
#         matplotlib.image.imsave('output/gt/'+str(names[i]), np.squeeze(mask_target),vmin =0.5, vmax = 0.7, cmap= cm)
#         plt.imshow(np.squeeze(mask_target),   cmap = cm)
        cnt += 1
        
# Starting epoch: 19 | phase:train | 🙊':04:35:08                                                                   │·······································································
# Loss: 0.0133 |dice: 0.4310                                                                                        │·······································································
# Starting epoch: 19 | phase:val | 🙊':04:35:44                                                                     │·······································································
# Loss: 0.0142 |dice: 0.4290   
main()