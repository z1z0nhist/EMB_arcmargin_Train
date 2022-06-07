import timm
import wandb
##### wandb train visualization ####
import loss_function

try:
    api_key = '30bae7361b34d493a9c5f8b8908e275a70c23bd1'
    wandb.login(key=api_key)
    anony = None
except:
    anony = "must"
    print('If you want to use your W&B account, go to Add-ons -> Secrets and provide your W&B access token. Use the Label name as wandb_api. \nGet your W&B access token from here: https://wandb.ai/authorize')
# base library #
import gc
import os
import time
import copy
import json
import argparse

from config.config import config as cfg

import torch
import torch.nn as nn

import torch.optim as optim

from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

import numpy as np
import albumentations as A

from albumentations.pytorch import transforms

from collections import defaultdict
from sklearn.preprocessing import LabelEncoder

from training import training_epoch_arc, valid_one_epoch_arc
from utils.make_scv import make_csv_file
from Module import EMB_arc, EMB_Dataset, EMB_Dataset_arc_test

parser = argparse.ArgumentParser(description= 'train code')
parser.add_argument('--train_path', type = str, help = 'inference file path')
parser.add_argument('--valid_path', type = str, help = 'inference file path')
args = parser.parse_args()

config = {
    "model_name" : 'tf_efficientnet_b6_ns',
    "sch" : 'CosineAnnealingLR',
    "epoch" : 20,
    "img_size" : 448,
    "patience" : 20
}
def data_transforms_img(img_size):
    data_transforms = {
        "train": A.Compose([
            # A.ToGray(p=1),
            A.Resize(img_size, img_size),
            A.ShiftScaleRotate(shift_limit=0.1,
                               scale_limit=0.15,
                               rotate_limit=60,
                               p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=0.2,
                sat_shift_limit=0.2,
                val_shift_limit=0.2,
                p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1),
                contrast_limit=(-0.1, 0.1),
                p=0.5
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
            transforms.ToTensorV2()]),

        "valid": A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0
            ),
            transforms.ToTensorV2()])
    }
    return data_transforms

def test_label_encoding(df, encoder):
  df['new_labels'] = 'None'
  for i in range(len(df)):
    if df['labels'][i] in encoder.classes_:
      df['new_labels'][i] = int((np.where(encoder.classes_ == df['labels'][i])[0]))
      df[df['new_labels'] != 'None']
  return df[df['new_labels'] != 'None']

def fetch_scheduler(optimizer):
    if config['sch'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=500,
                                                   eta_min=1e-6)
    elif config['sch'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.9, verbose=True)
    # elif sch == 'CosineAnnealingWarmRestarts':
    #     scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=CONFIG['T_0'],
    #                                                          eta_min=CONFIG['min_lr'])
    elif config['sch'] == None:
        return None

    return scheduler
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def run_training(model, optimizer, scheduler, device, num_epochs, train_type, fold, best_epoch_loss):
    wandb.watch(model, log_freq=100)

    if torch.cuda.is_available():
        print("[INFO] Using GPU: {}\n".format(torch.cuda.get_device_name()))
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    # best_epoch_loss = np.inf
    history = defaultdict(list)
    for epoch in range((fold*num_epochs)+1, (fold*num_epochs) + 2 + num_epochs):
        gc.collect()
        if train_type == 'arc':
            train_epoch_loss = training_epoch_arc(model, optimizer, scheduler,
                                              dataloader=Train_loader,
                                              device=device, epoch=epoch)

            val_epoch_loss = valid_one_epoch_arc(model, Valid_loader, device=device,
                                             epoch=epoch)
        history['Train Loss'].append(train_epoch_loss)

        # Log the metrics
        wandb.log({"Train Loss": train_epoch_loss})
        wandb.log({"Valid Loss": val_epoch_loss})

        if val_epoch_loss <= best_epoch_loss:
            print(f"Validation Loss Improved ({best_epoch_loss} ---> {val_epoch_loss})")
            best_epoch_loss = val_epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = "{}/Loss{:.4f}_epoch{:.0f}.bin".format(config['model_name'], best_epoch_loss, epoch)
            if not os.path.isdir('{0}/'.format(config['model_name'])):  # 없으면 새로 생성하는 조건문
                os.mkdir('{0}/'.format(config['model_name']))
            torch.save(model.state_dict(), PATH)

        print()

        end = time.time()
        time_elapsed = end - start
        print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
            time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
        print("Best Loss: {:.4f}".format(best_epoch_loss))

        # load best model weights
        model.load_state_dict(best_model_wts)

    return model, history, best_epoch_loss

if __name__ == "__main__":
    # train_path = 'C:/Users/ikh/Downloads/EMB_new/train/train'
    # vaild_path = 'C:/Users/ikh/Downloads/EMB_new/test_sub'
    train_path = args.train_path
    vaild_path = args.valid_path

    train_df = make_csv_file(train_path)
    if os.path.isfile('label.json'):
        with open('label.json', 'r') as file:
            label_name = json.load(file)
        encoder = LabelEncoder()
        if len(list(label_name.keys())) == len(os.listdir(train_path)):
            # *Caution*  if json A(before) and json B(now) length are the same. please check both have same file list
            encoder.fit(list(label_name.keys()))
            train_df['new_labels'] = encoder.fit_transform(train_df['labels'])
        else:
            encoder = LabelEncoder()
            train_df['new_labels'] = encoder.fit_transform(train_df['labels'])
            target_encodings = {t: i for i, t in enumerate(encoder.classes_)}
            with open('label.json', 'w') as f:
                json.dump(target_encodings, f)
    else:
        encoder = LabelEncoder()
        train_df['new_labels'] = encoder.fit_transform(train_df['labels'])
        target_encodings = {t: i for i, t in enumerate(encoder.classes_)}
        with open('label.json', 'w') as f:
            json.dump(target_encodings, f)

    target_size = len(encoder.classes_)

    vaild_df = test_label_encoding(make_csv_file(vaild_path), encoder)
    vaild_df['true'] = 0

    # loss type #
    # ElasticArcFace , ElasticArcFacePlus
    # ElasticCosFace , ElasticCosFacePlus
    # ArcFace , CosFace
    model = EMB_arc(model_name = config['model_name'], target_size = target_size, loss= 'ArcFace')
    model.to(device)
    best_epoch_loss = np.inf
    for k in range(5):
        data_transforms = data_transforms_img(config['img_size'])

        Train = EMB_Dataset(train_df, transforms=data_transforms['train'])
        Train_loader = DataLoader(Train, batch_size=8, num_workers=4, shuffle=True, pin_memory=True, drop_last=True)

        Valid = EMB_Dataset_arc_test(vaild_df, transforms=data_transforms['valid'])
        Valid_loader = DataLoader(Valid, batch_size=8, num_workers=4, shuffle=True, pin_memory=True, drop_last=True)


        optimizer = optim.Adam(model.parameters(), lr=0.1,
                               weight_decay=5e-4)
        scheduler = fetch_scheduler(optimizer)

        run = wandb.init(project='EMB',
                         job_type='Train',
                         tags=[str(config['model_name']), str(config['img_size']), 'normal'],
                         anonymous='must')
        model, history,best_epoch_loss = run_training(model, optimizer, scheduler,
                                      device=device,
                                      num_epochs=config['epoch'],train_type="arc",best_epoch_loss=best_epoch_loss,fold=k)