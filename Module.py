import cv2
import math
import loss_function

from config.config import config as cfg

import timm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class EMB_arc(nn.Module):
    def __init__(self,model_name,target_size,pretrained=True,loss = "ElasticArcFace"):
        super(EMB_arc, self).__init__()
        self.model  = timm.create_model(model_name = model_name,pretrained=pretrained)
        if model_name.find('efficient') != -1:
            n_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(n_features, cfg.embedding_size)
            if loss == "ElasticArcFace":
                self.arc = loss_function.ElasticArcFace(in_features=cfg.embedding_size, out_features=target_size, s=cfg.s,
                                             m=cfg.m, std=cfg.std)
            elif loss == "ElasticArcFacePlus":
                self.arc = loss_function.ElasticArcFace(in_features=cfg.embedding_size, out_features=target_size,
                                                        s=cfg.s,m=cfg.m, std=cfg.std,plus=True)
            elif loss == "ElasticCosFace":
                self.arc = loss_function.ElasticCosFace(in_features=cfg.embedding_size, out_features=target_size,
                                                        s=cfg.s, m=cfg.m, std=cfg.std)
            elif loss == "ElasticCosFacePlus":
                self.arc = loss_function.ElasticArcFace(in_features=cfg.embedding_size, out_features=target_size,
                                                        s=cfg.s,m=cfg.m, std=cfg.std,plus=True)
            elif loss == "ArcFace":
                self.arc = loss_function.ArcFace(in_features=cfg.embedding_size, out_features=target_size, s=cfg.s,
                                             m=cfg.m)
            elif loss == "CosFace":
                self.arc = loss_function.CosFace(in_features=cfg.embedding_size, out_features=target_size, s=cfg.s,
                                             m=cfg.m)
            else:
                print("unknown")
        else:
            n_features = self.model.fc.in_features
            self.model.fc = nn.Linear(n_features, cfg.embedding_size)
            if loss == "ElasticArcFace":
                self.arc = loss_function.ElasticArcFace(in_features=cfg.embedding_size, out_features=target_size, s=cfg.s,
                                             m=cfg.m, std=cfg.std)
            elif loss == "ElasticArcFacePlus":
                self.arc = loss_function.ElasticArcFace(in_features=cfg.embedding_size, out_features=target_size,
                                                        s=cfg.s,m=cfg.m, std=cfg.std,plus=True)
            elif loss == "ElasticCosFace":
                self.arc = loss_function.ElasticCosFace(in_features=cfg.embedding_size, out_features=target_size,
                                                        s=cfg.s, m=cfg.m, std=cfg.std)
            elif loss == "ElasticCosFacePlus":
                self.arc = loss_function.ElasticArcFace(in_features=cfg.embedding_size, out_features=target_size,
                                                        s=cfg.s,m=cfg.m, std=cfg.std,plus=True)
            elif loss == "ArcFace":
                self.arc = loss_function.ArcFace(in_features=cfg.embedding_size, out_features=target_size, s=cfg.s,
                                             m=cfg.m, std=cfg.std)
            elif loss == "CosFace":
                self.arc = loss_function.CosFace(in_features=cfg.embedding_size, out_features=target_size, s=cfg.s,
                                             m=cfg.m, std=cfg.std)
            else:
                print("unknown")
    def forward(self, images, labels):
        features = self.model(images)
        features = F.normalize(features)
        output = self.arc(features,labels)
        return output

class EMB_Dataset_arc_test(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.img_dir = df['PATH'].values
        self.labels = df['new_labels'].values
        self.true_ = df['true'].values
        self.transform = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.img_dir[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label = self.labels[index]
        true = self.true_[index]
        if self.transform:
            img = self.transform(image=img)["image"]

        return {'image': img,
                'new_labels': torch.tensor(label, dtype=torch.long),
                'true' : torch.tensor(true, dtype=torch.long),
                'path': img_path.split("\\")[1]}
class EMB_Dataset(Dataset):
    def __init__(self, df, transforms=None):
        self.df = df
        self.img_dir = df['PATH'].values
        self.labels = df['new_labels'].values
        self.transform = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.img_dir[index]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label = self.labels[index]
        if self.transform:
            img = self.transform(image=img)["image"]

        return {'image': img,
                'new_labels': torch.tensor(label, dtype=torch.long),
                'path': img_path}