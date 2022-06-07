import gc
from tqdm import tqdm

from utils import Averagemeter

import torch
import torch.nn as nn

def criterion(outputs, labels):
    return nn.CrossEntropyLoss()(outputs, labels)

############# train one epoch ###################################

def training_epoch_arc(model, optimizer, scheduler, dataloader, device, epoch):
    # loss_ = Averagemeter()
    model.train()

    dataset_size = 0
    running_loss = 0.0

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:

        img = data['image'].to(device, dtype=torch.float)
        labels = data['new_labels'].to(device, dtype=torch.long)

        batch_size = img.size(0)

        outputs = model(img,labels)
        loss = criterion(outputs, labels)
        loss.backward()
        # loss_.update(loss.item(), 1)
        if (step + 1) % 1 == 0:
            optimizer.step()

            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss,
                        LR=optimizer.param_groups[0]['lr'])
    gc.collect()
    return epoch_loss

@torch.inference_mode()
def valid_one_epoch_arc(model, dataloader, device, epoch):
    # loss_ = Averagemeter()
    model.eval()

    dataset_size = 0
    running_loss = 0.0

    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, data in bar:
        images = data['image'].to(device, dtype=torch.float)
        labels = data['true'].to(device, dtype=torch.long)
        true_labels = data['new_labels'].to(device, dtype=torch.long)

        batch_size = images.size(0)

        outputs = model(images,labels)
        loss = criterion(outputs, true_labels)

        # loss_.update(loss.item(), 1)

        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size

        bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss)

    gc.collect()

    return epoch_loss
