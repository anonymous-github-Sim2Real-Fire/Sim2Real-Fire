import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torchmetrics import AveragePrecision

from model.S2RFireTr import FireTr
from dataset.sim_dataset import SimFireDataset
from lib.dice_loss import BinaryDiceLoss
from lib.focal_loss import BinaryFocalLoss
from metrics.metrics import SegmentationMetric
from utils import setup_seed, load_config

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def setup_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    return device

def load_data(cfg):
    train_dataset = SimFireDataset(cfg['dataset'])
    val_dataset = SimFireDataset(cfg['dataset'], training=False)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg['training']['batch_size'], num_workers=cfg['dataset']['num_workers'], pin_memory=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg['training']['batch_size'], num_workers=cfg['dataset']['num_workers'], pin_memory=True, drop_last=True)
    print(f'len of train_dataloader:{len(train_dataloader)}')
    print(f'len of val_dataloader:{len(val_dataloader)}')
    return train_dataloader, val_dataloader

def initialize_model(cfg, device):
    model = FireTr(cfg['FireTr']).to(device)
    if not os.path.exists(cfg['checkpoint']['model_save_path']):
        os.makedirs(cfg['checkpoint']['model_save_path'])
    if cfg['checkpoint']['checkpoint']:
        checkpoint_path = cfg['checkpoint']['checkpoint_path']
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint, strict=False)
        print('Loaded model from checkpoint.')
    return model

def initialize_optimizer_scheduler(model, cfg):
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['training']['learning_rate'])
    scheduler = StepLR(optimizer, step_size=5, gamma=0.95)
    return optimizer, scheduler

def loss_fn(pred, target, dice, focal):
    return dice(pred, target, use_sigmoid=True) + focal(pred, target)

def train_loop(dataloader, model, optimizer, device, loss_dice, loss_focal):
    model.train()
    train_loss = []
    for batch, (index, input_squence, output_squence, fuel, vegetation, topography, satellite_images, weather_data, timestamps) in enumerate(dataloader):
        input_squence, output_squence, fuel, vegetation, topography, satellite_images, weather_data, timestamps = \
            input_squence.to(device), output_squence.to(device), fuel.to(device), vegetation.to(device), topography.to(device),satellite_images.to(device), weather_data.to(device), timestamps.to(device)
        optimizer.zero_grad()
        pred = model(input_squence, fuel, vegetation, topography, satellite_images, weather_data, timestamps)
        loss = loss_fn(pred, output_squence, loss_dice, loss_focal)
        train_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    mean_epoch_loss = np.average(train_loss)
    print(f'Train loss: {mean_epoch_loss:.4f}')
    return mean_epoch_loss

def val_loop(dataloader, model, device, loss_dice, loss_focal):
    model.eval()
    val_loss = []
    metric = SegmentationMetric(2)
    auprc = AveragePrecision('binary', average='macro')
    auprc_list = []

    with torch.no_grad():
        for batch, (index, input_squence, output_squence, fuel, vegetation, topography, satellite_images, weather_data, timestamps) in enumerate(dataloader):
            input_squence, output_squence, fuel, vegetation, topography, satellite_images, weather_data, timestamps = \
            input_squence.to(device), output_squence.to(device), fuel.to(device), vegetation.to(device), topography.to(device),satellite_images.to(device), weather_data.to(device), timestamps.to(device)
            b, n, h, w = input_squence.shape
            pred = model(input_squence, fuel, vegetation, topography, satellite_images, weather_data, timestamps)
            loss = loss_fn(pred, output_squence, loss_dice, loss_focal)
            val_loss.append(loss.item())

            pred = torch.sigmoid(pred)
            auprc.update(pred, output_squence.long())
            auprc_value = auprc.compute()
            if not torch.isnan(auprc_value):
                auprc_list.append(auprc_value.item())
            auprc.reset()

            pred = torch.where(pred > 0.5, torch.ones_like(pred), torch.zeros_like(pred))
            pred, output_squence = pred.cpu().numpy(), output_squence.cpu().numpy()
            pred, output_squence = pred.astype(np.int32), output_squence.astype(np.int32)
            metric.addBatch(pred, output_squence)
           
    mean_epoch_loss = np.average(val_loss)
    print(f'Val loss: {mean_epoch_loss:.4f}')
    iou = metric.IntersectionOverUnion()
    f1 = metric.f1()
    mAp = np.average(auprc_list)
    metric.reset()
    print(f'Valset: IOU: {iou[1]:.4f}  F1: {f1:.4f}  AUPRC: {mAp:.4f}')
    return iou[1], mAp

def save_best_model(model, iou, mAp, best_metrics, model_save_path):
    best_iou, best_map = best_metrics

    if iou > best_iou:
        print(f'Saving model with best IOU: {iou:.4f}')
        best_iou = iou
        torch.save(model.state_dict(), os.path.join(model_save_path, 'best_iou.pth'))

    if mAp > best_map:
        print(f'Saving model with best mAP: {mAp:.4f}')
        best_map = mAp
        torch.save(model.state_dict(), os.path.join(model_save_path, 'best_map.pth'))

    return best_iou, best_map

def main():
    setup_seed(12345)
    cfg = load_config('config/config.yaml')
    device = setup_device()
    train_dataloader, val_dataloader = load_data(cfg)
    model = initialize_model(cfg, device)
    optimizer, scheduler = initialize_optimizer_scheduler(model, cfg)
    
    loss_dice = BinaryDiceLoss()
    loss_focal = BinaryFocalLoss()
    
    best_metrics = (0.0, 0.0)
    
    for epoch in range(cfg['training']['epochs']):
        print(f'Epoch: {epoch + 1}')
        train_loss = train_loop(train_dataloader, model, optimizer, device, loss_dice, loss_focal)
        iou, mAp = val_loop(val_dataloader, model, device, loss_dice, loss_focal)
        scheduler.step()
        
        best_metrics = save_best_model(model, iou, mAp, best_metrics, cfg['checkpoint']['model_save_path'])

if __name__ == '__main__':
    main()
