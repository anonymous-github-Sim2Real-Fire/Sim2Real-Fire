import os
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchmetrics import AveragePrecision

from model.S2RFireTr import FireTr
from dataset.real_dataset import RealFireDataset
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
    test_dataset = RealFireDataset(cfg['dataset'])
    test_dataloader = DataLoader(test_dataset, batch_size=1)
    return test_dataloader

def initialize_model(cfg, device):
    model = FireTr(cfg['FireTr']).to(device)
    try:
        checkpoint_path = cfg['checkpoint']['checkpoint_path']
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint, strict=False)
        print('Loaded model from checkpoint.')
        return model
    except IOError:
        print("Checkpoint ERROR!")
        return model

def loss_fn(pred, target, dice, focal):
    return dice(pred, target, use_sigmoid=True) + focal(pred, target)

def test(dataloader, model, device, vis_dir, loss_dice, loss_focal):
    model.eval()
    val_loss = []
    metric = SegmentationMetric(2)
    auprc = AveragePrecision('binary', average='macro')
    total_auprc = 0
    total_iou = 0
    total_f1 = 0
    count = 0

    with torch.no_grad():
        for batch, (index, wxs, input_firesquence, output_firesquence, landfire, time_steps) in enumerate(dataloader):
            wxs = wxs.to(device)
            inputs = input_firesquence.to(device)
            targets = output_firesquence.to(device)
            landfire = landfire.to(device)
            time_steps = time_steps.to(device)
            pred = model(inputs, landfire, wxs, time_steps)
            loss = loss_fn(pred, targets, loss_dice, loss_focal)
            val_loss.append(loss.item())
            
            pred_sigmoid = torch.sigmoid(pred)
            pred_binary = torch.where(pred_sigmoid > 0.5, torch.ones_like(pred_sigmoid), torch.zeros_like(pred_sigmoid))
            pred_sigmoid, pred_binary, targets = pred_sigmoid.cpu().numpy(), pred_binary.cpu().numpy(), targets.cpu().numpy()
            pred_sigmoid, pred_binary, targets = pred_sigmoid.astype(np.float32), pred_binary.astype(np.int32), targets.astype(np.int32)
            
            b, t, h, w = pred_binary.shape
            inputs_array = inputs.cpu().numpy()
            
            for i in range(b):
                for j in range(t):
                    metric.addBatch(pred_binary[i, j], targets[i, j])
                    iou = metric.IntersectionOverUnion()[1]   
                    f1 = metric.f1()   
                    metric.reset()

                    auprc.update(torch.tensor(pred_sigmoid[i, j]), torch.tensor(targets[i, j]).long())
                    auprc_value = auprc.compute()
                    auprc.reset()
                    
                    print(f'Batch {batch}, Sample {i}, Time {j}')
                    print('iou_1: ' + str(iou))
                    print('f1: ' + str(f1))
                    print('auprc: ' + str(auprc_value))
                    
                    total_iou += iou
                    total_f1 += f1
                    total_auprc += auprc_value.item()
                    count += 1

                    name = f'{index[i].item()}_gt_{j+1}.png'
                    file_name = os.path.join(vis_dir, name)
                    img_gt = np.uint8(inputs_array[i, j, :, :] * 255)
                    cv2.imwrite(file_name, img_gt)
                    
                    name = f'{index[i].item()}_gt_{j+1+t}.png'
                    file_name = os.path.join(vis_dir, name)
                    img_gt = np.uint8(targets[i, j, :, :] * 255)
                    cv2.imwrite(file_name, img_gt)
                    
                    name = f'{index[i].item()}_pd_{j+1+t}.png'
                    file_name = os.path.join(vis_dir, name)
                    img_pd = pred_binary[i, j, :, :]
                    img_pd = np.maximum(img_pd, 0)
                    img_pd = np.minimum(img_pd, 1)
                    img_pd = np.uint8(img_pd * 255)
                    cv2.imwrite(file_name, img_pd)
            
    mean_epoch_loss = np.average(val_loss)
    mean_iou = total_iou / count
    mean_f1 = total_f1 / count
    mean_auprc = total_auprc / count

    print('val_loss: ' + str(mean_epoch_loss))
    print('mean_iou: ' + str(mean_iou))
    print('mean_f1: ' + str(mean_f1))
    print('mean_auprc: ' + str(mean_auprc))

    return mean_iou, mean_f1, mean_auprc

def main():
    setup_seed(12345)
    cfg = load_config('config/config.yaml')
    device = setup_device()
    test_dataloader= load_data(cfg)
    model = initialize_model(cfg, device)
    vis_dir = cfg['dataset']['vis_dir']
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)
    loss_dice = BinaryDiceLoss()
    loss_focal = BinaryFocalLoss()
          
    iou, f1, auprc = test(test_dataloader, model, device, vis_dir, loss_dice, loss_focal)

if __name__ == '__main__':
    main() 