import sys, os
import time
import matplotlib.pyplot as plt
import json
#
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
from sklearn.metrics import f1_score
#
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
#
from v3_models import (
    get_joint_squeezenet, get_joint_resnet,
    get_joint_resnext101, F1Loss,
)
from v3_dataset import get_ddp_data
from v3_train import batch_size, workers

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int)
args = parser.parse_args()
rank = args.local_rank
size = int(os.environ['WORLD_SIZE'])
print('rank', rank)

# setting
dist.init_process_group(backend='nccl', rank=rank)

def train(model, data, epoch, criterion_cate, criterion_attr,
    optimizer, device, use_tqdm=True
):
    model.train()
    if use_tqdm:
        print('==========Train Epoch {}=========='.format(epoch))
    loss_cate_list = []
    loss_attr_list = []
    acc_count_cate = 0
    acc_count_attr = 0
    acc_mean_attr = 0
    total_count = 0

    loader = enumerate(data)
    if use_tqdm:
        loader = tqdm(loader, ascii=True, total=len(data))
    for i, (image, category, attribute) in loader:
        image = image.to(device)
        category = category.to(device)
        attribute = attribute.to(device)

        # output
        optimizer.zero_grad()
        out_cate, out_attr = model(image)
        # calculate loss
        loss_cate = criterion_cate(out_cate, category)
        loss_attr = criterion_attr(out_attr, attribute)
        loss = loss_cate + loss_attr
        loss_cate_list.append(loss_cate.item())
        loss_attr_list.append(loss_attr.item())

        # calculate acc
        pred_cate = torch.argmax(out_cate, dim=1)
        acc_count_cate += pred_cate.eq(category).sum().item()

        # pred_attr = torch.sigmoid(out_attr).cpu().detach().numpy() > 0.5
        # acc_count_attr += f1_score(attribute.cpu().int().numpy(), pred_attr, average='samples')  * attribute.shape[0]
        pred_attr = torch.sigmoid(out_attr) > 0.5
        # acc_count_attr += (attribute == pred_attr).float().mean().item() * attribute.shape[0]
        acc_count_attr += f1_acc(attribute, pred_attr).item() * attribute.shape[0]
        
        # optimizer
        total_count += image.shape[0]
        loss.backward()
        optimizer.step()

    acc_cate = acc_count_cate / total_count * 100
    acc_attr = acc_count_attr / total_count * 100
    loss_cate = sum(loss_cate_list) / len(loss_cate_list)
    loss_attr = sum(loss_attr_list) / len(loss_attr_list)
    return loss_cate, loss_attr, acc_cate, acc_attr

def test(model, data, criterion_cate, criterion_attr, device, use_tqdm=True):
    model.eval()
    loss_cate_list = []
    loss_attr_list = []
    acc_count_cate = 0
    acc_count_attr = 0
    total_count = 0

    loader = enumerate(data)
    if use_tqdm:
        loader = tqdm(loader, ascii=True, total=len(data))
    for i, (image, category, attribute) in loader:
        image = image.to(device)
        category = category.to(device)
        attribute = attribute.to(device)

        out_cate, out_attr = model(image)
        loss_cate = criterion_cate(out_cate, category)
        loss_attr = criterion_attr(out_attr, attribute)
        loss_cate_list.append(loss_cate.item())
        loss_attr_list.append(loss_attr.item())

        pred_cate = torch.argmax(out_cate, dim=1)
        acc_count_cate += pred_cate.eq(category).sum().item()

        # pred_attr = torch.sigmoid(out_attr).cpu().detach().numpy() > 0.5
        # acc_count_attr += f1_score(attribute.cpu().int().numpy(), pred_attr, average='samples')  * attribute.shape[0]
        pred_attr = torch.sigmoid(out_attr) > 0.5
        # acc_count_attr += (attribute == pred_attr).float().mean().item() * attribute.shape[0]
        acc_count_attr += f1_acc(attribute, pred_attr).item() * attribute.shape[0]

        total_count += image.shape[0]

    acc_cate = acc_count_cate / total_count * 100
    acc_attr = acc_count_attr / total_count * 100
    loss_cate = sum(loss_cate_list) / len(loss_cate_list)
    loss_attr = sum(loss_attr_list) / len(loss_attr_list)
    return loss_cate, loss_attr, acc_cate, acc_attr

def f1_acc(true, pred, epsilon=1e-7):
    # pred = (torch.sigmoid(pred) > 0.5).int()
    pred = pred.int()

    tp = (true * pred).sum(dim=1)
    tn = ((1 - true) * (1 - pred)).sum(dim=1)
    fp = ((1 - true) * pred).sum(dim=1)
    fn = (true * (1 - pred)).sum(dim=1)

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    f1 = f1.clamp(min=epsilon, max=1-epsilon)
    return f1.mean()

def run(rank, max_epochs=5):
    if torch.cuda.is_available():
        device = torch.device('cuda', rank + 1)
    else:
        print('CUDA is not available!')
        return
    print(device)

    model = get_joint_resnext101().to(device)
    ddp_model = DDP(model, device_ids=[rank + 1], output_device=rank + 1)

    train_data, val_data, _ = get_ddp_data(rank, size, batch_size, workers)

    criterion_cate = nn.CrossEntropyLoss()
    # criterion_attr = nn.BCEWithLogitsLoss()
    criterion_attr = F1Loss()
    params_feat = [p for p in model.features.parameters() if p.requires_grad]
    params_cate = [p for p in model.fc_cate.parameters() if p.requires_grad]
    params_attr = [p for p in model.fc_attr.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {'params': params_feat, 'lr': 3e-5},
        {'params': params_cate, 'lr': 3e-4},
        {'params': params_attr, 'lr': 3e-4},
    ])

    t = time.time()
    loss_cate_list = []
    loss_attr_list = []
    acc_cate_list = []
    acc_attr_list = []

    best_acc_cate = 0.5
    best_records = []
    for epoch in range(1, max_epochs + 1):
        use_tqdm = (rank == 0)
        (
            train_loss_cate, train_loss_attr,
            train_acc_cate, train_acc_attr
        ) = train(model, train_data, epoch,
            criterion_cate, criterion_attr, optimizer,
            device, use_tqdm=use_tqdm)
        (
            test_loss_cate, test_loss_attr,
            test_acc_cate, test_acc_attr
        ) = test(model, val_data,
            criterion_cate, criterion_attr,
            device, use_tqdm=use_tqdm)

        if rank == 0:
            loss_cate_list.append((train_loss_cate, test_loss_cate))
            loss_attr_list.append((train_loss_attr, test_loss_attr))
            acc_cate_list.append((train_acc_cate, test_acc_cate))
            acc_attr_list.append((train_acc_attr, test_acc_attr))
            
            print('Epoch', epoch)
            plot_metric(epoch, 'loss_cate', loss_cate_list, 'loss_cate.png')
            plot_metric(epoch, 'loss_attr', loss_attr_list, 'loss_attr.png')
            plot_metric(epoch, 'acc_cate', acc_cate_list, 'acc_cate.png')
            plot_metric(epoch, 'acc_attr', acc_attr_list, 'acc_attr.png')

            if test_acc_cate > best_acc_cate:
                best_acc_cate = test_acc_cate
                torch.save(model.state_dict(), 'ckpts/e{:03}.pt'.format(epoch))
                best_records.append({
                    'epoch': epoch,
                    'train_acc_cate': train_acc_cate, 'train_acc_attr': train_acc_attr,
                    'test_acc_cate': test_acc_cate, 'test_acc_attr': test_acc_attr,
                    'train_loss_cate': train_loss_cate, 'train_loss_attr': train_loss_attr,
                    'test_loss_cate': test_loss_cate, 'test_loss_attr': test_loss_attr,
                })
                with open('ckpts/records.json', 'w') as f:
                    json.dump(best_records, f, indent='  ')
            if epoch % 2 == 0:
                torch.save(model.state_dict(), 'ckpts/save_e{:03}.pt'.format(epoch))
            
    if rank == 0:
        t = time.time() - t
        print('Total cost', t, 'Avg cost', t / max_epochs)

def plot_metric(epoch, title, metrics, f_name):
    print(title, 'Train: {:.04f} Test: {:.04f}'.format(metrics[-1][0],metrics[-1][1]))

    plt.plot(range(1, epoch + 1), [m[0] for m in metrics])
    plt.plot(range(1, epoch + 1), [m[1] for m in metrics], color='r')
    plt.title(title)
    plt.legend(['train', 'test'])
    plt.savefig(f_name)
    plt.cla()

# python3 -m torch.distributed.launch --nproc_per_node 3 v3_joint_train.py
if __name__ == '__main__':
    run(rank)
