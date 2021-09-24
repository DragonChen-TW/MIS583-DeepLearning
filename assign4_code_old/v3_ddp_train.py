import sys, os
import time
import matplotlib.pyplot as plt
#
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
#
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
#
from v3_models import (
    get_squeezenet, get_resnet,
)
from v3_train import (
    train, test,
    batch_size, workers
)
from v3_dataset import get_ddp_data

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--local_rank', type=int)
args = parser.parse_args()
rank = args.local_rank
size = int(os.environ['WORLD_SIZE'])
print('rank', rank)

# setting
dist.init_process_group(backend='nccl', rank=rank)

def run(rank, max_epochs=100):
    if torch.cuda.is_available():
        device = torch.device('cuda', rank + 1)
    else:
        print('CUDA is not available!')
        return
    print(device)

    model = get_resnet('cate').to(device)
    ddp_model = DDP(model, device_ids=[rank + 1], output_device=rank + 1)

    train_data, val_data, _ = get_ddp_data(rank, size, batch_size, workers)

    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4)

    t = time.time()
    train_loss_list = []
    train_acc_list = []
    test_loss_list = []
    test_acc_list = []
    for epoch in range(1, max_epochs + 1):
        use_tqdm = (rank == 0)
        train_loss, train_acc = train(model, train_data, epoch, criterion, optimizer,
            device, use_tqdm=use_tqdm)
        test_loss, test_acc = test(model, val_data, criterion,
            device, use_tqdm=use_tqdm)

        if rank == 0:
            train_acc_list.append(train_acc)
            train_loss_list.append(train_loss)
            test_acc_list.append(test_acc)
            test_loss_list.append(test_loss)
            print('epoch', epoch, 'Train acc', '{:.04}'.format(train_acc), \
                'Train loss', '{:.04}'.format(train_loss))
            print('epoch', epoch, 'Test acc', '{:.04}'.format(test_acc), \
                'Test loss', '{:.04}'.format(test_loss))

            plt.plot(range(1, epoch + 1), train_loss_list)
            plt.plot(range(1, epoch + 1), test_loss_list, color='r')
            plt.legend(['train_loss', 'test_loss'])
            plt.savefig('attr_loss.png')
            plt.cla()
            
            plt.plot(range(1, epoch + 1), train_acc_list)
            plt.plot(range(1, epoch + 1), test_acc_list, color='r')
            plt.ylim(0, 100)
            plt.legend(['train_acc', 'test_acc'])
            plt.savefig('attr_acc.png')
            plt.cla()
            
    if rank == 0:
        t = time.time() - t
        print('Total cost', t, 'Avg cost', t / max_epochs)

# python3 -m torch.distributed.launch --nproc_per_node 3 v2_ddp_train.py
if __name__ == '__main__':
    run(rank)
