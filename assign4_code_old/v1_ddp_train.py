import sys, os
import time
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
from v1_ddp_dataset import get_deep_fashion

def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group('nccl',
        init_method='tcp://127.0.0.1:8900',
        rank=rank, world_size=world_size
    )

def model_creator():
    model = models.squeezenet1_1(pretrained=True)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5, inplace=False),
        nn.Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1)),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Conv2d(64, 10, kernel_size=(1, 1), stride=(1, 1)),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(output_size=(1, 1)),
    )
    for param in model.parameters():
        param.requires_grad = False

    for param in list(model.features[-1].parameters()) + list(model.classifier.parameters()):
        param.requires_grad = True
    return model

def train(model, data, epoch, criterion, optimizer, device):
    model.train()
    print('==========Train Epoch {}=========='.format(epoch))
    loss_list = []
    acc_count = 0
    total_count = 0

    for i, (image, label, _) in tqdm(enumerate(data), ascii=True, total=len(data)):
        image = image.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        score = model(image) # predict the label
        loss = criterion(score, label) # calculate error
        loss_list.append(loss.item())

        pred = torch.argmax(score, dim=1)
        correct = pred.eq(label)
        acc_count += correct.sum().item()
        total_count += image.shape[0]
        
        loss.backward()  # back-propagation
        optimizer.step() # gradient descent

    acc = acc_count / total_count * 100
    return sum(loss_list) / len(loss_list), acc

def test(model, data, criterion, device):
    model.eval()
    loss_list = []
    acc_count = 0
    total_count = 0

    for i, (image, label, _) in tqdm(enumerate(data), ascii=True, total=len(data)):
        image = image.to(device)
        label = label.to(device)

        score = model(image)
        loss = criterion(score, label)
        loss_list.append(loss.item())

        pred = torch.argmax(score, dim=1)
        correct = pred.eq(label)
        acc_count += correct.sum().item()
        total_count += image.shape[0]

    acc = acc_count / total_count * 100
    print('----------Acc: {}%----------'.format(acc))
    return sum(loss_list) / len(loss_list), acc

def run(rank, world_size, max_epochs=5):
    # create model and move it to GPU with id rank
    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(rank))
        print(device)
    else:
        device = torch.device('cpu')

    model = model_creator().to(device)

    if torch.cuda.is_available():
        ddp_model = DDP(model, device_ids=[rank])
    else:
        ddp_model = DDP(model)

    train_data, val_data = get_deep_fashion(rank, world_size)

    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=3e-5)

    t = time.time()

    for epoch in range(1, max_epochs + 1):
        loss_list = []
        total_count = 0
        acc_count = 0

        loader = train_data
        if rank == 0:
            loader = tqdm(train_data, total=len(train_data))

        train_loss, train_acc = train(model, train_data, epoch, criterion, optimizer, device)
        if rank == 0:
            print('epoch', epoch, 'acc', train_acc, \
                'loss', '{:.03}'.format(train_loss))
        # for image, category, _ in loader:
        #     image = image.to(device)
        #     category = category.to(device)

        #     optimizer.zero_grad()
        #     outputs = ddp_model(image)
        #     loss = criterion(outputs, category)
        #     loss.backward()
        #     optimizer.step()

        #     total_count += outputs.shape[0]
        #     correct = torch.argmax(outputs, dim=1).eq(category)
        #     acc_count += correct.sum().item()
        #     loss_list.append(loss.item())

        # if rank == 0:
        #     print('epoch', epoch, 'acc', acc_count / total_count,\
        #         'loss', '{:.03}'.format(sum(loss_list) / len(loss_list)))

    # output
    if rank == 0:
        t = time.time() - t
        print('Cost Time:', t, 'avg time', t / max_epochs)

if __name__ == '__main__':
    argv = sys.argv
    rank = 0
    size = 1

    print(f'Running basic DDP example on rank {rank}.')
    setup(rank, size)

    run(rank, size)
