import time, os
os.environ['CUDA_VISABLE_DEVICES'] = '1,2,3'
#
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
from sklearn.metrics import f1_score
#
from v3_dataset import get_deep_fashion
from v3_models import (
    get_squeezenet, get_resnet,
)

batch_size = 126
workers = 16

def get_data():
    train_dataset, val_dataset, test_dataset = get_deep_fashion()
    train_data = DataLoader(dataset=train_dataset, batch_size=batch_size,
                            num_workers=workers, pin_memory=True,
                            shuffle=True)
    val_data = DataLoader(dataset=val_dataset, batch_size=batch_size,
                            num_workers=workers, pin_memory=True)
    test_data = DataLoader(dataset=test_dataset, batch_size=batch_size,
                            num_workers=workers, pin_memory=True)
    return train_data, val_data, test_data

def train(model, data, epoch, criterion, optimizer, device, use_tqdm=True):
    model.train()
    if use_tqdm:
        print('==========Train Epoch {}=========='.format(epoch))
    loss_list = []
    acc_count = 0
    total_count = 1

    loader = enumerate(data)
    if use_tqdm:
        loader = tqdm(loader, ascii=True, total=len(data))
    for i, (image, category, attribute) in loader:
        image = image.to(device)
        category = category.to(device)
        # attribute = attribute.to(device)

        optimizer.zero_grad()
        score = model(image)
        loss = criterion(score, category)
        # loss = criterion(score, attribute)
        loss_list.append(loss.item())

        pred = torch.argmax(score, dim=1)
        acc_count += pred.eq(category).sum().item()

        # pred = torch.sigmoid(score).cpu().detach().numpy() > 0.5
        # acc = f1_score(attribute.cpu().int().numpy(), pred, average='samples')  * attribute.shape[0]
        # acc_count += acc
        
        total_count += image.shape[0]
        loss.backward()
        optimizer.step()

    acc = acc_count / total_count * 100
    return sum(loss_list) / len(loss_list), acc

def test(model, data, criterion, device, use_tqdm=True):
    model.eval()
    loss_list = []
    acc_count = 0
    total_count = 1

    loader = enumerate(data)
    if use_tqdm:
        loader = tqdm(loader, ascii=True, total=len(data))
    for i, (image, category, attribute) in loader:
        image = image.to(device)
        category = category.to(device)
        # attribute = attribute.to(device)

        score = model(image)
        loss = criterion(score, category)
        # loss = criterion(score, attribute)
        loss_list.append(loss.item())

        pred = torch.argmax(score, dim=1)
        correct = pred.eq(category)
        acc_count += correct.sum().item()

        # pred = torch.sigmoid(score).cpu().detach().numpy() > 0.5
        # acc = f1_score(attribute.cpu().int().numpy(), pred, average='samples')  * attribute.shape[0]
        # acc_count += acc

        total_count += image.shape[0]

    acc = acc_count / total_count * 100
    return sum(loss_list) / len(loss_list), acc

def run(max_epochs=5):
    if torch.cuda.is_available():
        device = torch.device('cuda:1')
    else:
        device = torch.device('cpu')
    print('Device:', device)

    model = get_resnet('cate').to(device)
    # model = get_attr_model().to(device)
    train_data, val_data, test_data = get_data()

    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4)
    # optimizer = torch.optim.AdamW(params, lr=1e-3)

    t = time.time()
    for epoch in range(1, max_epochs + 1):
        train_loss, train_acc = train(model, train_data, epoch, criterion, optimizer, device)
        test_loss, test_acc = test(model, val_data, criterion, device)
        print('epoch', epoch, 'Train acc', '{:.04}'.format(train_acc), \
            'Train loss', '{:.04}'.format(train_loss))
        print('epoch', epoch, 'Test acc', '{:.04}'.format(test_acc), \
            'Test loss', '{:.04}'.format(test_loss))

if __name__ == '__main__':
    run()

    