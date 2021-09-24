import time, os
os.environ['CUDA_VISABLE_DEVICES'] = '1,2,3'
#
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
#
from v2_dataset import get_deep_fashion
from sklearn.metrics import f1_score

batch_size = 63
workers = 8

def get_model():
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

def get_attr_model():
    model = models.squeezenet1_1(pretrained=True)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5, inplace=False),
        nn.Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1)),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Conv2d(64, 15, kernel_size=(1, 1), stride=(1, 1)),
        nn.AdaptiveAvgPool2d(output_size=(1, 1)),
    )
    for param in model.parameters():
        param.requires_grad = False

    for param in list(model.features[-1].parameters()) + list(model.classifier.parameters()):
        param.requires_grad = True
    return model

class JointModel(nn.Module):
    def __init__(self):
        super().__init__()
        m = models.squeezenet1_1(pretrained=True)
        m.classifier = nn.Sequential()
        self.features = m
        
        self.fc_cate = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Conv2d(64, 10, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )
        self.fc_attr = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Conv2d(64, 15, kernel_size=(1, 1), stride=(1, 1)),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )
    
    def forward(self, x):
        features = self.features(x)
        print(features.shape)
        out_cate = self.fc_cate(features)
        out_attr = self.fc_attr(features)
        return out_cate, out_attr

def get_joint_model():
    model = JointModel()
    return model

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
        attribute = attribute.to(device)

        optimizer.zero_grad()
        score = model(image)
        # loss = criterion(score, category)
        loss = criterion(score, attribute)
        loss_list.append(loss.item())

        # pred = torch.argmax(score, dim=1)
        # correct = pred.eq(category)
        # acc_count += correct.sum().item()

        pred = torch.sigmoid(score).cpu().detach().numpy() > 0.5
        acc = f1_score(attribute.cpu().int().numpy(), pred, average='samples')  * attribute.shape[0]
        acc_count += acc
        
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
        attribute = attribute.to(device)

        score = model(image)
        # loss = criterion(score, category)
        loss = criterion(score, attribute)
        loss_list.append(loss.item())

        # pred = torch.argmax(score, dim=1)
        # correct = pred.eq(category)
        # acc_count += correct.sum().item()

        pred = torch.sigmoid(score).cpu().detach().numpy() > 0.5
        acc = f1_score(attribute.cpu().int().numpy(), pred, average='samples')  * attribute.shape[0]
        acc_count += acc

        total_count += image.shape[0]

    acc = acc_count / total_count * 100
    return sum(loss_list) / len(loss_list), acc

def run(max_epochs=100):
    if torch.cuda.is_available():
        device = torch.device('cuda:1')
    else:
        device = torch.device('cpu')
    print('Device:', device)

    model = get_model().to(device)
    # model = get_attr_model().to(device)
    train_data, val_data, test_data = get_data()

    criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = torch.optim.AdamW(params, lr=1e-4)
    optimizer = torch.optim.AdamW(params, lr=1e-3)

    t = time.time()
    for epoch in range(1, max_epochs + 1):
        train_loss, train_acc = train(model, train_data, epoch, criterion, optimizer, device)
        test_loss, test_acc = test(model, val_data, criterion, device)
        print('epoch', epoch, 'Train acc', '{:.04}'.format(train_acc), \
            'Train loss', '{:.04}'.format(train_loss))
        print('epoch', epoch, 'Test acc', '{:.04}'.format(test_acc), \
            'Test loss', '{:.04}'.format(test_loss))

if __name__ == '__main__':
    # run()
    model = get_joint_model()
    
    x = torch.rand(64, 3, 224, 224)
    out = model(x)
    print('out', out)
    