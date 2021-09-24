import torch
from torch import nn
from torchvision import models

def get_squeezenet(output_type):
    if output_type == 'cate':
        out_classes = 10
    else:
        out_classes = 15
    model = models.squeezenet1_1(pretrained=True)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5, inplace=False),
        nn.Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1)),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Conv2d(64, out_classes, kernel_size=(1, 1), stride=(1, 1)),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(output_size=(1, 1)),
    )
    for param in model.parameters():
        param.requires_grad = False

    for param in list(model.features[-1].parameters()) + list(model.classifier.parameters()):
        param.requires_grad = True
    return model

class JointSqueezeNet(nn.Module):
    # there are some error, cannot run normally now
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

def get_joint_squeezenet():
    model = JointSqueezeNet()
    return model

def get_resnet(output_type):
    if output_type == 'cate':
        out_classes = 10
    else:
        out_classes = 15
    model = models.resnext50_32x4d(pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, out_classes)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    return model

class JointResNet(nn.Module):
    def __init__(self):
        super().__init__()
        m = models.resnext50_32x4d(pretrained=True)
        in_features = m.fc.in_features
        m.fc = nn.Sequential()
        for param in m.parameters():
            param.requires_grad = False
        # for param in m.layer3[-2:].parameters():
        #     param.requires_grad = True
        for param in m.layer4[-2:].parameters():
            param.requires_grad = True
        self.features = m
        
        self.fc_cate = nn.Linear(in_features, 10)
        self.fc_attr = nn.Linear(in_features, 15)
    
    def forward(self, x):
        features = self.features(x)
        out_cate = self.fc_cate(features)
        out_attr = self.fc_attr(features)
        return out_cate, out_attr
    
def get_joint_resnet():
    model = JointResNet()
    return model

class JointResNext101(nn.Module):
    def __init__(self):
        super().__init__()
        m = models.resnext101_32x8d(pretrained=True)
        in_features = m.fc.in_features
        m.fc = nn.Sequential()
        for param in m.parameters():
            param.requires_grad = False
        # for param in m.layer3[-2:].parameters():
        #     param.requires_grad = True
        for param in m.layer4.parameters():
            param.requires_grad = True
        self.features = m
        
        self.fc_cate = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.Dropout(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 10),
        )
        self.fc_attr = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.Dropout(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 15),
        )
    
    def forward(self, x):
        features = self.features(x)
        out_cate = self.fc_cate(features)
        out_attr = self.fc_attr(features)
        return out_cate, out_attr

def get_joint_resnext101():
    model = JointResNext101()
    return model

class F1Loss(nn.Module):
    # Ref: https://gist.github.com/SuperShinyEyes/dcc68a08ff8b615442e3bc6a9b55a354
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = 1e-7

    def forward(self, true, pred):
        pred = (torch.sigmoid(pred) > 0.5).int()

        tp = (true * pred).sum(dim=1)
        tn = ((1 - true) * (1 - pred)).sum(dim=1)
        fp = ((1 - true) * pred).sum(dim=1)
        fn = (true * (1 - pred)).sum(dim=1)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
        f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)
        return 1 - f1.mean()

if __name__ == '__main__':
    import torch
    import time
    device = torch.device('cuda:1')
    x = torch.rand(64, 3, 224, 224).to(device)
    # m = get_squeezenet('cate')
    # print(m(x).shape)
    # m = get_squeezenet('attr')
    # print(m(x).shape)

    # m = get_joint_squeezenet()
    # print(m)

    # m = get_resnet('cate').to(device)
    # print(m(x).shape)
    # m = get_resnet('attr').to(device)
    # print(m(x).shape)

    m = get_joint_resnet().to(device)
    out = m(x)
    print(out[0].shape, out[1].shape)




