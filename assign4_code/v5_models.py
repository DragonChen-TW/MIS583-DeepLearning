import torch
from torch import nn
from torchvision import models

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

# class JointResNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         m = models.resnext50_32x4d(pretrained=True)
#         in_features = m.fc.in_features
#         m.fc = nn.Sequential()
#         for param in m.parameters():
#             param.requires_grad = False
#         # for param in m.layer3[-2:].parameters():
#         #     param.requires_grad = True
#         for param in m.layer4[-2:].parameters():
#             param.requires_grad = True
#         self.features = m
        
#         self.fc_cate = nn.Linear(in_features, 10)
#         self.fc_attr = nn.Linear(in_features, 15)
    
#     def forward(self, x):
#         features = self.features(x)
#         out_cate = self.fc_cate(features)
#         out_attr = self.fc_attr(features)
#         return out_cate, out_attr
    
def get_joint_resnet():
    model = JointResNet()
    return model

class JointResNet(nn.Module):
    def __init__(self, model, fc_dim=512):
        super().__init__()
        m = model
        in_features = m.fc.in_features
        m.fc = nn.Sequential()
        self.features = m

        self.shared = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(in_features),
            nn.Dropout(),
            nn.Linear(in_features, in_features),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(in_features),
            nn.Dropout(),
        )
        # for param in m.parameters():
        #     param.requires_grad = False
        # for param in m.layer3[-2:].parameters():
        #     param.requires_grad = True
        
        self.fc_cate = nn.Sequential(
            nn.Linear(in_features, fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(fc_dim, 10),
        )
        self.fc_attr = nn.Sequential(
            nn.Linear(in_features, fc_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(fc_dim, 15),
        )
    
    def forward(self, x):
        features = self.features(x)
        shared = self.shared(features)
        out_cate = self.fc_cate(shared)
        out_attr = self.fc_attr(shared)
        return out_cate, out_attr

def get_joint_resnet18():
    m = models.resnet18(pretrained=True)
    model = JointResNet(m, fc_dim=256)
    return model

def get_joint_resnet34():
    m = models.resnet34(pretrained=True)
    model = JointResNet(m, fc_dim=256)
    return model

def get_joint_resnet152():
    m = models.resnet152(pretrained=True)
    model = JointResNet(m)
    return model

def get_joint_resnext50():
    m = models.resnext50_32x4d(pretrained=True)
    model = JointResNet(m)
    return model

def get_joint_resnext101():
    m = models.resnext101_32x8d(pretrained=True)
    model = JointResNet(m)
    return model

class JointDenseNet(nn.Module):
    def __init__(self, model, fc_dim=512):
        super().__init__()
        m = model
        in_features = m.classifier.in_features
        m.classifier = nn.Sequential()
        for param in m.parameters():
            param.requires_grad = False
        # for param in  m.features.denseblock4.parameters():
        #     param.requires_grad = True
        self.features = m
        
        # self.fc_cate = nn.Linear(in_features, 10)
        self.fc_cate = nn.Sequential(
            nn.Linear(in_features, fc_dim),
            nn.Dropout(),
            nn.BatchNorm1d(fc_dim),
            nn.Linear(fc_dim, 10),
        )
        # self.fc_attr = nn.Linear(in_features, 15)
        self.fc_attr = nn.Sequential(
            nn.Linear(in_features, fc_dim),
            nn.Dropout(),
            nn.BatchNorm1d(fc_dim),
            nn.Linear(fc_dim, 15),
        )
    
    def forward(self, x):
        features = self.features(x)
        out_cate = self.fc_cate(features)
        out_attr = self.fc_attr(features)
        return out_cate, out_attr

def get_joint_densenet121():
    m = models.densenet121(pretrained=True)
    model = JointDenseNet(m, fc_dim=128)
    return model

def get_joint_densenet169():
    m = models.densenet169(pretrained=True)
    model = JointDenseNet(m, fc_dim=128)
    return model

# class F1LossOld(nn.Module):
#     # Ref: https://gist.github.com/SuperShinyEyes/dcc68a08ff8b615442e3bc6a9b55a354
#     def __init__(self, epsilon=1e-7):
#         super().__init__()
#         self.epsilon = 1e-7

#     def forward(self, true, pred):
#         # pred = (torch.sigmoid(pred) > 0.5).int()
#         pred = torch.sigmoid(pred)

#         tp = (true * pred).sum(dim=1)
#         tn = ((1 - true) * (1 - pred)).sum(dim=1)
#         fp = ((1 - true) * pred).sum(dim=1)
#         fn = (true * (1 - pred)).sum(dim=1)

#         precision = tp / (tp + fp + self.epsilon)
#         recall = tp / (tp + fn + self.epsilon)

#         f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)
#         f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)
#         return 1 - f1.mean()

class F1Loss(nn.Module):
    # Ref: https://gist.github.com/SuperShinyEyes/dcc68a08ff8b615442e3bc6a9b55a354
    def __init__(self, device, epsilon=1e-7):
        super().__init__()
        self.epsilon = 1e-7
        # self.preds = torch.empty(0).to(device)
        # self.trues = torch.empty(0).to(device)

    def forward(self, trues, preds):
        preds = (torch.sigmoid(preds) > 0.5).int()
        # pred = torch.sigmoid(pred)

        # preds = torch.cat((self.preds, preds))
        # trues = torch.cat((self.trues, trues))
        # self.preds = preds
        # self.trues = trues

        tp = (trues * preds).sum(dim=1)
        tn = ((1 - trues) * (1 - preds)).sum(dim=1)
        fp = ((1 - trues) * preds).sum(dim=1)
        fn = (trues * (1 - preds)).sum(dim=1)

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




