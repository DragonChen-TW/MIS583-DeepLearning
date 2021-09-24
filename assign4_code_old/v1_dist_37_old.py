import os, time
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'

from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, models

from v1_dataset import get_dataset

from ray.util.sgd.torch import TrainingOperator
from ray.util.sgd.torch.examples.train_example import LinearDataset

def model_creator

class MyTrainer(TrainingOperator):
    def setup(self, config):
        train_dataset, val_dataset, _ = get_dataset()

        batch_size = 64
        worker = 16
        train_data = DataLoader(train_dataset, batch_size=batch_size,
                    num_workers=worker, pin_memory=True,
                    shuffle=True)
        val_data = DataLoader(val_dataset, batch_size=batch_size,
                    num_workers=worker, pin_memory=True)

        # Setup model.
        model = models.squeezenet1_1(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False

        for param in model.features[-1].parameters():
            param.requires_grad = True

        # v2
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Conv2d(512, 64, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Conv2d(64, 10, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )

        # Setup optimizer.
        optimizer = torch.optim.AdamW(params, lr=3e-5)

        # Setup loss.
        criterion = nn.CrossEntropyLoss()

        # Setup scheduler.
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

        # Register all of these components with Ray SGD.
        # This allows Ray SGD to do framework level setup like Cuda, DDP,
        # Distributed Sampling, FP16.
        # We also assign the return values of self.register to instance
        # attributes so we can access it in our custom training/validation
        # methods.
        self.model, self.optimizer, self.criterion, self.scheduler = \
            self.register(models=model, optimizers=optimizer,
                          criterion=criterion,
                          schedulers=scheduler)
        self.register_data(train_loader=train_data, validation_loader=val_data)
