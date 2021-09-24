import os, time
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, models

import ray
from ray.util.sgd import TorchTrainer
from ray.util.sgd.torch import TrainingOperator
from ray.util.sgd.utils import BATCH_SIZE, NUM_SAMPLES
# 
from v1_dataset import get_dataset

def model_creator(config):
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
    return model

def data_creator(config):
    train_dataset, val_dataset, _ = get_dataset()
    train_data = DataLoader(train_dataset, batch_size=config[BATCH_SIZE],
                num_workers=config['worker'] or 16, pin_memory=True,
                shuffle=True)
    val_data = DataLoader(val_dataset, batch_size=config[BATCH_SIZE],
                num_workers=config['worker'] or 16, pin_memory=True)
    return train_data, val_data

def optimizer_creator(model, config):
    for param in model.parameters():
        param.requires_grad = False

    for param in list(model.features[-1].parameters()) + list(model.classifier.parameters()):
        param.requires_grad = True

    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=config['lr'])

def loss_creator(config):
    return nn.CrossEntropyLoss()

def scheduler_creator(optimizer, config):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

class MyOperator(TrainingOperator):
    def setup(self, config):
        pass
    
    def train_batch(self, batch, batch_info):
        model = self.model
        optimizer = self.optimizer
        criterion = self.criterion
        
        features, category, attribute = batch
        print('batch', features.shape)
        if self.use_gpu:
            features = features.cuda(non_blocking=True)
            category = category.cuda(non_blocking=True)
            attribute = attribute.cuda(non_blocking=True)
        
        with self.timers.record("fwd"):
            output = model(features)
            loss = criterion(output, category)
        
        with self.timers.record("grad"):
            optimizer.zero_grad()
            if self.use_fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
        
        with self.timers.record("apply"):
            optimizer.step()

        return {"train_loss": loss.item(), NUM_SAMPLES: features[0].size(0)}
    
    def validate_batch(self, batch, batch_info):
        model = self.model
        criterion = self.criterion

        features, category, attribute = batch
        if self.use_gpu:
            features = features.cuda(non_blocking=True)
            category = category.cuda(non_blocking=True)
            attribute = attribute.cuda(non_blocking=True)
        
        with self.timers.record("fwd"):
            output = model(features)
            loss = criterion(output, category)

        with self.timers.record("eval_fwd"):
            output = model(features)
            loss = criterion(output, category)
            _, predicted = torch.max(output.data, 1)

        num_correct = (predicted == category).sum().item()
        num_samples = category.shape[0]
        return {
            "val_loss": loss.item(),
            "val_accuracy": num_correct / num_samples,
            NUM_SAMPLES: num_samples
        }

if __name__ == '__main__':
    ray.init(num_cpus=48, num_gpus=3)
    print('gpu_id', ray.get_gpu_ids())
    print(ray.available_resources())
    trainer = TorchTrainer(
        model_creator=model_creator,
        data_creator=data_creator,
        optimizer_creator=optimizer_creator,
        loss_creator=loss_creator,
        scheduler_creator=scheduler_creator,
        training_operator_cls=MyOperator,

        scheduler_step_freq='epoch',  # if scheduler is used
        config={'lr': 3e-5, BATCH_SIZE: 64, 'worker': 8},
        num_workers=3,
        use_gpu=True,
        use_tqdm=True,
    )
    
    max_epochs = 2
    for epoch in range(1, max_epochs + 1):
        t = time.time()
        metrics = trainer.train()
        val_metrics = trainer.validate()
        print(metrics)
        print(val_metrics)
        print('-' * 30, 'Cost', time.time() - t, '-' * 30)

