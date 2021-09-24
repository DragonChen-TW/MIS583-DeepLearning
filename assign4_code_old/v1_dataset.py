import csv
from PIL import Image

import torch
from torchvision import transforms
from torchvision.datasets import VisionDataset

def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class DeepFashion(VisionDataset):
    def __init__(self, csv_file, mode, transform):
        self.mode = mode # train, test and val
        self.transform = transform
#         self.target_transform = target_transform

        img_list = []
        cate_list = []
        attr_list = []
        with open(csv_file) as f:
            reader = csv.DictReader(f)
            for line in reader:
                img_list.append(line['file_path'])
                if mode != 'test':
                    cate = int(line['category_label'])
                    cate_list.append(cate)

                    attrs = tuple(line.values())[2:]
                    attrs = [int(a) for a in attrs]
                    attr_list.append(attrs) # save using tuple
        self.img_list = img_list
        self.cate_list = cate_list
        self.attr_list = attr_list

    def __getitem__(self, idx):
        path = self.img_list[idx]
        img = pil_loader(path)
        if self.transform:
            img = self.transform(img)

        if self.mode != 'test':
            category = self.cate_list[idx]
            category = torch.tensor(category)
            attribute = torch.tensor(self.attr_list[idx])
            return img, category, attribute
        else:
            return img

    def __len__(self):
        return len(self.img_list)

def get_dataset():
    trans_train = transforms.Compose([
        transforms.Resize(240),
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    trans_test = transforms.Compose([
        transforms.Resize(240),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dataset = DeepFashion('deep_fashion/train.csv', 'train', trans_train)
    val_dataset = DeepFashion('deep_fashion/val.csv', 'val', trans_test)
    test_dataset = DeepFashion('deep_fashion/test.csv', 'test', trans_test)

    return train_dataset, val_dataset, test_dataset

if __name__ == '__main__':
    train_dataset, val_dataset, test_dataset = get_dataset()

    print(len(train_dataset))
    print(len(val_dataset))
    print(len(test_dataset))
