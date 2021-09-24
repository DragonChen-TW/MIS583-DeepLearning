from random import Random

from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from v1_dataset import get_dataset

class Partition:
    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        data_idx = self.index[idx]
        return self.data[data_idx]

class DataPartitioner:
    def __init__(self, data, sizes=[1], seed=1340):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)

        data_len = len(data)
        indexes = list(range(data_len))
        rng.shuffle(indexes)

        for part in sizes:
            part_len = int(part * data_len)
            self.partitions.append(indexes[0: part_len])
            indexes = indexes[part_len:]

    def use(self, rank):
        return Partition(self.data, self.partitions[rank])

def get_deep_fashion(rank, size):
    train_dataset, val_dataset, _ = get_dataset()

    batch_size = 63
    num_workers = 8

    batch_size_part = int(batch_size / size)
    partition_sizes = [1.0 / size for _ in range(size)]
    paritition = DataPartitioner(train_dataset, partition_sizes)
    paritition = paritition.use(rank)

    train_data = DataLoader(dataset=paritition,
                            batch_size=batch_size_part,
                            num_workers=num_workers,
                            shuffle=True)
    
    paritition = DataPartitioner(val_dataset, partition_sizes)
    paritition = paritition.use(rank)

    val_data = DataLoader(dataset=paritition,
                            batch_size=batch_size_part,
                            num_workers=num_workers)

    print('train data shape', next(iter(train_data))[0].shape)
    print('val data shape', next(iter(val_data))[0].shape)

    return train_data, val_data

if __name__ == '__main__':
    data = get_deep_fashion(0, 3)
