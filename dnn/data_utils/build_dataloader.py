import torch.utils.data as data
import torch
import time
from dnn.typing_definitions import *
import torchvision
from dnn.data_utils.custom_dataset import ImageTabolarDataSet, ImageDataSet
from dnn.data_utils.transforms.transform import Transform
from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np


def build_dataloader(data_params: dict, data_set: str, is_tabolar_mode: str, tabolar_data_path: DirPath = None) \
        -> data.DataLoader:

    # Initialize params to create a DataLoader object
    dataloader_params = data_params[f'loader_params_{data_set}']
    if data_set == 'train':
        dataset_root_path = data_params['train_dataset_path']
    elif data_set == 'test':
        dataset_root_path = data_params['test_dataset_path']
    else:
        print("Invalid input for parameter mode!")

    if is_tabolar_mode == 'yes':
        # TODO: Change labels !!!
        dataset = ImageTabolarDataSet(root=dataset_root_path,
                                      transform=Transform(aug_params=data_params['aug_params'],
                                                          crop_size=data_params['image_size'],
                                                          mean_imgs=data_params['mean'],
                                                          std_imgs=data_params['std'],
                                                          p=data_params['aug_prob'],
                                                          clahe=data_params['clahe']),
                                      tabolar_data_path=tabolar_data_path)
    else:
        dataset = ImageDataSet(root=dataset_root_path,
                                transform=Transform(aug_params=data_params['aug_params'],
                                                    crop_size=data_params['image_size'],
                                                    mean_imgs=data_params['mean'],
                                                    std_imgs=data_params['std'],
                                                    p=data_params['aug_prob'],
                                                    clahe=data_params['clahe']))

    if data_set == 'train':
        class_sample_count = np.array([len(np.where(dataset.targets == t)[0]) for t in np.unique(dataset.targets)])
        weights = 1. / class_sample_count
        sample_weights = torch.from_numpy(weights[dataset.targets])
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

        dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                                 batch_size=dataloader_params['batch_size'],
                                                 num_workers=dataloader_params['num_workers'],
                                                 shuffle=dataloader_params['shuffle'],
                                                 sampler=sampler)
    else:
        dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                                 batch_size=dataloader_params['batch_size'],
                                                 num_workers=dataloader_params['num_workers'],
                                                 shuffle=dataloader_params['shuffle']
                                                 )

    return dataloader
