import torch
from dnn.typing_definitions import *
import dnn.data_utils.transforms.transform as trsfrm
from dnn.data_utils.custom_ORIGINAL_FOLDER import CustomImageFolder
import pandas as pd


class ImageDataSet(CustomImageFolder):
    def __init__(self, root: DirPath, transform: trsfrm.Transform):
        super(ImageDataSet, self).__init__(root, transform)
        self.images = self.imgs

    def __getitem__(self, index):
        path, target = self.images[index]
        sample = self.loader(path)
        img_name = self.images[index][0].split('\\')[-1]

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, img_name


class ImageTabolarDataSet(ImageDataSet):
    def __init__(self, root: DirPath, transform: trsfrm.Transform, tabolar_data_path: DirPath):
        super(ImageTabolarDataSet, self).__init__(root, transform)
        # Index column should be the names of images
        self.tabolar_data_df = pd.read_csv(tabolar_data_path, index_col='image_name')

    def __getitem__(self, index):
        path, target = self.images[index]
        sample = self.loader(path)
        img_name = self.images[index][0].split('\\')[-1]
        tabolar_data = self.tabolar_data_df.loc[img_name].values

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, img_name, tabolar_data
