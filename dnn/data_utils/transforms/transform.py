from dnn.typing_definitions import *
import random
import cv2
from torchvision import transforms
# from dnn.data_utils.transforms.brightness_contrast_transform import BrightnessContrastTransform


class Transform(object):
    """
    Compose all transformers
    """
    def __init__(self, aug_params, crop_size, mean_imgs, std_imgs, p=0, clahe=True):
        self.aug_params = aug_params
        self.crop_size = crop_size
        self.mean = mean_imgs
        self.std = std_imgs
        self.p = p  # probability
        self.clahe = clahe

    def __call__(self, img: Image):
        h_flip, v_flip = self._rand_flip()
        brightness, contrast, degree = self.get_aug_values()
        transformed_imgs = []

        transformed_img = self.transform_per_image(img, h_flip=h_flip, v_flip=v_flip, degree=degree,
                                                   brightness=brightness, contrast=contrast)
        return transformed_img

    def get_aug_values(self):
        brightness, contrast, degree = 0, 1, 0
        if random.random() < self.p:
            degree = random.random() * self.aug_params['degree_max']
            brightness_min, brightness_max = self.aug_params['brightness_min'], self.aug_params['brightness_max']
            contrast_min, contrast_max = self.aug_params['contrast_min'], self.aug_params['contrast_max']
            if self.clahe:  # the params could be changed
                contrast_min, contrast_max = self.aug_params['clahe_contrast_min'], self.aug_params['clahe_contrast_max']
            brightness = (brightness_max - brightness_min) * random.random() + brightness_min
            contrast = (contrast_max - contrast_min) * random.random() + contrast_min
        return brightness, contrast, degree

    def _rand_flip(self):
        h_flip, v_flip = False, False
        if random.random() <= self.p:
            h_flip = True
        if random.random() <= self.p:
            v_flip = True
        return h_flip, v_flip

    def transform_per_image(self, img, h_flip=False, v_flip=False, degree=0, brightness=0, contrast=1):
        # img = np.asarray(img, dtype="float32")
        if self.clahe and len(img.shape)==2:
            clahe_obg = cv2.createCLAHE(clipLimit=4, tileGridSize=(12, 12))
            img = clahe_obg.apply(img)
        composed_transform_per_img = transforms.Compose([
            # torchvision.transforms.ColorJitter
            # BrightnessContrastTransform(brightness_factor=brightness, contrast_factor=contrast),
            # transforms.ToPILImage(),
            # transforms.RandomHorizontalFlip(p=h_flip),
            # transforms.RandomVerticalFlip(p=v_flip),
            # transforms.RandomRotation(degrees=(degree, degree)),  # should be (degree_min, degree_max)
            transforms.Resize(self.crop_size),
            # transforms.CenterCrop(self.crop_size),
            transforms.ToTensor(),
            # transforms.Normalize(mean=self.mean, std=self.std)
        ])
        return composed_transform_per_img(img)
