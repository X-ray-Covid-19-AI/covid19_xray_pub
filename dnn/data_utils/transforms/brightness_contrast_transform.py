from typing_definitions import *


class BrightnessContrastTransform(object):
    def __init__(self, brightness_factor, contrast_factor):
        self.brightness_factor = brightness_factor
        self.contrast_factor = contrast_factor

    def __call__(self, img: Image):
        img_transformed = img * self.contrast_factor + self.brightness_factor
        return img_transformed
