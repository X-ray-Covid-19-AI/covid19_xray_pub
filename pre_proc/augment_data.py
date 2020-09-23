import numpy as np

import pandas as pd

import os
from PIL import Image
import imgaug.augmenters as iaa

# define your own local path here
DATA_ROOT_PATH = "C:\\Users\\Alacrity\\Documents\\Side Projects\\CovidXray\\data\\coronavirus_task"
positive_path = "corona"
negative_path = "normal"

train_path = "train"
test_path = "test"

##### Define (light) augemntations

BRIGHTEN = iaa.Multiply((0.8, 1.4)) # make upper edge larger to get brighter images
SHARPEN = iaa.Sharpen(alpha=(0, 0.82)) # the larger alpha, the larger the effect. alpha <= 1

GAMMA_CONTRAST = iaa.GammaContrast((0.6, 1.3))
# param = Exponent for the contrast adjustment. Higher values darken the image. They seem to also remove
# some of the white clouds so careful with that

SHEAR = iaa.Affine(shear=(-10, 10))
ROTATE = iaa.Affine(rotate=(-10, 10))

SUBTLE_CLAHE = iaa.CLAHE(clip_limit=(0.5,2), tile_grid_size_px = (6, 10))
# previous, less pronounced : iaa.CLAHE(clip_limit=(0.5,2), tile_grid_size_px = (5, 8))

GBLUR = iaa.GaussianBlur((0,1.9)) # Values in the range 0.0 (no blur) to 3.0 (strong blur) are common.

SCALE = iaa.Affine(scale={"x": (0.9, 1.1), "y": (0.9, 1.1)})

AUG_NAMES = ['brighten', 'gamma_contrast', 'clahe', 'rotate', 'shear', 'scale', 'fliplr', 'gblur', 'sharpen']
######

def clean_dots(path):
    split_path = path.split(".")
    return path

def get_full_image_paths(path, im_ending=".tiff"):
    im_paths = []
    im_names = os.listdir(path)
    im_names = [name for name in im_names if (name.endswith(im_ending))]
    im_paths += [os.path.join(path, name) for name in im_names]
    #im_paths = [clean_dots(path) for path in im_paths]
    return im_paths, im_names

def load_data(path, im_ending='.tiff'):
    paths, names = get_full_image_paths(path, im_ending=im_ending)
    images = np.array([np.array(Image.open(img).convert('L')) for img in paths])
    names = [name.strip(im_ending) for name in names]
    return images, names

def get_heavy_augs():
    """
    Get heavier augmentations
    """
    aug_seq = iaa.Sequential([
        iaa.Sometimes(0.6, BRIGHTEN),
        iaa.Sometimes(0.4, SUBTLE_CLAHE),
        iaa.Sometimes(0.41, iaa.Affine(rotate=(-8, 8))),
        iaa.Sometimes(0.41, iaa.Affine(shear=(-8, 8))),
        iaa.OneOf([
            GBLUR,  # blur images with a sigma between 0 and 3.0
            SHARPEN
        ]),
    ], random_order=True)
    return aug_seq

def get_augs():
    """
    get augmentation sequence
    the augmentation parameters are defined as global variables here so they can easily be altered from outside
    the probabilities correspond to the probability of the augmentation happening
    """
    aug_seq = iaa.Sequential([
        iaa.Sometimes(0.4, BRIGHTEN),
        iaa.Sometimes(0.3, GAMMA_CONTRAST),
        iaa.Sometimes(0.4, SUBTLE_CLAHE),
        iaa.Sometimes(0.4, ROTATE),
        iaa.Sometimes(0.4, SHEAR),
        iaa.Sometimes(0.4, SCALE),
        iaa.Fliplr(0.5),
        iaa.OneOf([
            GBLUR,
            SHARPEN,
        ]),
    ], random_order=True)
    return aug_seq

def get_augs_done(image):
    """
    get augmentation sequence
    the augmentation parameters are defined as global variables here so they can easily be altered from outside
    the probabilities correspond to the probability of the augmentation happening
    """
    if image is None:
        return image, image

    aug_seq = [('prob', 0.4, BRIGHTEN),
               ('prob', 0.3, GAMMA_CONTRAST),
               ('prob', 0.4, SUBTLE_CLAHE),
               ('prob', 0.4, ROTATE),
               ('prob', 0.4, SHEAR),
               ('prob', 0.4, SCALE),
               ('prob', 0.5, iaa.Fliplr()),
               ('oneof',GBLUR,SHARPEN)
               ]

    augs_done = [0]*(len(aug_seq) + 1)

    for k, aug_tuple in enumerate(aug_seq):
        if aug_tuple[0] == 'prob':
            p = np.random.random()
            aug = aug_tuple[2]
            if p < aug_tuple[1]:
                image = aug(image=image)
                augs_done[k] = 1
        elif aug_tuple[0] == 'oneof':
            idx = np.random.choice(2)
            aug = aug_tuple[idx+1]
            image = aug(image=image)
            augs_done[-idx] = 1
    return image, augs_done

def augment_separately(aug_function, images, names, iter):
    """
    Use this option for images of different sizes

    images = list of images to augment
    names = file names of these images
    iter = iteration number (1,2,3..) or name (A, B, C..) - as we do the augmentation multiple times

    """
    new_images = [None] * (len(images))
    new_names = [None]*(len(names))
    augs_done = []
    for k, im in enumerate(images):
        new_images[k], curr_augs_done = aug_function(im)
        augs_done.append(curr_augs_done)
        new_names[k] = add_iter_to_name_new(names[k],str(iter))

    augs_done_df = pd.DataFrame(augs_done, columns=AUG_NAMES)
    augs_done_df['im_names'] = new_names # happens AFTER we add iter to all the names

    return new_images, new_names, augs_done_df

def filter_small(data, MINSIZE=900):
    """
    Small images are more harshly affected by the augmentations -
    only keep images larger than (MINSIZE, MINSIZE)
    """
    return [k for k in data if (k.shape[0] > MINSIZE and k.shape[1] > MINSIZE)]

def augment_and_save(aug_function, data_path, aug_output_folder, orig_output_folder, MINSIZE=900, im_ending=".tiff"):
    """
    For augmenting BEFORE training
    augments all images (larger than MINSIZE) in path, and save them into aug_output_folder
    save originals into orig_output_folder for debugging - we can remove this later
    """
    ITERS = ['A', 'B', 'C', 'D']

    images, names = load_data(data_path, im_ending=im_ending) # names are relative to the data path

    all_new_names = []

    for iter in ITERS:
        new_images, new_names, augs_done = augment_separately(aug_function, images, names, iter)
        augs_done.to_csv(os.path.join(aug_output_folder, 'augs_done.csv'))
        all_new_names += new_names
        for k, npim in enumerate(new_images):
            im = Image.fromarray(npim)
            name = new_names[k]
            im.save(os.path.join(aug_output_folder, name + im_ending))
            #orig_im = Image.fromarray(data[k])
            #orig_im.save(os.path.join(orig_output_folder, 'orig_' + str(k) +  '_' + iter + '.png'))
    return all_new_names

def augment_to_dict(aug_function, data_path, im_ending=".tiff", ITERS = ['A', 'B', 'C', 'D']):
    images, names = load_data(data_path, im_ending=im_ending)  # names are relative to the data path

    all_new_images = {}
    all_augs_done_df = None

    for iter in ITERS:
        new_images, new_names, augs_done = augment_separately(aug_function, images, names, iter)
        if all_augs_done_df is not None:
            all_augs_done_df = pd.concat([all_augs_done_df, augs_done])
        else:
            all_augs_done_df = augs_done
        for k, npim in enumerate(new_images):
            all_new_images[new_names[k]]  = new_images[k]

    return all_new_images, all_augs_done_df

def augment_images(aug_function, images, names, ITERS = ['A', 'B', 'C', 'D']):
    all_new_images = {}
    all_augs_done_df = None

    for iter in ITERS:
        new_images, new_names, augs_done = augment_separately(aug_function, images, names, iter)
        if all_augs_done_df is not None:
            all_augs_done_df = pd.concat([all_augs_done_df, augs_done])
        else:
            all_augs_done_df = augs_done
        for k, npim in enumerate(new_images):
            all_new_images[new_names[k]]  = new_images[k]

    return all_new_images, all_augs_done_df


def create_new_folders_within(root_path):
    """
    create a new folder within root_path that will contain the newly augmented images
    """
    aug_folder = os.path.join(root_path, "augmented")
    if not os.path.exists(aug_folder):  os.mkdir(aug_folder)
    orig_folder = os.path.join(root_path, "original")
    if not os.path.exists(orig_folder):  os.mkdir(orig_folder)
    return aug_folder, orig_folder

def aug_generator(features, labels, batch_size, IMSIZE = 1024):
    """
    generates augmented images
    can be used with keras fit_generator()
    """
    aug = get_augs()
    batch_features = np.zeros((batch_size,IMSIZE , IMSIZE, 1))
    batch_labels = np.zeros((batch_size,1))
    while True:
        for i in range(batch_size):
            index= np.random.choice(len(features),1)
            batch_features[i] = aug(image = features[index])
            batch_labels[i] = labels[index]
    yield batch_features, batch_labels

def add_iter_to_name(filename, iter):
    filename_split = filename.split(".")
    if len(filename_split) == 1: # no ending
        filename, ending = filename_split[0], ""
    elif len(filename_split) == 2: # with ending
        filename, ending = filename_split
        ending =  "." + ending
    else: # multiple dots - odd
        # print("got error in filename, misplaced dot", filename)
        filename, ending =  ".".join(filename_split[:-1]), filename_split[-1]
        ending =  "." + ending
    return filename + iter + ending

def add_iter_to_name_new(filename, iter):
    return filename + iter

if __name__ == '__main__':

    aug_fun = get_augs_done
    data_path = os.path.join(DATA_ROOT_PATH, train_path, positive_path)
    aug_output_path, orig_output_path = create_new_folders_within(data_path)
    im_dict, augs_done = augment_to_dict(aug_fun, data_path=data_path, im_ending=".png")
    print(augs_done)



