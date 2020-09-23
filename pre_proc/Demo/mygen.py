import pandas as pd
import keras
import numpy as np
from skimage import morphology,io, color, exposure, img_as_float, transform
from skimage.color import rgb2gray
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(process)d - %(asctime)s - %(pathname)s - %(levelname)s - %(message)s')
class myGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, csv_file,path,log_name,pred_dim,original_size=(1024,1024),
                 batch_size=1, n_channels=1,
                  shuffle=False):
        'Initialization'
        ch = logging.FileHandler(log_name)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        self.df = pd.read_csv(csv_file)

        self.dim = pred_dim
        self.original_size = original_size
        self.path = path
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle

        self.on_epoch_end()


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.df.shape[0] / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        files = [self.path +'/' + self.df.iloc[i][0] for i in indexes]

         # Generate data
        originals, minimized = self.__data_generation(files)

        return originals, minimized

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.df.shape[0])
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, files):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        originals = []
        cropped = []
        

        # Generate data
        for i in files:
            logger.debug("converting file {}".format(i))
            img = img_as_float(io.imread(i))
            img = rgb2gray(img)
            original = np.asarray(np.expand_dims(transform.resize(img, self.original_size), -1))
            originals.append(original)

            #print("shape is {}".format(original.shape))
            img = transform.resize(img, self.dim)
            img = exposure.equalize_hist(img)
            img = np.expand_dims(img, -1)
            img = np.asarray(img)
            # convert from integers to floats
            img = img.astype('float32')
            # calculate global mean and standard deviation
            mean, std = img.mean(), img.std()
            img = (img - mean) / std

            cropped.append(img)
    


        return np.array(originals),np.array(cropped)
