import cv2
import nibabel as nib
import matplotlib.pyplot as plt
import os
import numpy as np
import keras
# Change the path to your path
import tensorflow as tf

path = 'data/ours/mni_tissues_APN27.nii.gz'
TRAIN_DATASET_PATH = 'D:\\Projects\\morpho\\mni_macrostructures\\train'
train_and_val_directories = [f.path for f in os.scandir(TRAIN_DATASET_PATH) if f.is_dir()]
# train_and_val_directories = train_and_val_directories.split('/')[-1]
# x = TRAIN_DATASET_PATH+'train'
# print(x)
train_and_val_directories = [x.replace(f'{TRAIN_DATASET_PATH}', '') for x in train_and_val_directories]
IMG_SIZE = 127

SEGMENT_CLASSES = {
    0 : 'bg',
    1 : 'left', # or NON-ENHANCING tumor CORE
    2 : 'right',
    3 : 'temich' # original 4 -> converted into 3 later
}

# there are 155 slices per volume
# to start at 5 and use 145 slices means we will skip the first 5 and last 5
VOLUME_SLICES = 100
VOLUME_START_AT = 22

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, dim=(IMG_SIZE, IMG_SIZE), batch_size=1, n_channels=2, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        Batch_ids = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(Batch_ids)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, Batch_ids):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.zeros((self.batch_size * VOLUME_SLICES, *self.dim, self.n_channels))
        y = np.zeros((self.batch_size * VOLUME_SLICES, 240, 240))
        Y = np.zeros((self.batch_size * VOLUME_SLICES, *self.dim, 4))

        # Generate data
        for c, i in enumerate(self.list_IDs):
            case_path = os.path.join(TRAIN_DATASET_PATH, i)

            #             data_path = os.path.join(case_path, f'{i}_flair.nii');
            #             flair = nib.load(data_path).get_fdata()

            data_path = os.path.join(case_path, f'mni_macrostructures_{i}.nii.gz');
            ce = nib.load(data_path).get_fdata()

            data_path = os.path.join(case_path, f'mni_t1_{i}.nii.gz');
            seg = nib.load(data_path).get_fdata()

            for j in range(VOLUME_SLICES):
                X[j + VOLUME_SLICES * c, :, :, 0] = cv2.resize(ce[:, :, j + VOLUME_START_AT], (IMG_SIZE, IMG_SIZE));
                #                  X[j +VOLUME_SLICES*c,:,:,1] = cv2.resize(ce[:,:,j+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE));

                y[j + VOLUME_SLICES * c] = seg[:, :, j + VOLUME_START_AT];

        # Generate masks
        y[y == 4] = 3;
        mask = tf.one_hot(y, 4);
        Y = tf.image.resize(mask, (IMG_SIZE, IMG_SIZE));
        return X / np.max(X), Y


training_generator = DataGenerator(train_and_val_directories[0])
training_generator.__getitem__(1)
valid_generator = DataGenerator(train_and_val_directories[1])