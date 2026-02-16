import os
import keras
import numpy as np
from PIL import Image

def one_hot_encoder_mask(num_class):
    def make_mask(mask):
        height, width, *_ = mask.shape
        shape = (height, width, num_class)
        
        mask_one_hot = np.zeros(shape)
        mask = mask.astype(np.int32)
        
        ind = np.indices(mask.shape[:2])
        xind = ind[0].reshape(-1)
        yind = ind[1].reshape(-1)
        zind = mask.reshape(-1)

        mask_one_hot[xind, yind, zind] = 1
        return mask_one_hot
    return make_mask


class SegmentationGenerator(keras.utils.Sequence):
    'Generador de datos para segmentación en Keras'
    def __init__(self, images_path, masks_path, num_classes, rescale=None, target_size=(512, 512), batch_size=1, shuffle=True):
        'Initialization'
        self.images_path = images_path
        self.masks_path = masks_path
        self.target_size = target_size
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.shuffle = shuffle
        # Almacena imágenes
        self.list_IDs = os.listdir(images_path)
        
        #One hot encoder
        self.one_hot_encoder = one_hot_encoder_mask(num_classes)
        
        self.rescale = rescale
        
        self.on_epoch_end()
         
        
    def __len__(self):
        '''Denotes the number of batches per epoch'''
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        images = [self.__open_image(self.images_path, im_ID) for im_ID in list_IDs_temp]
        
        if self.rescale:
            X = [im*self.rescale for im in images]
        
        # Máscaras
        y = [self.__open_image(self.masks_path, im_ID) for im_ID in list_IDs_temp]
        
        if self.num_classes > 2:
            # Si hay más de dos clases se hace codificación one-hot
            y = [self.one_hot_encoder(mask) for mask in y]
        else:
            # Añade dimensión de canales
            y = [mask.reshape((*mask.shape, 1)) for mask in y]
            
        X = np.stack(X, axis=0)
        y = np.stack(y, axis=0)
        return X, y
    
    def __open_image(self, path, im_id):
        image = Image.open(os.path.join(path, im_id))
        height, width = self.target_size
        image = image.resize((width, height),0)
        return np.array(image)

