import os
import numpy as np
import cv2
import tensorflow as tf

class SequenceGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, base_path, seq_length, img_size, classes, batch_size=32, shuffle=True):
        self.df = df
        self.base_path = base_path
        self.seq_length = seq_length
        self.img_size = img_size
        self.classes = classes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.df) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

    def __getitem__(self, index):
        start_index = index * self.batch_size
        end_index = (index + 1) * self.batch_size
        batch_df = self.df.iloc[start_index:end_index]

        X = np.empty((self.batch_size, self.seq_length, *self.img_size, 3), dtype=np.float32)
        Y = np.empty((self.batch_size, self.seq_length, self.classes), dtype=np.float32)

        for i, (_, row) in enumerate(batch_df.iterrows()):
            for t in range(self.seq_length):
                path_col = f'path_{t}'
                full_path = os.path.join(self.base_path, row[path_col].lstrip('/'))

                img = cv2.imread(full_path)
                if img is None:
                    img_final = np.zeros((*self.img_size, 3), dtype=np.float32)
                else:
                    img_final = cv2.resize(img, self.img_size)

                X[i, t] = img_final.astype(np.float32) / 255.0

            labels = [row[f'class_{t}'] for t in range(self.seq_length)]
            Y[i] = tf.keras.utils.to_categorical(labels, num_classes=self.classes)

        return X, Y
