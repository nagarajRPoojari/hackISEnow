import tensorflow as tf
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler
import random

scaler = MinMaxScaler()
import nibabel as nib
import numpy as np
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt


def predict_for(flair, t2, t1ce, model):
    flair = scaler.fit_transform(flair.reshape(-1, flair.shape[-1])).reshape(
        flair.shape
    )
    t1ce = scaler.fit_transform(t1ce.reshape(-1, t1ce.shape[-1])).reshape(t1ce.shape)
    t2 = scaler.fit_transform(t2.reshape(-1, t2.shape[-1])).reshape(t2.shape)
    temp_combined_images = np.stack([flair, t1ce, t2], axis=3)
    temp_combined_images = temp_combined_images[56:184, 56:184, 13:141]

    inp = np.expand_dims(temp_combined_images, axis=0)

    ##predicting

    res = model.predict(inp)
    print(np.unique(res))
    return res
