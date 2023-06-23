import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import glob
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import cv2 as cv

scaler = MinMaxScaler()
import os
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")
import random
from tensorflow.keras.models import load_model
from matplotlib import cm

st.set_page_config(layout="wide")


def predict_for(flair, t2, t1ce, model):
    flair = scaler.fit_transform(flair.reshape(-1, flair.shape[-1])).reshape(
        flair.shape
    )
    t1ce = scaler.fit_transform(t1ce.reshape(-1, t1ce.shape[-1])).reshape(t1ce.shape)
    t2 = scaler.fit_transform(t2.reshape(-1, t2.shape[-1])).reshape(t2.shape)
    temp_combined_images = np.stack([flair, t1ce, t2], axis=3)
    temp_combined_images = temp_combined_images[56:184, 56:184, 13:141]

    inp = np.expand_dims(temp_combined_images, axis=0)

    print("asdjhafhsadihfvd", temp_combined_images.shape)
    ##predicting

    res = model.predict(inp)
    print(np.unique(res))
    return res, temp_combined_images[:, :, :, 0]


def random_predictor(model):
    p = "sample_demo_data"
    li = os.listdir(path=p)
    n = random.randint(0, len(li) - 1)
    flair = np.load(f"sample_demo_data/{li[n]}/flair.np.npy")
    t2 = np.load(f"sample_demo_data/{li[n]}/t2.np.npy")
    t1ce = np.load(f"sample_demo_data/{li[n]}/t1ce.np.npy")

    res, flair_2 = predict_for(flair, t2, t1ce, model)

    return res, flair, t2, t1ce, flair_2


model = load_model("unet3d_0.h5", compile=False)


@st.cache_data
def get_model(path="unet3d_0.h5", compile=False):
    return load_model(path, compile)


## overlay mask over brain image background


st.title("Multimodal 3D Brain tumor segmentation")
st.text(
    "End to end deep learning model for segmenting and labeling brain tumor in 3D MRI scan"
)

st.write(
    """
    **Accuracy :** **`96.89%`** 
    """
)
st.write(
    """
    **Architecture :** **`3D Unet - Encoder Decoder architecture`**
    """
)
st.write(
    """
    **Dataset :** **`BraTS 2020`**
    """
)
st.write(
    """
    **Mean IOU score :** **`0.74`**
    """
)
st.write(
    """
    **Dice coefficient of whole tumor :** **`0.989`**
    """
)


upload_file = st.sidebar.file_uploader("Enter file: flair", type=["nii"])
upload_file = st.sidebar.file_uploader("Enter file: t1 contrast", type=["nii"])
upload_file = st.sidebar.file_uploader("Enter file: t2 ", type=["nii"])


col1, col2 = st.columns([1, 1])
c1, c2, c3 = col2.columns([1, 1, 1])
slice_no = 45

if upload_file is not None:
    res, flair, t2, t1ce, flair_2 = random_predictor(model)

    flair = np.uint8(flair)
    t2 = np.uint8(t2)
    t1ce = np.uint8(t1ce)
    whole_tumor = 1 - res[0, :, :, slice_no, 0]
    whole_tumor = np.pad(whole_tumor, [(56, 56), (56, 56)], mode="constant")
    whole_tumor2 = 1 - res[0, :, :, :, 0]

    col1.image(
        whole_tumor,
        width=400,
        caption="Whole Tumor",
    )

    c1.image(
        flair[:, :, slice_no],
        use_column_width="auto",
        clamp=True,
        caption="Flair",
    )
    c2.image(
        t2[:, :, slice_no],
        use_column_width="auto",
        clamp=True,
        caption="T2",
    )
    c3.image(
        t1ce[:, :, slice_no],
        use_column_width="auto",
        clamp=True,
        caption="T1ce",
    )

    c1.image(
        np.pad(res[0, :, :, slice_no, 1], [(56, 56), (56, 56)], mode="constant"),
        caption="Necrotic/core",
        use_column_width="auto",
    )
    c2.image(
        np.pad(res[0, :, :, slice_no, 2], [(56, 56), (56, 56)], mode="constant"),
        caption="Edema",
        use_column_width="auto",
    )
    c3.image(
        np.pad(res[0, :, :, slice_no, 3], [(56, 56), (56, 56)], mode="constant"),
        caption="Enhancing",
        use_column_width="auto",
    )


slice_no = col2.slider("slice_no", min_value=0, max_value=127, step=1, value=45)
