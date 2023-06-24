
# Multimodal 3D Tumor segmentation

An end to end deep learning model for 3d brain tumor segmentation.
Model follows encoder decoder architecture , built from scratch
and trained on [BraTS2020](https://www.kaggle.com/datasets/awsaf49/brats2020-training-data) dataset.

## [view web app](https://hackisenow-23c39t8062j.streamlit.app/)

## Performence
- Accurcay `97.89%`
- Mean IOU `0.745`
- Dice coefficient `0.989`

## Run Locally

Clone the project
```bash
  git clone https://github.com/nagarajRPoojari/hackISEnow.git
```
Install dependencies
```bash
  pip install -r requirements.txt
```
Start the server
```bash
  streamlit run app.py
```
## Sample result
<img src="https://github.com/nagarajRPoojari/hackISEnow/assets/116948655/23db2ff0-c80c-45d6-a5fc-9b79d52947b9" width="470" />
<img src="https://github.com/nagarajRPoojari/hackISEnow/assets/116948655/2cb3688e-cd38-4638-b6bb-3986f44569df" width="400" />




