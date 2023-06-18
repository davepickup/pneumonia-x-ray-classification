# Pneumonia Chest X-ray detection

A `Python` package to allow building and testing of a image classification model for the prediction of whether a patient has Pneumonia from their Chest X-ray. The dataset use was obtained from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

## Quickstart details

Follow these steps to get started:
1. Clone the repository
2. Install the requirements.txt packages for development purposes (`pip install -r requirements.txt`)
3. Install the package `pip install -e .`

The notebooks assume the dataset is in `data/` directory at the top level of the repo with directories for `train`, `validation` and `test` and within those directories a `NORMAL/` and `PNEUMONIA` folder with the corresponding image file. Here is an example folder structure where the normal train images are to be found:
`/data/chest_xray/train/NORMAL/`
