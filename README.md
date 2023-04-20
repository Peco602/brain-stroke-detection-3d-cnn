# Brain stroke detection from CT scans via 3D Convolutional Neural Network

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/https://github.com/Peco602/brain-stroke-detection-3d-cnn/blob/main/brain_stroke_detection_3d_cnn.ipynb)

Tutorial on how to train a 3D Convolutional Neural Network (3D CNN) to detect the presence of brain stroke.

## Objective

The [Brain Stroke CT Image Dataset](https://www.kaggle.com/datasets/afridirahman/brain-stroke-ct-image-dataset) from Kaggle provides normal and stroke brain Computer Tomography (CT) scans. The dataset presents very low activity even though it has been uploaded more than 2 years ago. It may be probably due to its quite low usability (3.13). The challenge is to get some interesting result, i.e., to try to perform brain stroke detection, even from this low-quality dataset.

## Approach

The followed approach is based on the usage of a 3D Convolutional Neural Network (CNN) in place of a standard 2D one. 2D CNNs are commonly used to process both grayscale (1 channel) and RGB images (3 channels), while a 3D CNN represents the 3D equivalent since it takes as input a 3D volume or a sequence of 2D frames, e.g. slices in a CT scan. The provided example takes inspiration from the great work [3D image classification from CT scans](https://keras.io/examples/vision/3D_image_classification/) done by [Hasib Zunair](https://twitter.com/hasibzunair) who clearly demonstrated how to use a 3D CNN to predict the presence of viral pneumonia from CT scans.

## Usage

The notebook can be run on Google Colab. Copy the [URL](https://github.com/Peco602/brain-stroke-detection-3d-cnn/blob/main/brain_stroke_detection_3d_cnn.ipynb) of the notebook [here](https://colab.research.google.com/github/).

## Authors

- [Giovanni Pecoraro](https://www.peco602.com)

## References

- [3D image classification from CT scans](https://keras.io/examples/vision/3D_image_classification/)
- [Medical Images In python (Computed Tomography)](https://vincentblog.xyz/posts/medical-images-in-python-computed-tomography)
- [Data augmentation for medical image analysis in deep learning](https://www.imaios.com/en/resources/blog/ai-for-medical-imaging-data-augmentation)
