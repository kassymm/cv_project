# Robustness and Explainability of Deep Convolutional Classifiers via Grad-CAM and Adversarial Attacks

This project focuses on robust image classification and explainability for the Oxford-IIIT Pet dataset, using a variety of convolutional neural networks (CNNs) and adversarial training techniques. The project also includes Grad-CAM visualizations to interpret model predictions.

## Overview

- **Dataset**: Oxford-IIIT Pet Dataset (with enriched captions and bounding boxes from [visual-layer/oxford-iiit-pet-vl-enriched](https://huggingface.co/datasets/visual-layer/oxford-iiit-pet-vl-enriched)).
- **Models**: MobileNetV2, ResNet18, EfficientNet-B0, DenseNet121, VGGNet19.
- **Tasks**:
  - Standard and adversarial (FGSM) training and evaluation.
  - Robustness analysis under adversarial attacks.
  - Model explainability using Grad-CAM.
- **Outputs**: Trained model weights, classification results, Grad-CAM heatmaps, and IoU-based explainability metrics.

## Features

- **Data Processing**: Images are resized and padded to 224x224, and only samples with valid cat/dog bounding boxes are used.
- **Training**: Both standard and adversarial (FGSM) training loops are implemented for all models.
- **Evaluation**: Top-1 and Top-5 accuracy metrics are reported for both clean and adversarial test sets.
- **Explainability**: Grad-CAM is used to generate heatmaps, and the overlap (IoU) between Grad-CAM bounding boxes and ground-truth boxes is computed for quantitative analysis.

## Usage

- **Data Preparation**: See `data_processing.ipynb` for dataset loading, filtering, and preprocessing steps.
- **Training & Evaluation**: Use `train.ipynb` to train and evaluate models. Both standard and adversarial training are supported.
- **Explainability**: Use `gradcam.ipynb` to generate Grad-CAM visualizations and compute IoU metrics for model interpretability.
- **Results**: CSV files in the `results/` directory contain per-sample Grad-CAM IoU scores and other evaluation metrics.

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- timm
- datasets
- matplotlib
- numpy
- PIL

*(See `requirements.txt` for exact versions.)*

## References

- [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- [FGSM Adversarial Training](https://arxiv.org/pdf/1412.6572)
- [Grad-CAM: Visual Explanations from Deep Networks](https://arxiv.org/abs/1610.02391) 
