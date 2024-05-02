# EC500_Project7

## Overview
### Automated Evaluation of Lung Histology Inflammation Scoring Using Deep Neural Networks
This repository contains the source code and documentation for our project, which focuses on applying deep learning techniques to analyze medical imaging data, specifically on whole slide images of lung tissue. The project utilizes advanced neural network models:
- HoVer-Net = nuclear segmentation and classification
- ResNet50 and PLIP = feature extraction
- Set Transformer and Deep Sets = analysis on extracted features

## Features
- **ResNet50 and PLIP Integration:** Utilizes these models to extract meaningful features from pathology images of lung tissues.
- **Set Transformer and Deep Sets:** Analyzes sets of features to predict outcomes accurately.
- **Class Balancing Techniques:** Implements methods to handle imbalanced data, improving the robustness of the model.
- **Classification:** There are 6 classification models that can be used to classify annotated data. Optional: ground truth and predicted labels can be overlaid on whole slide images.
- **Clustering:** There are 3 clustering models that can be used to experiment the dataset with supervised and unsupervised clustering. Optional: clusters can be overlaid on whole slide images.


## Data Preprocessing
The code for data preprocessing is located under the 'Data Preprocessing' directory. It includes scripts for segmenting patches from original whole slide images and pairing the counts with inflammatory levels.

## Feature Extraction
In the Feature Extraction directory, you'll find code for extracting features using ResNet50 and [PLIP (Pathology Language and Image Pre-Training)](https://github.com/PathologyFoundation/plip.git). This process is crucial for preparing the data for subsequent model training.

## Models Used
- [HoVer-Net Model](https://github.com/vqdang/hover_net.git): Utilized with weights trained on the [MoNuSAC](https://monusac-2020.grand-challenge.org) dataset for precise nuclear segmentation.
- [Set Transformer](https://github.com/juho-lee/set_transformer.git): Employed for its advanced handling of set-based data, benefiting from its self-attention mechanism.

Due to licensing restrictions, we cannot include detailed information about the models used. Please refer to the original repository for further details.

## Installation
To set up this project, follow these steps:
1. Clone the repository:

```bash
# Clone the repository
git clone https://github.com/yourusername/yourproject.git
```
2. Install required dependencies:

```bash
pip install -r hover_net/requirements.txt
```

## Usage
To run the project, execute the following command:

```python
python hover_net/run_tile.py
```



















