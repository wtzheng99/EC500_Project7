# EC500_Project7

## Overview
### Automated Evaluation of Lung Histology Inflammation Scoring Using Deep Neural Networks
This repository contains the source code and documentation for our project, which focuses on applying deep learning techniques to analyze medical imaging data. The project utilizes advanced neural network models like ResNet50 and PLIP for feature extraction and employs architectures such as Set Transformer and Deep Sets for further analysis.

## Features
- **ResNet50 and PLIP Integration:** Utilizes these models to extract meaningful features from pathology images.
- **Set Transformer and Deep Sets:** Analyzes sets of features to predict outcomes accurately.
- **Class Balancing Techniques:** Implements methods to handle imbalanced data, improving the robustness of the model.

## Installation
To set up this project, follow these steps:
1. Clone the repository:




The codes for data preprocessing is under the 'Data Preprocessing' directory, including how to segment patches from the original whole slide image, how to pair the counts with inflammtory levels.

In Feature Extraction, you will find the code for extracting features from ResNet50 and [PLIP (Pathology Language and Image Pre-Training)](https://github.com/PathologyFoundation/plip.git).

We also used [HoVer-Net model](https://github.com/vqdang/hover_net.git) with weights trained on [MoNuSAC](https://monusac-2020.grand-challenge.org) dataset and [Set Transformer](https://github.com/juho-lee/set_transformer.git) in this project.







