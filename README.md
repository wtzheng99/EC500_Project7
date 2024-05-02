# EC500_Project7

### Automated Evaluation of Lung Histology Inflammation Scoring Using Deep Neural Networks

Our task is to build a model that estimates histological scores for human lung inflammation from analyzing patches from whole slide images (wsi). We plan to achieve this by using nuclei segmentation and counting. Challenges would be in feature extraction. To be specific, it would be challenging to locate nuclei boundaries in overlapping cells. Another obstacle in this project would be incorporating various neural network models into our model’s architecture.



The codes for data preprocessing is under the 'Data Preprocessing' directory, including how to segment patches from the original whole slide image, how to pair the counts with inflammtory levels.

In Feature Extraction, you will find the code for extracting features from ResNet50 and [PLIP (Pathology Language and Image Pre-Training)](https://github.com/PathologyFoundation/plip.git).

We also used [HoVer-Net model](https://github.com/vqdang/hover_net.git) with weights trained on [MoNuSAC](https://monusac-2020.grand-challenge.org) dataset and [Set Transformer](https://github.com/juho-lee/set_transformer.git) in this project.


