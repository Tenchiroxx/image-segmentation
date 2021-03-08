# Multispectral and Hyperspectral Image segmentation with CNNs

Author : Emmanuel Gardin
E-mail : emmanuel.gardin@mines-paristech.fr 
Date : March 2021


This project is aimed at the segmentation of multispectral and hyperspectral images using neural networks. It is based on the paper Jeppesen, 2019, which highlights the difficulty of the U-Net architecture in exploiting spectral information for multispectral images in the SPARCS dataset.

In this project, three neural networks are used:
- A classical U-Net;
- A Separable U-Net, in which the convolution layers are replaced by separable convolutions in the encoder part of the architecture;
- A 1D CNN, which for a given pixel, applies convolutions on the spectral dimension.

These neural networks are evaluated on two sets of data:

- Landsat 8, SPARCS (Multispectral): https://www.usgs.gov/core-science-systems/nli/landsat/spatial-procedures-automated-removal-cloud-and-shadow-sparcs
- AVIRIS, Salinas (Hyperspectral): http://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes#Salinas_scene


# General information

- To use one of the architectures, modify "salinas_pipeline.py" or "landsat_pipeline.py" to choose the architecture and then execute it at the command prompt. 

- The files containing "tuner" in their name allow to find the optimal hyperparameters for a given architecture using a Bayesian optimization algorithm. (https://keras-team.github.io/keras-tuner/documentation/tuners/)

- The file unet.py contains all the neural network architectures used. 

- The confusion_matrix.py module is useful to draw detailed confusion matrices.

- The "images" folder contains segmentation results for the two datasets, and for each of the architectures. 

- The "salinas_utils" and "landsat_utils" folders contain the utility python files to execute the pipelines. 

