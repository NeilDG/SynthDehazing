# <center> A New Approach for Training a Physics-Based Dehazing Network Using Synthetic Images
### <center>Neil Patrick Del Gallego, Joel Ilao, Macario Cordel II, Conrado Ruiz Jr.</center>
### <center>De La Salle University, Grup de Recerca en Tecnologies Media, La Salle - Universitat Ramon Llull</center>

# DISCLAIMER
**The source code is currently still in pre-deployment stage. Please stay tuned for the production and test-ready source code.**

[//]: # (### <center>In Signal Processing - Elsevier)

<br>

### <strong>Paper (coming soon) | Source code (coming soon) | Pretrained model (coming soon)

<img src="web_img/results_3.png">

### Abstract

<p align="justify"> In this study, we propose a new approach for training a physics-based dehazing network, using RGB images and depth maps gathered from a 3D urban virtual environment, with simulated global illumination and physically-based shaded materials. Since 3D scenes are rendered with depth buffers, full image depth can be extracted based on this information, using a custom shader, unlike
the extraction of real-world depth maps, which tend to be sparse. Our proposed physics-based dehazing network uses generated transmission and atmospheric maps from RGB images and depth maps from the virtual environment. To make our network compatible with real-world images, we incorporate a novel strategy of using unlit image priors during training, which can also be extracted
from the virtual environment. We formulate the training as a supervised image-to-image translation task, using our own DLSU-SYNSIDE (SYNthetic Single Image Dehazing Dataset), which consists of clear images, unlit image priors, transmission, and atmospheric maps. </p>

<p align="justify"> Our approach makes training stable and easier as compared to unsupervised approaches. Experimental results demonstrate the competitiveness of our approach against state-of-the-art dehazing works, using known benchmarking datasets such as I-Haze, O-Haze, and RESIDE, without our network seeing any real-world images during training. </p>

### Directory for High-Resolution Image Results

<strong><a href="https://github.com/NeilDG/SynthDehazing/tree/gh-pages/web_img/">SynthDehazing Results</a>

### DLSU-SYNSIDE (SYNthetic Single Dehazing Dataset)
The training images used in our paper, will be released soon.

### Citation

Coming Soon

### Acknowledgements

We would like to acknowledge De La Salle University (DLSU), Department of Science and Technology (DOST), and the Google Cloud Research program, for funding this research.
