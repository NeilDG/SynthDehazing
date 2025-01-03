
# <center> A New Approach for Training a Physics-Based Dehazing Network Using Synthetic Images
### <center>In Signal Processing - Elsevier
### <center>Neil Patrick Del Gallego, Joel Ilao, Macario Cordel II, Conrado Ruiz Jr.</center>
### <center>De La Salle University, Grup de Recerca en Tecnologies Media, La Salle - Universitat Ramon Llull</center>

### Abstract

<p align="justify"> In this study, we propose a new approach for training a physics-based dehazing network, using RGB images and depth maps gathered from a 3D urban virtual environment, with simulated global illumination and physically-based shaded materials. Since 3D scenes are rendered with depth buffers, full image depth can be extracted based on this information, using a custom shader, unlike
the extraction of real-world depth maps, which tend to be sparse. Our proposed physics-based dehazing network uses generated transmission and atmospheric maps from RGB images and depth maps from the virtual environment. To make our network compatible with real-world images, we incorporate a novel strategy of using unlit image priors during training, which can also be extracted
from the virtual environment. We formulate the training as a supervised image-to-image translation task, using our own DLSU-SYNSIDE (SYNthetic Single Image Dehazing Dataset), which consists of clear images, unlit image priors, transmission, and atmospheric maps. </p>

<p align="justify"> Our approach makes training stable and easier as compared to unsupervised approaches. Experimental results demonstrate the competitiveness of our approach against state-of-the-art dehazing works, using known benchmarking datasets such as I-Haze, O-Haze, and RESIDE, without our network seeing any real-world images during training. </p>

  
### DLSU-SYNSIDE (SYNthetic Single Dehazing Dataset)
The training images used in our paper, will be released soon.
  
### DLSU-SYNSIDE Pre-Trained Models
Pre-trained models include the style transfer network, unlit network, airlight and transmission estimators, as described in the paper. <br>
Link: <a href="https://drive.google.com/file/d/11HqA6xYMfrNRmNZOtN0S6jhZVWwa8roz/view?usp=sharing">Pre-trained models </a>

<br>
Assuming you have the source project, place all models in <b>"./checkpoint" </b> directory.

### Training
Training our models is not end-to-end. We do not have one ```train.py``` script at the moment. Our training procedure, as described in the paper, is divided into
several modules, namely ```cyclegan_main.py```, ```albedo_main.py```, ```transmission_main.py```, ```airlight_main.py```, which corresponds to the style-transfer, unlit image prior network, transmission map, atmospheric light estimation training respectively.

```pc_main.py``` contains commented code, on how these modules are trained sequentially, with different weights, patch sizes, batch size, alpha-beta scattering terms, etc.

### Inference
Provided you already have the pre-trained models, you can perform inference by: 
```
python  inference.py  --path="<hazy image directory>" --output="<dehazed image directory>"
```
Example:
```
python  inference.py  --path="E:/Hazy Dataset Benchmark/I-HAZE/hazy/*.jpg" --output="./output/dehazed/I-Haze/"
``` 
You can further check ```infer_main.py``` for some examples.
  
### Citation
```
@article{DELGALLEGO2022108631,
        title = {A new approach for training a physics-based dehazing network using synthetic images},
        journal = {Signal Processing},
        volume = {199},
        pages = {108631},
        year = {2022},
        issn = {0165-1684},
        doi = {https://doi.org/10.1016/j.sigpro.2022.108631},
        url = {https://www.sciencedirect.com/science/article/pii/S0165168422001712},
        author = {Neil Patrick {Del Gallego} and Joel Ilao and Macario Cordel and Conrado Ruiz}
}
```

### Acknowledgements
We would like to acknowledge De La Salle University (DLSU), Department of Science and Technology (DOST), and the Google Cloud Research program, for funding this research.
