# Federated segmentation of whole slide images
Federated learning using the Digital Slide Archive and HistomicsTK-deeplab

<img src = "./federated diagram.jpg" width="60%"/>

<h2>Setup</h2>

Federated workers can be setup by intalling the [Digital Slide Archive](https://digitalslidearchive.github.io/digital_slide_archive/)  
This code uses the [Histo-Cloud](https://github.com/SarderLab/Histo-cloud) plugin to train segmentation networks via the DSA instances.

The train_network plugin is called via the REST API to train a network on each DSA instance before the resulting models are downloaded to the master server and averaged. 

<h2>Useage</h2>

The Deeplab ImageNet pretrained model can be downloaded from [here](https://buffalo.box.com/s/izjhqdqtlrznm9zh4k5dv2d6j9tz7n77). This model should be placed in the same folder as this code prior to training.

The hyper-parameters for training can be adjusted in the [federated_training.py](https://github.com/SarderLab/federated_learning/blob/main/federated_training.py) code. 

An example of how to start the training is provided in [run_federated_learning.py](https://github.com/SarderLab/federated_learning/blob/main/run_federated_learning.py)
