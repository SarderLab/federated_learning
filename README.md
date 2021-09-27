# federated_learning
Federated learning using the Digital Slide Archive and HistomicsTK-deeplab

Federated workers can be setup by intalling the [Digital Slide Archive](https://digitalslidearchive.github.io/digital_slide_archive/)  

This code uses the [Histo-Cloud](https://github.com/SarderLab/Histo-cloud) plugin to train segmentation networks via the DSA instances.

The hyper-parameters for training can be adjusted in the [federated_training.py](https://github.com/SarderLab/federated_learning/blob/main/federated_training.py) code. An example of how to start the training is provided in [run_federated_learning.py](https://github.com/SarderLab/federated_learning/blob/main/run_federated_learning.py)
