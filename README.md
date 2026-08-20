AlexNet on CIFAR-10

Implementation of AlexNet using PyTorch for image classification on the CIFAR-10 dataset.

Model

* 5 convolutional layers
* 3 max-pooling layers
* 3 fully connected layers
* ReLU activation
* Dropout

Training

* Epochs: 200
* Batch size: 128
* Optimizer: SGD
* Initial learning rate: 0.01
* Learning rate scheduler: MultiStepLR
* GPU: Tesla T4 (Google Colab)

Results
* Best Test Accuracy: 85.76%
* Final Train Accuracy: 99.23%
* Final Test Accuracy: 85.57%
* Average Training Time per Epoch: 18.29s

