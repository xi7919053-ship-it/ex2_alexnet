AlexNet on CIFAR-10

This project implements AlexNet using PyTorch and trains the model on the CIFAR-10 dataset.

1. Model

The AlexNet model is implemented in models/alexnet.py.

The network consists of:

* 5 convolutional layers
* 3 max-pooling layers
* ReLU activation functions
* 3 fully connected layers
* Dropout for reducing overfitting

For CIFAR-10 images with an input size of 3 × 32 × 32, the network structure is:

Input: 3 × 32 × 32
Conv1: 3 → 64
MaxPool
Conv2: 64 → 192
MaxPool
Conv3: 192 → 384
Conv4: 384 → 256
Conv5: 256 → 256
MaxPool
Flatten: 256 × 2 × 2 = 1024
FC1: 1024 → 4096
FC2: 4096 → 4096
FC3: 4096 → 10

2. Dataset

The model is trained and evaluated on the CIFAR-10 dataset.

CIFAR-10 contains 60,000 RGB images of size 32 × 32 from 10 classes:

* 50,000 training images
* 10,000 test images

Data augmentation is applied to the training set using:

* Random Crop
* Random Horizontal Flip
* Normalization

3. Training Configuration

* Framework: PyTorch
* Epochs: 200
* Batch size: 128
* Loss function: Cross Entropy Loss
* Optimizer: SGD
* Initial learning rate: 0.01
* Momentum: 0.9
* Weight decay: 0.0005
* GPU: NVIDIA Tesla T4 (Google Colab)

A learning rate scheduler is used during training:

Epoch 1–80:    0.01
Epoch 81–140:  0.001
Epoch 141–180: 0.0001
Epoch 181–200: 0.00001

4. Project Structure

ex2_alexnet/
│
├── models/
│   ├── __init__.py
│   └── alexnet.py
│
├── data.py
├── train.py
├── main.py
├── util.py
├── outputs/
├── weights/
└── README.md

5. Training Results

The model is trained for 200 epochs using an NVIDIA Tesla T4 GPU on Google Colab.

Final results will be updated after training is completed.

* Best Test Accuracy: TBD
* Final Train Accuracy: TBD
* Final Test Accuracy: TBD
* Average Training Time per Epoch: TBD

6. Run

Run the training program with:

python main.py
