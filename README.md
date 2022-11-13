# Simple-Neural-Network-From-Scratch
The purpose of this project, is to implement a Fully Connected Layer, a Relu Layer, a softmax layer, and a cross-entropy loss function and to understand the concept of back-propagation in order to train.

## Fully Connected Layer

### Initialization:

  - ![equation](https://latex.codecogs.com/svg.image?W%20%5Cmapsto%20N(0,%20%5Csqrt%7B%5Cfrac%7B2%7D%7Bn_%7Bi%7D&plus;n_%7Bo%7D%7D%7D))

  - ![equation](https://latex.codecogs.com/svg.image?b%20%5Cmapsto%200)

### Forward:

  - ![equation](https://latex.codecogs.com/svg.image?f_%7Bfull%7D(x_%7Bi%7D)%20=%20x_%7Bi%7DW%5E%7BT%7D%20&plus;%20b)
  
### Back Propagation:

  - ![equation](https://latex.codecogs.com/svg.image?%5Cfrac%7B%5Cpartial%20f_%7Bfull%7D%7D%7B%5Cpartial%20x_%7Bi%7D%7D%20=%20W)
  
  - ![equation](https://latex.codecogs.com/svg.image?%5Cfrac%7B%5Cpartial%20f_%7Bfull%7D%7D%7B%5Cpartial%20b%7D%20=%20I_%7Bn0%7D)
  
  - ![equation](https://latex.codecogs.com/svg.image?%5Cfrac%7B%5Cpartial%20l%7D%7B%5Cpartial%20x_%7Bi%7D%5E%7Bl%7D%7D%20=%20%5Cfrac%7B%5Cpartial%20l%7D%7B%5Cpartial%20x_%7Bi%7D%5E%7Bl&plus;1%7D%7D%20W%5E%7Bl%7D)
  
  - ![equation](https://latex.codecogs.com/svg.image?%5Cfrac%7B%5Cpartial%20l%7D%7B%5Cpartial%20W%5E%7Bl%7D%7D%20=%20%5Csum_%7Bi%7D(%5Cfrac%7B%5Cpartial%20l%7D%7B%5Cpartial%20x_%7Bi%7D%5E%7Bl&plus;1%7D%7D)%5E%7BT%7D%20x_%7Bi%7D%5E%7Bl%7D)
  
  - ![equation](https://latex.codecogs.com/svg.image?%5Cfrac%7B%5Cpartial%20l%7D%7B%5Cpartial%20b%5E%7Bl%7D%7D%20=%20%5Csum_%7Bi%7D(%5Cfrac%7B%5Cpartial%20l%7D%7B%5Cpartial%20x_%7Bi%7D%5E%7Bl&plus;1%7D%7D))
  
### Parameters Update:

  - ![equation](https://latex.codecogs.com/svg.image?b%5E%7B'%7D%20=%20b%20-%20%5Ceta%20(%5Cfrac%7B%5Cpartial%20l%7D%7B%5Cpartial%20b%7D))

  - ![equation](https://latex.codecogs.com/svg.image?W%5E%7B'%7D%20=%20W%20-%20%5Ceta%20(%5Cfrac%7B%5Cpartial%20l%7D%7B%5Cpartial%20W%7D))

## ReLu Layer


## SoftMax Layer


## Cross-Entropy


## Sequential Neural Network


# Note

Make sure to have cifar-100-python in your directory for data initialization, it can be obtained through this link:
https://www.cs.toronto.edu/~kriz/cifar.html

