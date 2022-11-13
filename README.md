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
  - ![equation](https://latex.codecogs.com/svg.image?f_%7Brelu%7D(x_%7Bi,j%7D)%20=%20x_%7Bi,j%7D%20%20for%20%20%20%20x_%7Bi,j%7D%3E%200)

  - ![equation](https://latex.codecogs.com/svg.image?%5Cfrac%7B%5Cpartial%20f_%7Brelu%7D(x_%7Bi,j%7D)%7D%7B%5Cpartial%20x%7D%20=%201,%20%20for,%20%20%20%20x_%7Bi,j%7D%3E%200)

## SoftMax Layer

### Forward

  - ![equation](https://latex.codecogs.com/svg.image?y_%7Bi,j%7D%5E%7B'%7D%20=%20y_%7Bi,j%7D%20-%20max_%7Bi,j%7D(y_%7Bi,j%7D))
  - ![equation](https://latex.codecogs.com/svg.image?z_%7Bi,j%7D%20=%20f_%7Bsoftmax%7D(y_%7Bi,j%7D)%20=%20%5Cfrac%7Be%5E%7By_%7Bi,j%7D%5E%7B'%7D%7D%7D%7B%5Csum_%7Bk%7De%5E%7By_%7Bi,k%7D%5E%7B'%7D%7D%7D)

## Cross-Entropy

 
### Loss

  - ![equation](https://latex.codecogs.com/svg.image?l(z,t)=-%5Cfrac%7B1%7D%7Bn_%7Bb%7D%7D%5Csum%20_%7Bi%7D%20%5Csum%20_%7Bj%7D%20t_%7Bi,j%7Dlog(z_%7Bi,j%7D))

### Gradient
  - ![equation](https://latex.codecogs.com/svg.image?%5Cfrac%7B%5Cpartial%20l(z,%20t)%7D%7B%5Cpartial%20z_%7Bi,j%7D%7D%20=%20-%5Cfrac%7B1%7D%7Bn_%7Bb%7D%7D%5Cfrac%7Bt_%7Bi,j%7D%7D%7Bz_%7Bi,j%7D%7D)
## Sequential Neural Network
  - Forward function that calls the forward of each class
  - Backward function that calls the backward of each class
  - Fit function to fit training data
  - Predict function to test test data
  - Update params that will use update params of each class  

# Result

![acc_lr](https://user-images.githubusercontent.com/97703581/201528816-ff6e40e6-bf43-437d-8545-5f6b178f1893.png)

# Note

Make sure to have cifar-100-python in your directory for data initialization, it can be obtained through this link:
https://www.cs.toronto.edu/~kriz/cifar.html

