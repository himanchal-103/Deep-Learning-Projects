# HUMAN ACTION RECOGNITION

## Objective
The primary objective of this Human Activity Recognition project is to develop an accurate and efficient model that can  detect and classify various human activities.

## Steps
1. Download and visualize the data with its labels
2. Preprocess the dataset
3. Split the data into train and test
4. Implement the convLSTM approach
    1. Contruct the model
    2. Compile and train the model
    3. Plot model's loss and accuracy curves
5. Implement the LRCN approach
    1. Construct the model
    2. Compile and train the model
    3. Plot model's loss and accuracy curves


## Dataset
The UCF50 - Action Recognition Dataset is a comprehensive and widely-used collection designed for human activity recognition tasks. It provides a diverse set of videos categorized into 50 distinct action classes, enabling the development and evaluation of models for action recognition.

To download and extract the UCF50 - Action Recognition Dataset, use the following commands:
```bash
# Download the dataset
!wget --no-check-certificate https://www.crcv.ucf.edu/data/UCF50.rar

# Extract the dataset
!unrar x UCF50.rar
```
Description of dataset content and structure:
1. 50 action categories
2. 25 groups of videos per action category
3. 133 average videos per action category
4. 199 average number of frames per video
5. 320 average frames width per video
6. 240 average frames height per video
7. 26 average frames per second per video

## Model
1. ConvLSTM approach:
  - A ConvLSTM cell is a variant of an LSTM network that contains convolution operations in the network. it is an LSTM with convolution embedded in the architecture, which makes it capable of identifying spatial features of the data while keeping into account the temporal relation. 
  - For video classification, this approach effectively captures the spatial relation in the individual frames and the temporal relation across the different frames. As a result of this convolution structure, the ConvLSTM is capable of taking in 3-dimensional input `(width, height, num_of_channels)` whereas a simple LSTM only takes in 1-dimensional input hence an LSTM is incompatible with modeling Spatio-temporal data on its own.
  - Published paper [**Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting**](https://arxiv.org/abs/1506.04214v1) by Xingjian Shi (NIPS 2015), to learn more about this architecture.
2. LRCN approach:
- In this step, we will implement the Long-term Recurrent Convolutional Network (LRCN) approach by combining Convolution and LSTM layers in a single model.
- It combines CNN and LSTM layers in a single model. The Convolutional layers are used for spatial feature extraction from the frames, and the extracted spatial features are fed to LSTM layer(s) at each time step for temporal sequence modeling. This way the network learns spatiotemporal features directly in an end-to-end training, resulting in a robust model.
- Published paper [Long-term Recurrent Convolutional Networks for Visual Recognition and Description](https://arxiv.org/abs/1411.4389?source=post_page---------------------------) by Jeff Donahue (CVPR 2015), to learn more about this architecture.
- In this approach [**`TimeDistributed`**](https://keras.io/api/layers/recurrent_layers/time_distributed/) wrapper layer, which allows applying the same layer to every frame of the video independently. So it makes a layer (around which it is wrapped) capable of taking input of shape `(no_of_frames, width, height, num_of_channels)` if originally the layer's input shape was `(width, height, num_of_channels)` which is very beneficial as it allows to input the whole video into the model in a single shot.

