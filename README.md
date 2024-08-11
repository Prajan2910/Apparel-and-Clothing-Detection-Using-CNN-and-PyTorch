# Apparel-and-Clothing-Detection-Using-CNN-and-PyTorch

---

# FashionNet: Simple Apparel Classifier with PyTorch

This repository contains a PyTorch implementation of a neural network designed to classify images from the Fashion MNIST dataset. The dataset consists of grayscale images of various clothing items, with each image being 28x28 pixels.

## Project Overview

The project includes the following key components:

- **Data Loading and Preprocessing**: The Fashion MNIST dataset is loaded and split into training, validation, and test sets. The images are normalized and transformed into tensors for processing by the neural network.
  
- **Model Architecture**: A simple neural network is built using PyTorch's `nn.Sequential` with layers including fully connected (linear) layers, ReLU activations, and dropout for regularization. The model outputs class probabilities using `LogSoftmax`.

- **Training and Validation**: The model is trained using the Adam optimizer and negative log likelihood loss (`NLLLoss`). The training and validation losses, as well as validation accuracy, are tracked and printed for each epoch.

- **Visualization**: The training and validation losses are plotted to observe the model's performance over time. Additionally, the model's predictions on sample images are visualized alongside the predicted class probabilities.

- **Image Prediction**: Three options are provided to predict images:
  1. **Random Image from Dataset**: Use a random image from the Fashion MNIST dataset to predict the class.
  2. **Image from URL**: Preprocess and predict the class of an image from a given URL.

## Dependencies

- Python 3.x
- PyTorch
- torchvision
- matplotlib
- numpy
- requests
- PIL (Pillow)

You can install the required packages using:

```bash
pip install torch torchvision matplotlib numpy requests pillow
```

## Usage

### 1. Training the Model
To train the model, simply run the main script. The model will train for 32 epochs, and the training and validation losses will be displayed.

```bash
python train.py
```

### 2. Predicting from a Random Dataset Image
To predict the class of a random image from the Fashion MNIST dataset, use the following command:

```bash
python predict_random.py
```

### 3. Predicting from a URL
To predict the class of an image from a URL, use the following command:

```bash
python predict_url.py --image_url "http://example.com/image.jpg"
```

## Results

The model achieves a validation accuracy of approximately 89% after 45 epochs of training. The trained model can accurately predict the class of various clothing items in the Fashion MNIST dataset.

## Examples

### Training and Validation Losses

![Training and Validation Losses](https://github.com/Prajan2910/Apparel-and-Clothing-Detection-Using-CNN-and-PyTorch/blob/13c72b0855e873c04f390cd9dd26d2603b372bc5/images/training%20loss.png)

### Sample Prediction

I have given the Image URL and plotted the result:

![Sample Prediction](https://github.com/Prajan2910/Apparel-and-Clothing-Detection-Using-CNN-and-PyTorch/blob/13c72b0855e873c04f390cd9dd26d2603b372bc5/images/Test%20image1.png)

![result](https://github.com/Prajan2910/Apparel-and-Clothing-Detection-Using-CNN-and-PyTorch/blob/13c72b0855e873c04f390cd9dd26d2603b372bc5/images/test1%20result.png)

This project was done by **Prajan Kannan**.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-%230077B5.svg?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/prajan-kannan/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
