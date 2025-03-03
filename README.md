# Skin Cancer Detection Using CNN

## Overview
This project is a deep learning-based skin cancer detection system using Convolutional Neural Networks (CNN). The model is trained on the HAM10000 dataset, which consists of dermatoscopic images labeled into seven categories of skin lesions. The trained model can classify skin images and predict the type of skin lesion.

## Features
- Image classification using CNN
- Handles class imbalance with oversampling
- Image augmentation for robust training
- Training visualization (accuracy and loss graphs)
- Model evaluation with confusion matrix
- Model export to TensorFlow Lite for mobile inference
- Supports predictions from camera images

## Dataset
The dataset used is `hmnist_28_28_RGB.csv`, containing 28x28 RGB images with corresponding labels:

| Label | Abbreviation | Full Name |
|--------|-------------|--------------------------------|
| 0 | akiec | Actinic keratoses and intraepithelial carcinomae |
| 1 | bcc | Basal cell carcinoma |
| 2 | bkl | Benign keratosis-like lesions |
| 3 | df | Dermatofibroma |
| 4 | nv | Melanocytic nevi |
| 5 | vasc | Pyogenic granulomas and hemorrhage |
| 6 | mel | Melanoma |

## Model Architecture
The CNN model consists of:
- Convolutional layers with ReLU activation
- Batch normalization
- Max pooling layers
- Fully connected dense layers
- Dropout regularization
- Softmax output layer for classification

## Installation & Setup
### Prerequisites
Ensure you have Python installed along with the required dependencies:
```bash
pip install tensorflow keras numpy pandas matplotlib seaborn imbalanced-learn
```

### Training the Model
Run the following script to train the model:
```bash
python train.py
```
This will train the CNN on the dataset and save the trained model as `Skin_Cancer.h5` and `Skin.tflite`.

### Model Evaluation
The model is evaluated using:
- Training and validation loss/accuracy visualization
- Confusion matrix analysis
- Model performance metrics (accuracy, loss)

### Prediction
To predict an image using the trained model:
```bash
python predict.py --image_path <path_to_image>
```
Example:
```bash
python predict.py --image_path test_image.jpg
```

The script loads the trained model and predicts the class of the input image.

### Mobile Compatibility
The model is converted to TensorFlow Lite for deployment on mobile devices:
```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('Skin.tflite', 'wb') as f:
    f.write(tflite_model)
```
This allows running the model on edge devices.

## Results
- Training and validation accuracy/loss plots are generated.
- Confusion matrix visualization helps analyze misclassifications.
- Model achieves high accuracy on skin lesion classification.

## Future Improvements
- Improve model generalization with additional data augmentation.
- Implement Grad-CAM for explainability.
- Deploy as a web or mobile application for real-world use.

## Author
Eshwar B.

## License
This project is open-source and available under the MIT License.

