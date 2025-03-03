import os
import numpy as np
import tensorflow as tf
from utils import preprocess_image, classes, plot_image_with_result

def predict_image(img_path):
    model_path = 'Skin_Cancer.h5'
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"No such file: '{model_path}'")

    print(f"Predicting image: {img_path}")
    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully")

    img_array = preprocess_image(img_path)
    print(f"Image preprocessed: {img_array.shape}")
    
    print("Making prediction")
    predictions = model.predict(img_array)
    print(f"Predictions array shape: {predictions.shape}")
    print(f"Predictions: {predictions}")

    if predictions.shape[1] != len(classes):
        raise ValueError("Mismatch between the number of classes and predictions array shape")

    predicted_class_index = np.argmax(predictions[0])
    predicted_class = classes[predicted_class_index]

    return predicted_class

if __name__ == "__main__":
    image_path = r"E:\skin_cancer_detection\server\archive (1)\Skin cancer ISIC The International Skin Imaging Collaboration\Train\actinic keratosis\ISIC_0025803.jpg"

    try:
        result = predict_image(image_path)
        print(f"Predicted Class: {result}")
        plot_image_with_result(image_path, result)
    except Exception as e:
        print(f"Error: {e}")
