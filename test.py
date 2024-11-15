import cv2
import numpy as np
import pandas as pd
import joblib


model = joblib.load("C:/Users/Asus/Desktop/Minor Project/Final/Melanoma/xgb_model.joblib")


input_width = 224  
input_height = 224  


def extract_features(image):
    
    image_flattened = cv2.resize(image, (input_width, input_height)).flatten()  
    
    features = image_flattened[:50]  
    return features


def preprocess_image(image_path):
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image at path {image_path} could not be loaded. Check the file path.")

    
    img_resized = cv2.resize(img, (input_width, input_height))  
    
    img_array = img_resized.astype('float32') / 255.0
    
    features = extract_features(img_array)
    return features


def predict_lesion(image_path):
    
    features = preprocess_image(image_path)
    
    
    features = features.reshape(1, -1)  
    prediction = model.predict(features)

    
    if prediction.ndim == 1:
        class_label = int(prediction[0])  
    else:
        class_label = np.argmax(prediction, axis=1)[0]  

    
    labels = ['Benign', 'Malignant']  
    result = labels[class_label]
    
    return result


if __name__ == "__main__":
    
    image_path = r"C:/Users/Asus/Desktop/Minor Project/Final/Melanoma/Validation/malignantvalid/1 (2).jpg"  
    
        
    result = predict_lesion(image_path)
    print(f"The image is classified as: {result}")
