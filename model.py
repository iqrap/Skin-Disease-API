import tensorflow as tf
import numpy as np
from PIL import Image
import io

class DiseaseModel:
    def __init__(self, model_path='Final_Disease_Datasets.h5'):
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = ['Acne', 'Benign Tumors', 'Lichen', 'No Disease', 'Vitiligo']
        self.img_size = 224
    
    def preprocess_image(self, image_bytes):
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert('RGB')
        img = img.resize((self.img_size, self.img_size))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = tf.keras.applications.mobilenet.preprocess_input(img_array)
        return img_array
    
    def predict(self, image_bytes):
        processed_img = self.preprocess_image(image_bytes)
        predictions = self.model.predict(processed_img)[0]
        predicted_class_idx = np.argmax(predictions)
        confidence = float(predictions[predicted_class_idx] * 100)
        
        return {
            'diseaseName': self.class_names[predicted_class_idx],
            'confidence': round(confidence, 2)
        }

model_instance = None

def get_model():
    global model_instance
    if model_instance is None:
        model_instance = DiseaseModel()
    return model_instance