import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import cv2
import matplotlib.pyplot as plt


seq_model = tf.keras.models.load_model('seq_model.keras')
model_2 = tf.keras.models.load_model('model_2.keras')

models = [seq_model, model_2]

class_labels = ['Crack', 'Missing Head', 'Paint-off']
image_paths = ['data/test/crack/test_crack.jpg', 'data/test/missing-head/test_missinghead.jpg', 'data/test/paint-off/test_paintoff.jpg']
true_labels = { 'data/test/crack/test_crack.jpg': 'Crack', 'data/test/paint-off/test_paintoff.jpg': 'Paint-off', 'data/test/missing-head/test_missinghead.jpg': 'Missing Head'}

for model in models:   
    for image_path in image_paths:
        image = load_img(image_path, target_size = (256, 256))
        image_array = img_to_array(image)
        image_array = np.expand_dims(image_array, axis = 0)
        image_array /= 255.0
    
        predictions = model.predict(image_array)
        predicted_probs = predictions[0] * 100
        predicted_label = class_labels[np.argmax(predicted_probs)]
    
        true_label = true_labels[image_path]
    
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
        plt.imshow(image, cmap = 'gray')
        plt.axis('off')
        plt.title(f"True Class Classification Label: {true_label}\nPredicted Class Classification Label: {predicted_label}")
    
        text_y_position = 30
        spacing = 35
        for i, (label, prob) in enumerate(zip(class_labels, predicted_probs)):
            plt.text(10, text_y_position, f"{label}: {prob:.1f}%", color='green', fontsize = 12)
            text_y_position += spacing
    
        plt.show()
