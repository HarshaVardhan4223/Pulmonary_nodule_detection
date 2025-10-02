import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import sys
import os

# ✅ Load the trained model
model_path = "models/pulmonary_nodule_model.keras"

if not os.path.exists(model_path):
    print("⚠️ Model file not found! Check if training completed successfully.")
    sys.exit()

model = tf.keras.models.load_model(model_path)
print("✅ Model loaded successfully!")

# ✅ Function to preprocess the image
def preprocess_image(image_path):
    if not os.path.exists(image_path):
        print("⚠️ Image file not found! Check the path and try again.")
        sys.exit()
    
    img = image.load_img(image_path, target_size=(224, 224))  # Resize
    img_array = image.img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize
    return img_array

# ✅ Get image path from command line
if len(sys.argv) < 2:
    print("⚠️ Please provide an image path!")
    sys.exit()

image_path = sys.argv[1]
input_image = preprocess_image(image_path)

# ✅ Make prediction
prediction = model.predict(input_image)
predicted_class = np.argmax(prediction, axis=1)  # Get highest probability class
confidence = np.max(prediction) * 100  # Confidence Score

# ✅ Define class labels (Adjust based on your dataset)
class_labels = ["Benign", "Malignant", "Unlabeled"]

# ✅ Display Prediction Result
print(f"🩺 Prediction: {class_labels[predicted_class[0]]} with confidence {confidence:.2f}%")
