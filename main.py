from flask import Flask, request, render_template
import numpy as np
from PIL import Image
import tensorflow as tf
import json

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("plant_disease_prediction_model.h5")

# Load the class indices
with open("class_indices.json", "r") as file:
    class_indices = json.load(file)

class_names = {int(k): v for k, v in class_indices.items()}

def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = image_array.astype("float32") / 255.0
    return image_array

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files or request.files['file'].filename == '':
        return render_template("index.html", error="Please upload an image file.")

    file = request.files['file']
    try:
        # Open the image file
        image = Image.open(file)
        # Preprocess the image
        processed_image = preprocess_image(image)
        # Predict the class
        predictions = model.predict(processed_image)
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class_name = class_names[predicted_class_index]
        confidence = np.max(predictions) * 100
        return render_template("result.html",
                               predicted_class=predicted_class_name,
                               confidence=confidence)
    except Exception as e:
        return render_template("index.html", error="Error processing the image. Please try again.")

if __name__ == "__main__":
    app.run(debug=True)
