import os
import numpy as np
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from PIL import Image
import cv2

app = Flask(__name__)

# Load the model
model = load_model('BrainTumor10EpochsCategorical.h5')
print('Model loaded. Check http://127.0.0.1:5000/')


def get_className(classNo):
    if classNo == 0:
        return "No Brain Tumor"
    elif classNo == 1:
        return "Yes Brain Tumor"


def getResult(img_path):
    # Load image
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((64, 64))
    image = np.array(image)

    # Prepare the image for prediction
    input_img = np.expand_dims(image, axis=0)

    # Make prediction
    predicted_probabilities = model.predict(input_img)
    predicted_class_index = np.argmax(predicted_probabilities)
    predicted_class = get_className(predicted_class_index)
    
    print("Predicted Class:", predicted_class)
    return predicted_class_index


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        f.save(file_path)

        value = getResult(file_path)
        result = get_className(value)
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)
