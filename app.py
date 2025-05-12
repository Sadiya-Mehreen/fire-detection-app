from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import os
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load your pre-trained model (relative path for Render deployment)
model = load_model('models/custom_fire_detection_model.h5')

# Set threshold for fire detection
threshold = 0.4

# Setup upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Route for homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    # Save file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Load and preprocess image
    img = Image.open(filepath)
    img = img.resize((64, 64))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    result = "🔥 Fire detected!" if prediction[0] > threshold else "❌ No fire detected."

    return render_template('index.html', result=result, image_file=file.filename)

if __name__ == "__main__":
    app.run(debug=True)
