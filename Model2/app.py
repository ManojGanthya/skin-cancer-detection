from flask import Flask, render_template, request, jsonify
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

# Load the pre-trained Keras model
model = load_model('skin_cancer_detection_model_11.keras')

# Function to preprocess image for model prediction
def preprocess_image(image_path):
    try:
        img = image.load_img(image_path, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array / 255.0  # Normalize pixel values
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {str(e)}")

# Function to predict the class and confidence
def predict_class(image_path):
    try:
        preprocessed_image = preprocess_image(image_path)
        predictions = model.predict(preprocessed_image)
        class_index = np.argmax(predictions)
        confidence = float(predictions[0][class_index])
        return class_index, confidence
    except Exception as e:
        raise ValueError(f"Error predicting image: {str(e)}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Save the uploaded file temporarily
        upload_dir = 'uploads'
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)
        file.save(file_path)

        # Make prediction
        class_index, confidence = predict_class(file_path)
        classes = ['actinic_keratosis', 'basal_cell_carcinoma', 'dermatofibroma', 'melanoma', 'nevus', 'pigmented_benign_keratosis', 'seborrheic_keratosis', 'squamous_cell_carcinoma', 'vascular_lesion']  # Replace with your actual classes
        class_name = classes[class_index]
        accuracy = confidence * 100

        # Return prediction result
        return jsonify({
            'class': class_name,
            'confidence': confidence,
            'accuracy': accuracy
        }), 200

    except ValueError as ve:
        return jsonify({'error': str(ve)}), 500  # Return detailed error message
    except Exception as e:
        return jsonify({'error': 'Unexpected error occurred: ' + str(e)}), 500  # Generic error message

if __name__ == '__main__':
    app.run(debug=True)
