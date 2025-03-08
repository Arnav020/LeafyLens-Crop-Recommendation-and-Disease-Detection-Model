from flask import Flask, render_template, request, redirect, url_for, flash
import os
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
from PIL import Image
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
dataset = pd.read_csv('Crop_recommendation.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# OneHotEncode the labels
encoder = OneHotEncoder(sparse_output=False)
y = y.reshape(-1, 1)
y_encoded = encoder.fit_transform(y)

# Train the RandomForest model
classifier = RandomForestClassifier(n_estimators=100, random_state=0)
classifier.fit(x, y_encoded)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'  # Directory to temporarily store uploaded images
app.secret_key = 'your_secret_key'

# Constants for image dimensions
IMG_SIZE = 128
img_height, img_width = IMG_SIZE, IMG_SIZE

# Load the saved models and class indices for both crops
def load_models_for_crop(crop_type):
    try:
        if crop_type == 'maize':
            # Maize uses CNN and VGG16 models
            print("Loading maize models...")
            cnn_model = load_model('CNN_final.keras')
            vgg16_model = load_model('VGG16_final.keras')
            print("Maize models loaded successfully.")
            with open('maize_class_indices.pkl', 'rb') as f:
                class_indices = pickle.load(f)
            return [cnn_model, vgg16_model], class_indices
        elif crop_type == 'cashew':
            # Cashew uses CNN and MLP models
            print("Loading cashew models...")
            cnn_model = load_model('cnn_cashew.keras')
            mlp_model = load_model('mlp_cashew.keras')
            print("Cashew models loaded successfully.")
            with open('cashew_class_indices.pkl', 'rb') as f:
                class_indices = pickle.load(f)
            return [cnn_model, mlp_model], class_indices
        else:
            print("Invalid crop type.")
            return None, None
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return None, None

# Solution dictionary for crop diseases
disease_solutions = {
    'Cashew anthracnose': "In the initial stages of infection, prune the affected parts and burn the same. "
                          "This will prevent the spread of the disease. Effective control can be obtained by "
                          "cutting and removal of the affected plant parts and by spraying Bordeaux mixture 1% or "
                          "mancozeb 0.2% or copper oxychloride 0.3%.",
    'Maize rust': "Spray mancozeb (0.25%) at the first appearance of pustules. Ensure proper irrigation and use "
                  "resistant varieties when planting. Avoid overcrowding and maintain good plant spacing.",
    # Add more disease solutions here as needed
}

# Prediction function for the growth stage using multiple models
def predict_image(image_path, models, class_indices):
    try:
        print(f"Starting prediction for image: {image_path}")
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_height, img_width))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize image

        # Get predictions from all models
        predictions = [model.predict(img_array) for model in models]
        print("Predictions from models:", predictions)

        # Average the predictions (ensemble method)
        ensemble_pred = np.mean(predictions, axis=0)
        predicted_class_index = np.argmax(ensemble_pred)

        # Convert class index to class name
        class_names = {v: k for k, v in class_indices.items()}
        predicted_class_name = class_names.get(predicted_class_index, "Unknown")
        print("Predicted class name:", predicted_class_name)

        return predicted_class_name
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

# Sowing Stage route
@app.route('/sowing_stage', methods=['GET', 'POST'])
def sowing_stage():
    if request.method == 'POST':
        # Get form data
        nitrogen = float(request.form['nitrogen'])
        phosphorus = float(request.form['phosphorus'])
        potassium = float(request.form['potassium'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        # Create input vector
        x_new = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
        
        # Make predictions
        y_pred_encoded = classifier.predict(x_new)
        y_pred = encoder.inverse_transform(y_pred_encoded)

        # Print the recommended crop type to the console
        print(f'Recommended Crop Type: {y_pred[0][0]}')

        return render_template('sowing_stage.html', prediction=y_pred[0][0])
    
    return render_template('sowing_stage.html', prediction=None)

# Growth Stage route
@app.route('/growth_stage', methods=['GET', 'POST'])
def growth_stage():
    if request.method == 'POST':
        crop_type = request.form.get('crop_type')
        print(f"Selected crop type: {crop_type}")

        if 'file' not in request.files or crop_type not in ['maize', 'cashew']:
            flash('Please select a valid crop and upload an image.')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and file.filename.endswith(('jpg', 'jpeg', 'png')):
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)  # Save the uploaded file
            print(f"File {filename} saved successfully.")

            # Load the appropriate models based on crop type
            models, class_indices = load_models_for_crop(crop_type)

            if not models:
                flash('Error loading models for the selected crop.')
                return redirect(request.url)

            # Predict the class of the uploaded image
            predicted_class = predict_image(filename, models, class_indices)

            # Remove the uploaded file after prediction
            os.remove(filename)

            # Define solutions for predicted diseases
            solutions = {
                'Cashew anthracnose': (
                    "In the early stages of infection, it's important to trim and dispose of the affected areas of the plant "
                    "by burning them. This helps stop the disease from spreading. To effectively manage the infection, "
                    "remove the damaged parts of the plant and apply a Bordeaux mixture at 1%, or use mancozeb at "
                    "0.2% or copper oxychloride at 0.3%."
                )
                # Add more diseases and their solutions here
            }

            if predicted_class:
                print(f"Prediction: {predicted_class}")
                solution = solutions.get(predicted_class, "No specific solution available for this disease.")
                return render_template('growth_stage.html', prediction=predicted_class, solution=solution)
            else:
                flash('An error occurred during prediction.')
                return redirect(request.url)

    return render_template('growth_stage.html')

if __name__ == '__main__':
    app.run(debug=True)
