# LeafyLens

LeafyLens is a web-based portal designed to assist farmers in two crucial stages of crop growth: the **Sowing Stage** and the **Growth Stage**. It integrates machine learning models to provide crop recommendations and disease detection, ensuring better agricultural outcomes.

## Features

### 🌱 Sowing Stage
- **Crop Recommendation System**
- Uses **Random Forest Regression**
- Considers soil parameters: Nitrogen, Phosphorus, Potassium levels
- Incorporates weather conditions: Rainfall, Temperature, Humidity
- Recommends the most suitable crop based on the provided data

### 🌾 Growth Stage
- **Crop Disease Detection Model**
- Built using **Convolutional Neural Networks (CNN)**
- Supports **pre-trained models like ResNet50 and VGG16**
- Works with datasets of various crop diseases
- Users can upload multiple-angle images of a crop for analysis
- Identifies diseases and suggests potential solutions

## Models Used
- **Maize:** CNN, VGG16
- **Cashew:** CNN, MLP

## Tech Stack
- **Backend:** Flask
- **Frontend:** HTML, CSS, JavaScript
- **Machine Learning:** TensorFlow, Keras, OpenCV, Pandas, NumPy
- **Dataset Processing:** Image preprocessing and augmentation techniques

   ```
   
## Usage
1. Select **Sowing Stage** or **Growth Stage**.
2. For **Sowing Stage**, enter soil and weather details to get a crop recommendation.
3. For **Growth Stage**, select the crop type, upload images, and get disease predictions with solutions.

## Contributing
Pull requests are welcome. For significant changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer
This project is intended for informational and educational purposes only. While the models strive to provide accurate crop recommendations and disease detection, they should not replace professional agricultural advice. Users should verify results with agricultural experts before making critical decisions.



