# Complete Technology Documentation - Anxiety Detection Project

## Table of Contents
1. [Project Overview](#project-overview)
2. [Technology Stack](#technology-stack)
3. [Backend Technologies](#backend-technologies)
4. [Machine Learning Technologies](#machine-learning-technologies)
5. [Frontend Technologies](#frontend-technologies)
6. [Data Processing Technologies](#data-processing-technologies)
7. [Model Architecture & Formats](#model-architecture--formats)
8. [System Architecture](#system-architecture)
9. [API Endpoints](#api-endpoints)
10. [Data Flow](#data-flow)
11. [Deployment Considerations](#deployment-considerations)

---

## Project Overview

This project is a **web-based anxiety detection system** that uses machine learning to analyze physiological data (Heart Rate, Electrodermal Activity, and Skin Temperature) to detect and classify anxiety levels. The system provides two prediction modes:
- **Binary Classification**: Detects presence/absence of anxiety
- **Multiclass Classification**: Classifies anxiety into three levels (Low, Moderate, High)

---

## Technology Stack

### Core Technologies
- **Python 3.12**: Primary programming language
- **Flask 2.0.1**: Web framework for backend API
- **TensorFlow 2.6.0**: Deep learning framework
- **Keras**: High-level neural network API (included in TensorFlow)
- **NumPy 1.21.2**: Numerical computing library
- **scikit-learn 0.24.2**: Machine learning utilities for preprocessing
- **joblib 1.0.1**: Model serialization and persistence

### Frontend Technologies
- **HTML5**: Markup language
- **Bootstrap 5.3.0**: CSS framework for responsive UI
- **Font Awesome 6.0.0**: Icon library
- **CSS3**: Custom styling
- **JavaScript**: Client-side interactivity (via Bootstrap)

---

## Backend Technologies

### 1. Flask Framework

**Purpose**: Flask serves as the web application framework that handles HTTP requests, routes, and template rendering.

**Usage in Project**:
```python
from flask import Flask, request, render_template
app = Flask(__name__)
```

**Key Features Used**:
- **Route Handling**: Defines endpoints (`/` and `/predict`, `/predict_multiclass`)
- **Request Processing**: Extracts form data using `request.form`
- **Template Rendering**: Uses Jinja2 templating engine (built into Flask) to render HTML templates with dynamic data
- **Debug Mode**: Runs in development mode with `debug=True` for hot-reloading

**Routes Implemented**:
- `GET /`: Home page route that renders the main interface
- `POST /predict`: Binary classification endpoint
- `POST /predict_multiclass`: Multiclass classification endpoint

**Why Flask?**
- Lightweight and minimalistic
- Easy to set up and deploy
- Excellent for ML model serving
- Built-in development server
- Flexible routing system

---

### 2. NumPy

**Purpose**: NumPy provides efficient array operations and numerical computations required for data preprocessing and model inference.

**Usage in Project**:
```python
import numpy as np

# Creating input arrays
input_array = np.array([[input1, input2, input3]], dtype=np.float32)

# Array reshaping for time series
features_scaled = features_scaled.reshape(1, timestep, 3)

# Finding maximum probability index
predicted_class = np.argmax(prediction, axis=1)[0]
```

**Key Operations**:
- **Array Creation**: Converts Python lists/values to NumPy arrays
- **Data Type Specification**: Uses `dtype=np.float32` for TensorFlow Lite compatibility
- **Array Reshaping**: Transforms 2D arrays to 3D for time series models `(samples, timesteps, features)`
- **Argmax Operation**: Finds the index of maximum value in prediction probabilities

**Why NumPy?**
- Essential for TensorFlow/Keras operations
- Efficient memory management
- Fast mathematical operations
- Standard format for ML data exchange

---

## Machine Learning Technologies

### 3. TensorFlow & Keras

**Purpose**: TensorFlow is the core deep learning framework, and Keras provides a high-level API for building and loading neural network models.

**Usage in Project**:

#### A. TensorFlow Lite (TFLite)
```python
import tensorflow as tf

# Loading TFLite model for binary classification
binary_interpreter = tf.lite.Interpreter(model_path="final_model_federated.tflite")
binary_interpreter.allocate_tensors()

# Getting input/output tensor details
binary_input_details = binary_interpreter.get_input_details()
binary_output_details = binary_interpreter.get_output_details()

# Running inference
binary_interpreter.set_tensor(binary_input_details[0]['index'], scaled_input)
binary_interpreter.invoke()
prediction = binary_interpreter.get_tensor(binary_output_details[0]['index'])
```

**TFLite Benefits**:
- **Optimized for Mobile/Edge**: Smaller model size, faster inference
- **Lower Memory Footprint**: Ideal for production deployments
- **Cross-Platform**: Can run on various devices (mobile, IoT, embedded)
- **Quantization Support**: Can use quantized models for even smaller size

**Why TFLite for Binary Model?**
- The binary classification model uses TFLite format for optimized inference
- Faster prediction times
- Lower resource consumption

#### B. Keras H5 Model Format
```python
from keras.models import load_model

# Loading multiclass time series model
multiclass_model = load_model('final_model_time_series_multiclass_timeseries_2.h5')

# Making predictions
prediction = multiclass_model.predict(features_scaled)
```

**H5 Format Benefits**:
- **Complete Model Storage**: Saves architecture, weights, and optimizer state
- **Easy Loading**: Single command to load entire model
- **Compatibility**: Works seamlessly with Keras/TensorFlow ecosystem

**Model Types**:
1. **Binary Classification Model** (`final_model_federated.tflite`):
   - Input: 3 features (HR, EDA, Skin Temperature)
   - Output: Binary probability (anxiety/no anxiety)
   - Architecture: Federated learning model (indicated by filename)

2. **Multiclass Time Series Model** (`final_model_time_series_multiclass_timeseries_2.h5`):
   - Input: Time series data with 4 timesteps × 3 features
   - Output: 3 classes (Low, Moderate, High anxiety)
   - Architecture: Likely LSTM/GRU or CNN-based time series model

---

### 4. scikit-learn

**Purpose**: scikit-learn provides preprocessing utilities for data normalization and label encoding.

**Usage in Project**:

#### A. MinMaxScaler
```python
from sklearn.preprocessing import MinMaxScaler

# Loaded via joblib
binary_scaler = joblib.load('scaler.save')
multiclass_scaler = joblib.load('multiclass_timeseries_scaler.save')

# Transforming input data
scaled_input = binary_scaler.transform(input_array)
features_scaled = multiclass_scaler.transform(features)
```

**Purpose of Scaling**:
- **Normalization**: Scales features to a common range (typically 0-1)
- **Model Compatibility**: Neural networks perform better with normalized inputs
- **Consistency**: Ensures prediction data matches training data distribution

**Why Two Scalers?**
- Different models were trained with different preprocessing pipelines
- Binary model uses one scaler for 3 features
- Multiclass model uses another scaler for time series data

#### B. LabelEncoder
```python
# Loaded via joblib
multiclass_label_encoder = joblib.load('multiclass_timeseries_label_encoder.save')

# Inverse transform to get class label
label = multiclass_label_encoder.inverse_transform([predicted_class])[0]
```

**Purpose**:
- **Label Mapping**: Converts numeric predictions (0, 1, 2) to meaningful labels
- **Inverse Transformation**: Maps model output back to original class names
- **Consistency**: Ensures same encoding used during training and inference

**Class Mapping** (inferred from code):
- `0`: Low Anxiety
- `1`: Moderate Anxiety
- `2`: High Anxiety

---

### 5. joblib

**Purpose**: joblib is used for efficient serialization and deserialization of Python objects, particularly NumPy arrays and scikit-learn objects.

**Usage in Project**:
```python
import joblib

# Loading saved preprocessing objects
binary_scaler = joblib.load('scaler.save')
multiclass_scaler = joblib.load('multiclass_timeseries_scaler.save')
multiclass_label_encoder = joblib.load('multiclass_timeseries_label_encoder.save')
```

**Why joblib?**
- **Efficient for NumPy**: Optimized for large NumPy arrays
- **Preserves State**: Saves complete object state including fitted parameters
- **Fast Loading**: Quick deserialization compared to pickle
- **Standard Practice**: Industry standard for ML model persistence

**Saved Objects**:
- `scaler.save`: MinMaxScaler fitted on binary classification training data
- `multiclass_timeseries_scaler.save`: MinMaxScaler fitted on time series training data
- `multiclass_timeseries_label_encoder.save`: LabelEncoder with class mappings

---

## Frontend Technologies

### 6. HTML5

**Purpose**: Provides the structure and semantic markup for the web interface.

**Key Features Used**:
- **Semantic Elements**: Proper use of HTML5 semantic tags
- **Form Elements**: Input fields for user data collection
- **Meta Tags**: Viewport settings for responsive design
- **Template Integration**: Jinja2 templating for dynamic content

**Structure**:
- Form-based input collection
- Conditional rendering based on prediction results
- Integration with Bootstrap grid system

---

### 7. Bootstrap 5.3.0

**Purpose**: Bootstrap provides a responsive CSS framework with pre-built components and utilities.

**Usage**:
```html
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
```

**Key Components Used**:
- **Grid System**: `container`, `row`, `col-lg-6` for responsive layout
- **Form Controls**: `form-control`, `form-label` for styled inputs
- **Buttons**: Custom styled buttons with Bootstrap classes
- **Utilities**: Margin, padding, text alignment classes

**Benefits**:
- **Responsive Design**: Mobile-first approach
- **Consistent Styling**: Professional appearance out of the box
- **Component Library**: Pre-built UI components
- **Cross-Browser Compatibility**: Works across all modern browsers

---

### 8. Font Awesome 6.0.0

**Purpose**: Provides scalable vector icons for enhanced UI/UX.

**Usage**:
```html
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
```

**Icons Used**:
- `fa-brain`: Main header icon
- `fa-search`: Analyze button icon
- `fa-exclamation-triangle`: Warning status icon
- `fa-check-circle`: Success status icon
- `fa-leaf`: Suggestion card icons
- `fa-heart`: Footer icon
- `fa-code`: Technology indicator
- `fa-layer-group`: Multiclass prediction icon
- `fa-info-circle`: Information icon

**Benefits**:
- **Visual Enhancement**: Makes UI more intuitive and engaging
- **Scalable**: Vector-based icons that scale without quality loss
- **Consistent Style**: Unified icon design language

---

### 9. CSS3

**Purpose**: Custom styling to enhance the Bootstrap base design.

**Key CSS Features Used**:

#### Custom Styling:
```css
/* Gradient backgrounds */
background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);

/* Box shadows */
box-shadow: 0 0.5rem 1rem rgba(0, 0, 0, 0.15);

/* Border radius */
border-radius: 15px;

/* Transitions */
transition: transform 0.3s ease;

/* Flexbox */
display: flex;
align-items: flex-start;
```

**Design Elements**:
- **Gradient Backgrounds**: Modern purple-to-blue gradients
- **Card-based Layout**: Elevated cards with shadows
- **Color-coded Results**: Green for success, yellow for warnings
- **Hover Effects**: Interactive button animations
- **Responsive Typography**: Scalable font sizes

---

## Data Processing Technologies

### 10. Data Preprocessing Pipeline

**Binary Classification Pipeline**:
1. **Input Collection**: 3 features (HR, EDA, Skin Temperature)
2. **Array Conversion**: Convert to NumPy array with `dtype=np.float32`
3. **Scaling**: Apply MinMaxScaler transformation
4. **Model Inference**: Pass to TFLite interpreter
5. **Threshold Application**: Convert probability to binary class (>= 0.5)

**Multiclass Classification Pipeline**:
1. **Input Collection**: 4 timesteps × 3 features (12 total inputs)
2. **Array Construction**: Build 2D array (4, 3)
3. **Scaling**: Apply time series scaler
4. **Reshaping**: Transform to 3D array (1, 4, 3) for time series model
5. **Model Inference**: Pass to Keras model
6. **Class Prediction**: Use argmax to find predicted class
7. **Label Decoding**: Convert numeric class to text label

---

## Model Architecture & Formats

### Model Files

1. **`final_model_federated.tflite`**
   - **Format**: TensorFlow Lite
   - **Type**: Binary Classification
   - **Input Shape**: (1, 3) - Single sample with 3 features
   - **Output Shape**: (1, 1) - Single probability value
   - **Inference Engine**: TFLite Interpreter
   - **Use Case**: Fast, lightweight binary anxiety detection

2. **`final_model_time_series_multiclass_timeseries_2.h5`**
   - **Format**: Keras H5
   - **Type**: Multiclass Time Series Classification
   - **Input Shape**: (batch_size, 4, 3) - 4 timesteps, 3 features
   - **Output Shape**: (batch_size, 3) - Probabilities for 3 classes
   - **Inference Engine**: Keras Model
   - **Use Case**: Detailed anxiety level classification

3. **Preprocessing Artifacts**:
   - `scaler.save`: Binary model scaler
   - `multiclass_timeseries_scaler.save`: Time series scaler
   - `multiclass_timeseries_label_encoder.save`: Class label mappings

---

## System Architecture

### High-Level Architecture

```
┌─────────────────┐
│   Web Browser   │
│  (Frontend UI)  │
└────────┬────────┘
         │ HTTP POST
         ▼
┌─────────────────┐
│   Flask Server  │
│   (app.py)      │
└────────┬────────┘
         │
         ├──► Route Handler
         │    ├──► /predict (Binary)
         │    └──► /predict_multiclass
         │
         ├──► Data Preprocessing
         │    ├──► NumPy Array Creation
         │    ├──► MinMaxScaler
         │    └──► Data Reshaping
         │
         ├──► Model Inference
         │    ├──► TFLite Interpreter (Binary)
         │    └──► Keras Model (Multiclass)
         │
         ├──► Post-Processing
         │    ├──► Threshold Application
         │    ├──► Argmax Operation
         │    └──► Label Decoding
         │
         └──► Response Generation
              └──► Template Rendering
```

### Component Interaction Flow

1. **User Input** → HTML Form Submission
2. **Flask Route** → Receives POST request
3. **Data Extraction** → `request.form` parsing
4. **Preprocessing** → NumPy + scikit-learn scaling
5. **Model Loading** → TFLite/Keras model inference
6. **Post-Processing** → Class prediction + label mapping
7. **Template Rendering** → Jinja2 template with results
8. **Response** → HTML page with predictions and suggestions

---

## API Endpoints

### 1. GET `/`
**Purpose**: Renders the main application interface

**Response**: HTML page (`index_2.html`)

**Features**:
- Two input forms (binary and multiclass)
- Welcome message
- Instructions for users

---

### 2. POST `/predict`
**Purpose**: Binary anxiety detection

**Request Parameters**:
- `input1` (float): Heart Rate (BPM)
- `input2` (float): Electrodermal Activity (EDA)
- `input3` (float): Skin Temperature

**Processing Steps**:
1. Extract form data
2. Convert to NumPy array (float32)
3. Apply binary scaler
4. Run TFLite inference
5. Apply threshold (0.5)
6. Generate suggestions based on result

**Response**: HTML page with:
- `prediction_text`: "Anxiety Detected" or "No Anxiety Detected"
- `status`: "warning" or "success"
- `message`: Contextual message
- `suggestions`: List of health suggestions

---

### 3. POST `/predict_multiclass`
**Purpose**: Multiclass anxiety level classification

**Request Parameters**:
- `input1_1` to `input1_4` (float): Heart Rate for 4 timesteps
- `input2_1` to `input2_4` (float): EDA for 4 timesteps
- `input3_1` to `input3_4` (float): Skin Temperature for 4 timesteps

**Processing Steps**:
1. Extract 12 input values (4 timesteps × 3 features)
2. Build 2D array (4, 3)
3. Apply multiclass scaler
4. Reshape to 3D (1, 4, 3)
5. Run Keras model prediction
6. Find argmax for class prediction
7. Decode label using LabelEncoder
8. Generate class-specific suggestions

**Response**: HTML page with:
- `prediction_text`: "Low/Moderate/High Anxiety Detected"
- `status`: "success" (all use success for multiclass)
- `message`: Contextual message based on anxiety level
- `suggestions`: Level-specific health suggestions

**Error Handling**:
- Try-except block catches exceptions
- Returns error message in template
- Validates all inputs are provided

---

## Data Flow

### Binary Classification Data Flow

```
User Input (3 values)
    ↓
NumPy Array: shape (1, 3), dtype float32
    ↓
MinMaxScaler Transform
    ↓
TFLite Interpreter Input
    ↓
Model Inference
    ↓
Output: Probability [0-1]
    ↓
Threshold Check (>= 0.5)
    ↓
Binary Class (0 or 1)
    ↓
Result Mapping + Suggestions
```

### Multiclass Classification Data Flow

```
User Input (12 values: 4 timesteps × 3 features)
    ↓
NumPy Array: shape (4, 3)
    ↓
MinMaxScaler Transform
    ↓
Reshape: (1, 4, 3) for time series
    ↓
Keras Model Input
    ↓
Model Inference
    ↓
Output: Probabilities [3 classes]
    ↓
Argmax: Find highest probability class
    ↓
LabelEncoder Inverse Transform
    ↓
Class Label (0, 1, or 2)
    ↓
Result Mapping + Level-Specific Suggestions
```

---

## Deployment Considerations

### Development Environment
- **Server**: Flask development server (`app.run(debug=True)`)
- **Port**: Default Flask port (5000)
- **Hot Reloading**: Enabled via debug mode

### Production Considerations

#### 1. **Web Server Options**:
   - **Gunicorn**: WSGI HTTP Server for production
   - **uWSGI**: Alternative WSGI server
   - **Waitress**: Cross-platform production server

#### 2. **Model Optimization**:
   - TFLite models already optimized for deployment
   - Consider model quantization for further size reduction
   - Batch processing for multiple predictions

#### 3. **Scalability**:
   - Load balancing for multiple instances
   - Model caching to avoid reloading
   - Async processing for time-consuming predictions

#### 4. **Security**:
   - Input validation and sanitization
   - Rate limiting for API endpoints
   - HTTPS for secure data transmission
   - Error handling to prevent information leakage

#### 5. **Monitoring**:
   - Logging for prediction requests
   - Performance metrics (latency, throughput)
   - Error tracking and alerting

---

## Technology Integration Summary

### Backend Stack
- **Flask**: Web framework and routing
- **NumPy**: Numerical computations
- **TensorFlow/Keras**: Deep learning inference
- **scikit-learn**: Data preprocessing
- **joblib**: Model persistence

### Frontend Stack
- **HTML5**: Structure
- **Bootstrap 5**: Responsive UI framework
- **Font Awesome**: Icons
- **CSS3**: Custom styling
- **Jinja2**: Template engine (Flask)

### ML Pipeline
- **TFLite**: Optimized binary classification
- **Keras H5**: Time series multiclass classification
- **MinMaxScaler**: Feature normalization
- **LabelEncoder**: Class label mapping

### Data Flow Technologies
- **NumPy Arrays**: Data structure for ML operations
- **Type Conversion**: float32 for TFLite compatibility
- **Array Reshaping**: Time series data preparation

---

## Key Design Decisions

1. **Dual Model Approach**: 
   - TFLite for fast binary detection
   - Keras H5 for detailed multiclass analysis

2. **Separate Preprocessing**:
   - Different scalers for different models
   - Ensures training/inference consistency

3. **Template-Based Rendering**:
   - Server-side rendering for simplicity
   - Dynamic content based on predictions

4. **Error Handling**:
   - Try-except blocks for robustness
   - User-friendly error messages

5. **Responsive Design**:
   - Bootstrap for mobile compatibility
   - Modern UI with gradients and animations

---

## Future Enhancement Opportunities

1. **API Endpoints**: Convert to REST API with JSON responses
2. **Real-time Processing**: WebSocket support for streaming data
3. **Model Versioning**: Support for multiple model versions
4. **Caching**: Redis for model and preprocessing object caching
5. **Database Integration**: Store predictions and user history
6. **Authentication**: User accounts and prediction history
7. **Visualization**: Charts and graphs for time series data
8. **Mobile App**: Native mobile application using TFLite models

---

## Conclusion

This project demonstrates a complete machine learning web application stack, integrating:
- **Web Framework** (Flask) for serving
- **Deep Learning** (TensorFlow/Keras) for predictions
- **Data Science** (NumPy, scikit-learn) for preprocessing
- **Modern Frontend** (Bootstrap, Font Awesome) for UX
- **Model Optimization** (TFLite) for performance

The architecture is designed for both development and production use, with clear separation of concerns and efficient data processing pipelines.

---

**Documentation Version**: 1.0  
**Last Updated**: 2024  
**Project**: Anxiety Detection System

