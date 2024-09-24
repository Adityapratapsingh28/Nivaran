from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import pickle
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session management

# Load models
base_dir = r"C:\health_tech"

brain_tumor_model = load_model(os.path.join(base_dir, 'model_brain_tumor.h5'))
with open(os.path.join(base_dir, 'model_COPD.pkl'), 'rb') as f:
    copd_model = pickle.load(f)

xray_model = load_model(os.path.join(base_dir, 'model_bone_fracture.h5'))
lung_cancer_model = load_model(os.path.join(base_dir, 'model_chest_cancer.h5'))

with open(os.path.join(base_dir, 'model_diabetes.pkl'), 'rb') as f:
    diabetes_model = pickle.load(f)

with open(os.path.join(base_dir, 'model_cancer.pkl'), 'rb') as f:
    cancer_model = pickle.load(f)


# Helper function to preprocess images
def preprocess_image(img, target_size):
    img = img.resize(target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype(np.float32) / 255.0
    return img_array

@app.route('/disease')
def disease():
    disease_name = request.args.get('disease')
    # You can use disease_name to customize the content for each disease
    return render_template('disease.html', disease=disease_name)

# Brain Tumor Prediction
@app.route('/brain_tumor', methods=['POST'])
def predict_brain_tumor():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error="No file part")
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="No selected file")
        if file:
            img = Image.open(file.stream)
            img_array = preprocess_image(img, (150, 150))
            prediction = brain_tumor_model.predict(img_array)
            class_names = ['No Tumor', 'Glioma Tumor', 'Meningioma Tumor', 'Pituitary Tumor']
            predicted_class = class_names[np.argmax(prediction)]
            confidence = float(np.max(prediction)) * 100
            print(predicted_class)
            print(confidence)
            
            result = {
                'predicted_class': predicted_class,
                'confidence': confidence
            }
            print(f"Brain Tumor Prediction: {predicted_class}, Confidence: {confidence:.2f}%")
            return render_template('index.html', result=result)
    return render_template('index.html')



# COPD Prediction
@app.route('/copd', methods=['POST'])
def predict_copd():
    features = request.get_json().get('features', {})
    if not features:
        return redirect(url_for('result4'))
    input_features = [[
        features.get('AGE', 0), features.get('PackHistory', 0), features.get('MWT1Best', 0), features.get('FEV1', 0), features.get('FVC', 0),
        features.get('CAT', 0), features.get('HAD', 0), features.get('SGRQ', 0), features.get('gender', 0), features.get('smoking', 0),
        features.get('Diabetes', 0), features.get('muscular', 0), features.get('hypertension', 0), features.get('AtrialFib', 0), features.get('IHD', 0)
    ]]
    
    prediction = copd_model.predict(input_features)
    class_names = ['GOLD 1: Mild COPD', 'GOLD 2: Moderate COPD', 'GOLD 3: Severe COPD', 'GOLD 4: Very Severe COPD']
    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction)) * 100
    
    # Store results in session
    session['result'] = {
        'predicted_class': predicted_class,
        'confidence': confidence
    }
    
    return redirect(url_for('result4'))


# X-ray Fracture Prediction
@app.route('/xray', methods=['POST'])
def predict_xray():
    if 'file' not in request.files:
        return redirect(url_for('result3'))
    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
        return redirect(url_for('result3'))
    if uploaded_file:
        img = Image.open(uploaded_file)
        img_array = preprocess_image(img, (150, 150))
        prediction = xray_model.predict(img_array)
        class_names = ['Fractured', 'Non-Fractured']
        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction)) * 100
        
        # Store results in session
        session['result'] = {
            'predicted_class': predicted_class,
            'confidence': confidence
        }
        
        return redirect(url_for('result3'))

    return redirect(url_for('result3'))


# Lung Cancer Prediction
@app.route('/lung_cancer', methods=['POST'])
def predict_lung_cancer():
    if 'file' not in request.files:
        return redirect(url_for('result6'))
    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
        return redirect(url_for('result6'))
    if uploaded_file:
        img = Image.open(uploaded_file)
        img_array = preprocess_image(img, (150, 150))
        prediction = lung_cancer_model.predict(img_array)
        predicted_class_probability = float(np.max(prediction))
        predicted_class_name = 'squamous.cell.carcinoma' if predicted_class_probability > 0.5 else 'not squamous.cell.carcinoma'
        
        # Store results in session
        session['result'] = {
            'predicted_class_name': predicted_class_name,
            'predicted_class_probability': predicted_class_probability
        }
        
        return redirect(url_for('result6'))

    return redirect(url_for('result6'))


# General Cancer Prediction (using other model)
@app.route('/cancer', methods=['POST'])
def predict_cancer():
    features = request.get_json().get('features', {})
    if not features:
        return redirect(url_for('result2'))
    input_features = [[
        features.get('radius_mean', 0), features.get('texture_mean', 0), features.get('perimeter_mean', 0), features.get('area_mean', 0), features.get('smoothness_mean', 0),
        features.get('compactness_mean', 0), features.get('concavity_mean', 0), features.get('concave points_mean', 0), features.get('symmetry_mean', 0),
        features.get('fractal_dimension_mean', 0), features.get('radius_se', 0), features.get('texture_se', 0), features.get('perimeter_se', 0), features.get('area_se', 0),
        features.get('smoothness_se', 0), features.get('compactness_se', 0), features.get('concavity_se', 0), features.get('concave points_se', 0), features.get('symmetry_se', 0),
        features.get('fractal_dimension_se', 0), features.get('radius_worst', 0), features.get('texture_worst', 0), features.get('perimeter_worst', 0), features.get('area_worst', 0),
        features.get('smoothness_worst', 0), features.get('compactness_worst', 0), features.get('concavity_worst', 0), features.get('concave points_worst', 0),
        features.get('symmetry_worst', 0), features.get('fractal_dimension_worst', 0)
    ]]
    
    prediction = cancer_model.predict(input_features)
    predicted_class = 'Malignant' if prediction[0] > 0.5 else 'Benign'
    confidence = float(prediction[0]) * 100
    
    # Store results in session
    session['result'] = {
        'predicted_class': predicted_class,
        'confidence': confidence
    }
    
    return redirect(url_for('result2'))


# Diabetes Prediction
@app.route('/diabetes', methods=['POST'])
def predict_diabetes():
    features = request.get_json().get('features', {})
    if not features:
        return redirect(url_for('result5'))
    input_features = [[
        features.get('Pregnancies', 0), features.get('Glucose', 0), features.get('BloodPressure', 0), features.get('SkinThickness', 0),
        features.get('Insulin', 0), features.get('BMI', 0), features.get('DiabetesPedigreeFunction', 0), features.get('Age', 0)
    ]]
    
    prediction = diabetes_model.predict(input_features)
    predicted_class = 'Diabetic' if prediction[0] > 0.5 else 'Not Diabetic'
    confidence = float(prediction[0]) * 100
    
    # Store results in session
    session['result'] = {
        'predicted_class': predicted_class,
        'confidence': confidence
    }
    print(f"Brain Tumor Prediction: {predicted_class}, Confidence: {confidence:.2f}%")
    
    return redirect(url_for('result5'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/result1')
def result1():
    result = session.get('result', {})
    return render_template('result1.html', result=result)


@app.route('/result2')
def result2():
    result = session.get('result', {})
    return render_template('result2.html', result=result)


@app.route('/result3')
def result3():
    result = session.get('result', {})
    return render_template('result3.html', result=result)


@app.route('/result4')
def result4():
    result = session.get('result', {})
    return render_template('result4.html', result=result)


@app.route('/result5')
def result5():
    result = session.get('result', {})
    return render_template('result5.html', result=result)


@app.route('/result6')
def result6():
    result = session.get('result', {})
    return render_template('result6.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)
