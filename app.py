from flask import Flask, render_template, request
import numpy as np
from PIL import Image

import pickle

app = Flask(__name__, template_folder='templates')

# Load all models during application initialization
diabetes_model = pickle.load(open('models/diabetes.pkl', 'rb'))
breast_cancer_model = pickle.load(open('models/breast_cancer.pkl', 'rb'))
heart_model = pickle.load(open('models/heart.pkl', 'rb'))
kidney_model = pickle.load(open('models/kidney.pkl', 'rb'))
liver_model = pickle.load(open('models/liver.pkl', 'rb'))

malaria_model = load_model("models/malaria.h5")
pneumonia_model = load_model("models/pneumonia.h5")

def predict(values):
    try:
        values = np.asarray(values)
        if len(values) == 8:
            return diabetes_model.predict(values.reshape(1, -1))[0]
        elif len(values) == 26:
            return breast_cancer_model.predict(values.reshape(1, -1))[0]
        elif len(values) == 13:
            return heart_model.predict(values.reshape(1, -1))[0]
        elif len(values) == 18:
            return kidney_model.predict(values.reshape(1, -1))[0]
        elif len(values) == 10:
            return liver_model.predict(values.reshape(1, -1))[0]
    except Exception as e:
        return f"Error in prediction: {e}"

@app.route("/")
def home():
    return render_template('home.html')

@app.route("/diabetes", methods=['GET', 'POST'])
def diabetesPage():
    return render_template('diabetes.html')

@app.route("/cancer", methods=['GET', 'POST'])
def cancerPage():
    return render_template('breast_cancer.html')

@app.route("/heart", methods=['GET', 'POST'])
def heartPage():
    return render_template('heart.html')

@app.route("/kidney", methods=['GET', 'POST'])
def kidneyPage():
    return render_template('kidney.html')

@app.route("/liver", methods=['GET', 'POST'])
def liverPage():
    return render_template('liver.html')

@app.route("/malaria", methods=['GET', 'POST'])
def malariaPage():
    return render_template('malaria.html')

@app.route("/pneumonia", methods=['GET', 'POST'])
def pneumoniaPage():
    return render_template('pneumonia.html')

@app.route("/predict", methods=['POST'])
def predictPage():
    try:
        to_predict_dict = request.form.to_dict()
        to_predict_list = list(map(float, list(to_predict_dict.values())))
        pred = predict(to_predict_list)
        return render_template('predict.html', pred=pred)
    except Exception as e:
        return render_template("home.html", message=f"Please enter valid data: {e}")

@app.route("/malariapredict", methods=['POST'])
def malariapredictPage():
    try:
        if 'image' in request.files:
            img = Image.open(request.files['image'])
            img = img.resize((36, 36))
            img = np.asarray(img)
            img = img.reshape((1, 36, 36, 3))
            img = img.astype(np.float64)
            pred = np.argmax(malaria_model.predict(img)[0])
            return render_template('malaria_predict.html', pred=pred)
    except Exception as e:
        return render_template('malaria.html', message=f"Error processing image: {e}")

@app.route("/pneumoniapredict", methods=['POST'])
def pneumoniapredictPage():
    try:
        if 'image' in request.files:
            img = Image.open(request.files['image']).convert('L')
            img = img.resize((36, 36))
            img = np.asarray(img)
            img = img.reshape((1, 36, 36, 1))
            img = img / 255.0
            pred = np.argmax(pneumonia_model.predict(img)[0])
            return render_template('pneumonia_predict.html', pred=pred)
    except Exception as e:
        return render_template('pneumonia.html', message=f"Error processing image: {e}")

if __name__ == '__main__':
    app.run(debug=True)
