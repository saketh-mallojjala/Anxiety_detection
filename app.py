


from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import joblib

app = Flask(__name__)

# Load the trained model
model = load_model('final_model_federated.h5')

# Health suggestions based on anxiety detection
anxiety_suggestions = [
    "Practice deep breathing exercises for 5 minutes several times a day",
    "Try progressive muscle relaxation techniques",
    "Maintain a regular sleep schedule",
    "Limit caffeine and alcohol consumption",
    "Consider mindfulness meditation",
    "Stay physically active with regular exercise",
    "Connect with supportive friends or family members",
    "Journal about your thoughts and feelings"
]

general_wellness = [
    "Stay hydrated throughout the day",
    "Eat a balanced diet rich in fruits and vegetables",
    "Take regular breaks from screens and technology",
    "Spend time in nature when possible",
    "Practice gratitude by noting things you're thankful for",
    "Maintain a consistent sleep schedule",
    "Consider limiting news consumption if it increases stress"
]


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    input1 = float(request.form['input1'])
    input2 = float(request.form['input2'])
    input3 = float(request.form['input3'])
    # scaler = MinMaxScaler()
    scaler = joblib.load('scaler.save')
    # # Prepare the input array
    new_input = np.array([[input1, input2, input3]])


    new_input_scaled = scaler.transform(new_input)

    prediction = model.predict(new_input_scaled)

    predicted_class = int(prediction[0][0] >= 0.5)

    print("predicted class ")

    print(predicted_class)

    output = predicted_class

    if output == 1:
        result = "Anxiety Detected"
        status = "warning"
        message = "Based on your inputs, we've detected signs of anxiety. Consider the following suggestions:"
        suggestions = anxiety_suggestions
    else:
        result = "No Anxiety Detected"
        status = "success"
        message = "Great! Your inputs don't indicate anxiety. Here are some general wellness tips:"
        suggestions = general_wellness

    return render_template('index.html',
                           prediction_text=result,
                           status=status,
                           message=message,
                           suggestions=suggestions)


if __name__ == "__main__":
    app.run(debug=True)
